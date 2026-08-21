#!/usr/bin/env python3
"""
Ex09: Multi-Architecture Benchmark
====================================
Structured Pruning vs Unstructured+GFCS on 4 architectures × 4 hard datasets × 2 seeds.

Architectures:
  A: SimpleMLP     3 hidden (128-128-128)         ~34K params
  B: DeepMLP       5 hidden (128-128-128-128-128)  ~83K params
  C: WideMLP       3 hidden (256-256-256)          ~134K params
  D: BottleneckMLP 3 hidden (256-64-256)           ~83K params

Methods:
  1. Taylor Pruning (structured)
  2. Network Slimming (structured)
  3. Unstructured + GFCS (ours)

All measured with RCU.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, json, time, threading, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex09_lib.core import (
    set_seed, get_dataloaders, CompactMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
)

# ═══ RCU Anchor ═══
_tls = threading.local()
def _get_anchor():
    if not hasattr(_tls, "init"):
        _tls.a = np.random.rand(1024); _tls.b = np.random.rand(1024)
        _tls.c = np.empty_like(_tls.a); _tls.loops = 50
        while _anchor_run(_tls.a, _tls.b, _tls.c, _tls.loops) < 5_000_000:
            _tls.loops *= 2
        _tls.init = True
    return _tls.a, _tls.b, _tls.c, _tls.loops

def _anchor_run(a, b, c, loops):
    s = time.thread_time_ns()
    for _ in range(loops): np.multiply(a, b, out=c); np.add(c, a, out=c)
    return time.thread_time_ns() - s

def get_anchor_time():
    a, b, c, loops = _get_anchor()
    return _anchor_run(a, b, c, loops)

def profile_rcu(func, *args, **kwargs):
    pre = get_anchor_time()
    s = time.thread_time_ns()
    result = func(*args, **kwargs)
    t = time.thread_time_ns() - s
    post = get_anchor_time()
    return result, t / max((pre + post) / 2.0, 1)

def measure_inference_rcu(model, test_dl, n=100):
    model.eval(); X = next(iter(test_dl))[0]
    with torch.no_grad():
        for _ in range(10): model(X)
    pre = get_anchor_time()
    s = time.thread_time_ns()
    with torch.no_grad():
        for _ in range(n): model(X)
    t = time.thread_time_ns() - s
    post = get_anchor_time()
    return (t / max((pre + post) / 2.0, 1)) / n


# ═══════════════════════════════════════════
#  Generic MLP: works for any number of hidden layers
# ═══════════════════════════════════════════
class GenericMLP(nn.Module):
    """MLP with arbitrary hidden layer sizes."""
    def __init__(self, input_dim, hidden_sizes, n_classes):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_sizes = list(hidden_sizes)

        dims = [input_dim] + self.hidden_sizes + [n_classes]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_nonzero(self):
        return sum((p.data != 0).sum().item() for p in self.parameters())

    def get_hidden_layers(self):
        """Return list of hidden Linear layers (excluding output)."""
        return list(self.layers[:-1])

    def get_output_layer(self):
        return self.layers[-1]


# ═══════════════════════════════════════════
#  Generic GFCS (works on any GenericMLP)
# ═══════════════════════════════════════════
def _compute_flow_importance(W_in, W_out):
    in_flow = W_in.abs().sum(dim=1)
    out_flow = W_out.abs().sum(dim=0)
    return in_flow * out_flow

def _compute_flow_affinity(W_in, W_out):
    n = W_in.shape[0]
    A = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            in_overlap = torch.min(W_in[i].abs(), W_in[j].abs()).sum()
            out_overlap = torch.min(W_out[:, i].abs(), W_out[:, j].abs()).sum()
            A[i, j] = A[j, i] = in_overlap * out_overlap
    return A

def _greedy_flow_merge(W_in, b_in, W_out, phi, affinity, target_k):
    n = W_in.shape[0]
    active = list(range(n))
    W_in = W_in.clone()
    b_in = b_in.clone() if b_in is not None else None
    W_out = W_out.clone()
    phi = phi.clone()

    while len(active) > target_k:
        min_idx = min(active, key=lambda i: phi[i].item())
        active_no_min = [j for j in active if j != min_idx]
        if not active_no_min: break

        best_partner = max(active_no_min, key=lambda j: affinity[min_idx, j].item())

        pi, pj = phi[min_idx], phi[best_partner]
        total = pi + pj + 1e-12
        W_in[best_partner] = (pi * W_in[min_idx] + pj * W_in[best_partner]) / total
        if b_in is not None:
            b_in[best_partner] = (pi * b_in[min_idx] + pj * b_in[best_partner]) / total
        W_out[:, best_partner] = W_out[:, min_idx] + W_out[:, best_partner]
        phi[best_partner] = pi + pj
        active.remove(min_idx)

    active_idx = sorted(active)
    return W_in[active_idx], (b_in[active_idx] if b_in is not None else None), W_out[:, active_idx], active_idx

def generic_gfcs_convert(sparse_model, n_classes, use_evolution=True,
                         pop_size=20, generations=30, min_ratio=0.1, max_ratio=0.8):
    """GFCS conversion for any GenericMLP."""
    hidden_layers = sparse_model.get_hidden_layers()
    output_layer = sparse_model.get_output_layer()
    all_layers = list(hidden_layers) + [output_layer]
    n_hidden = len(hidden_layers)

    # Determine keep ratios
    if use_evolution:
        ratios = _generic_evolve(sparse_model, n_classes, pop_size, generations, min_ratio, max_ratio)
    else:
        ratios = []
        for i, layer in enumerate(hidden_layers):
            W = layer.weight.data
            alive = (W.abs().sum(dim=1) > 1e-8)
            if i + 1 < len(all_layers):
                out_alive = (all_layers[i+1].weight.data.abs().sum(dim=0) > 1e-8)
                alive = alive & out_alive
            density = max(alive.float().mean().item(), 0.05)
            ratios.append(max(min_ratio, min(max_ratio, density)))

    # Merge per layer
    merged = []
    for i, layer in enumerate(hidden_layers):
        W_in = layer.weight.data
        b_in = layer.bias.data if layer.bias is not None else None
        W_out = all_layers[i + 1].weight.data

        alive = (W_in.abs().sum(dim=1) > 1e-8)
        if i + 1 < len(all_layers):
            out_alive = (W_out.abs().sum(dim=0) > 1e-8)
            alive = alive & out_alive
        n_active = max(alive.sum().item(), 4)
        target_k = max(4, int(n_active * ratios[i]))

        phi = _compute_flow_importance(W_in, W_out)
        aff = _compute_flow_affinity(W_in, W_out)
        new_Win, new_bin, new_Wout, idx = _greedy_flow_merge(W_in, b_in, W_out, phi, aff, target_k)
        merged.append((new_Win, new_bin, new_Wout, idx))

    # Build compact
    hiddens = [m[0].shape[0] for m in merged]
    compact = CompactMLP(input_dim=sparse_model.input_dim, hiddens=tuple(hiddens), n_classes=n_classes)

    with torch.no_grad():
        for i, (new_Win, new_bin, _, idx) in enumerate(merged):
            tgt = compact.net[i * 2]
            if i == 0:
                tgt.weight.data.copy_(new_Win)
            else:
                prev_idx = merged[i-1][3]
                tgt.weight.data.copy_(new_Win[:, prev_idx])
            if new_bin is not None:
                tgt.bias.data.copy_(new_bin)

        last_idx = merged[-1][3]
        tgt_out = compact.net[-1]
        tgt_out.weight.data.copy_(output_layer.weight.data[:, last_idx])
        if output_layer.bias is not None:
            tgt_out.bias.data.copy_(output_layer.bias.data)

    orig_p = sparse_model.count_params()
    comp_p = compact.count_params()
    info = {
        'method': 'gfcs', 'merged_hiddens': hiddens,
        'compression': orig_p / max(comp_p, 1), 'ratios': [float(r) for r in ratios],
    }
    return compact, info


def _generic_evolve(model, n_classes, pop_size, generations, min_r, max_r):
    """(μ+λ)-ES for generic MLP."""
    hidden_layers = model.get_hidden_layers()
    output_layer = model.get_output_layer()
    all_layers = list(hidden_layers) + [output_layer]
    n_hidden = len(hidden_layers)

    def fitness(ratios):
        total_flow = 0; kept_flow = 0; total_neurons = 0; kept_neurons = 0
        for i, layer in enumerate(hidden_layers):
            W_in = layer.weight.data
            W_out = all_layers[i+1].weight.data
            phi = _compute_flow_importance(W_in, W_out)
            alive = (W_in.abs().sum(dim=1) > 1e-8) & (W_out.abs().sum(dim=0) > 1e-8)
            n_active = max(alive.sum().item(), 4)
            k = max(4, int(n_active * ratios[i]))
            total_flow += phi.sum().item()
            _, topk_idx = phi.topk(min(k, len(phi)))
            kept_flow += phi[topk_idx].sum().item()
            total_neurons += n_active
            kept_neurons += k
        flow_preserved = kept_flow / max(total_flow, 1e-12)
        compress_reward = 1 - kept_neurons / max(total_neurons, 1)
        return flow_preserved + 0.5 * compress_reward

    # Initialize population
    pop = [np.random.uniform(min_r, max_r, n_hidden) for _ in range(pop_size)]
    sigma = 0.12
    best_g, best_f = None, -1e9

    for gen in range(generations):
        scored = [(g, fitness(g)) for g in pop]
        scored.sort(key=lambda x: x[1], reverse=True)
        if scored[0][1] > best_f:
            best_g, best_f = scored[0]

        mu = max(2, pop_size // 4)
        parents = [s[0] for s in scored[:mu]]
        new_pop = list(parents)
        sigma_t = sigma * (1 - gen / generations * 0.5)

        while len(new_pop) < pop_size:
            p = parents[np.random.randint(mu)]
            child = p + np.random.randn(n_hidden) * sigma_t
            child = np.clip(child, min_r, max_r)
            new_pop.append(child)
        pop = new_pop

    return best_g


# ═══════════════════════════════════════════
#  Generic Taylor Pruning
# ═══════════════════════════════════════════
def generic_taylor_pruning(teacher, train_dl, keep_ratio, n_classes):
    model = copy.deepcopy(teacher)
    model.train()
    hidden_layers = model.get_hidden_layers()

    activations = {}
    hooks = []
    for i, layer in enumerate(hidden_layers):
        name = f"h{i}"
        def make_hook(n):
            def hook_fn(mod, inp, out):
                activations[n] = out
                out.retain_grad()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(name)))

    importance = {f"h{i}": None for i in range(len(hidden_layers))}
    criterion = nn.CrossEntropyLoss()
    n_batches = 0

    for X, y in train_dl:
        model.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        for i in range(len(hidden_layers)):
            name = f"h{i}"
            act = activations[name]
            grad = act.grad
            if grad is not None:
                taylor = (grad * act).abs().mean(dim=0)
                importance[name] = taylor.detach() if importance[name] is None else importance[name] + taylor.detach()
        n_batches += 1

    for h in hooks: h.remove()
    for k in importance:
        if importance[k] is not None: importance[k] /= n_batches

    # Select neurons
    all_layers_list = list(hidden_layers) + [model.get_output_layer()]
    indices = []
    for i in range(len(hidden_layers)):
        n_neurons = hidden_layers[i].out_features
        k = max(4, int(n_neurons * keep_ratio))
        idx = torch.topk(importance[f"h{i}"], k).indices.sort().values
        indices.append(idx)

    # Build compact
    state = model.state_dict()
    layer_keys = []
    for name, _ in model.named_parameters():
        if name not in [k for k, _ in layer_keys]:
            base = name.rsplit('.', 1)[0]
            if base not in [k for k, _ in layer_keys]:
                layer_keys.append((base, None))

    # Direct weight extraction
    hidden_sizes = [len(idx) for idx in indices]
    compact = CompactMLP(input_dim=teacher.input_dim, hiddens=hidden_sizes, n_classes=n_classes)

    with torch.no_grad():
        for i, layer in enumerate(hidden_layers):
            W = layer.weight.data
            b = layer.bias.data if layer.bias is not None else None
            idx = indices[i]

            # Rows = output neurons to keep
            W_new = W[idx]
            # Cols = input neurons from previous layer
            if i > 0:
                prev_idx = indices[i-1]
                W_new = W_new[:, prev_idx]

            tgt = compact.net[i * 2]
            tgt.weight.data.copy_(W_new)
            if b is not None:
                tgt.bias.data.copy_(b[idx])

        # Output layer
        out_layer = model.get_output_layer()
        W_out = out_layer.weight.data[:, indices[-1]]
        tgt_out = compact.net[-1]
        tgt_out.weight.data.copy_(W_out)
        if out_layer.bias is not None:
            tgt_out.bias.data.copy_(out_layer.bias.data)

    return compact


# ═══════════════════════════════════════════
#  Generic Network Slimming
# ═══════════════════════════════════════════
def generic_net_slimming(teacher, train_dl, keep_ratio, n_classes, reg_epochs=50, lam=0.01):
    model = copy.deepcopy(teacher)
    hidden_layers = model.get_hidden_layers()

    # Learnable scales per hidden layer
    scales = [nn.Parameter(torch.ones(layer.out_features)) for layer in hidden_layers]

    optimizer = torch.optim.SGD(
        list(model.parameters()) + scales, lr=0.01, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(reg_epochs):
        for X, y in train_dl:
            optimizer.zero_grad()
            x = X
            for i, layer in enumerate(hidden_layers):
                x = F.relu(layer(x)) * scales[i]
            out = model.get_output_layer()(x)
            loss = criterion(out, y) + lam * sum(s.abs().sum() for s in scales)
            loss.backward()
            optimizer.step()

    # Prune by scale
    indices = []
    for i, scale in enumerate(scales):
        n = len(scale)
        k = max(4, int(n * keep_ratio))
        idx = torch.topk(scale.detach().abs(), k).indices.sort().values
        indices.append(idx)

    hidden_sizes = [len(idx) for idx in indices]
    compact = CompactMLP(input_dim=teacher.input_dim, hiddens=hidden_sizes, n_classes=n_classes)

    with torch.no_grad():
        state = model.state_dict()
        for i, layer in enumerate(hidden_layers):
            W = layer.weight.data
            b = layer.bias.data if layer.bias is not None else None
            idx = indices[i]
            s = scales[i].detach()

            W_new = W[idx] * s[idx].unsqueeze(1)
            if i > 0:
                prev_idx = indices[i-1]
                W_new = W_new[:, prev_idx]

            tgt = compact.net[i * 2]
            tgt.weight.data.copy_(W_new)
            if b is not None:
                tgt.bias.data.copy_(b[idx] * s[idx])

        out_layer = model.get_output_layer()
        W_out = out_layer.weight.data[:, indices[-1]]
        tgt_out = compact.net[-1]
        tgt_out.weight.data.copy_(W_out)
        if out_layer.bias is not None:
            tgt_out.bias.data.copy_(out_layer.bias.data)

    return compact


# ═══════════════════════════════════════════
#  Architectures
# ═══════════════════════════════════════════
ARCHITECTURES = {
    'SimpleMLP':     lambda dim, nc: GenericMLP(dim, [128, 128, 128], nc),
    'DeepMLP':       lambda dim, nc: GenericMLP(dim, [128, 128, 128, 128, 128], nc),
    'WideMLP':       lambda dim, nc: GenericMLP(dim, [256, 256, 256], nc),
    'BottleneckMLP': lambda dim, nc: GenericMLP(dim, [256, 64, 256], nc),
}

# Hard datasets only (where differences show)
DATASETS = ['spirals', 'gaussian_quantiles', 'highdim', 'sequence_cls']
SEEDS = [42, 123]
SPARSITY = 0.85  # Fixed for fair comparison


def main():
    print("Calibrating anchor...")
    for _ in range(5): get_anchor_time()
    print(f"  Anchor: {get_anchor_time()/1e6:.2f}ms\n")

    all_results = []
    keep_ratio = 1.0 - SPARSITY  # 0.15

    for arch_name, arch_fn in ARCHITECTURES.items():
        for ds in DATASETS:
            for seed in SEEDS:
                set_seed(seed)
                train_dl, _, test_dl, n_classes = get_dataloaders(seed, ds, 64)
                input_dim = next(iter(test_dl))[0].shape[1]

                # Train teacher
                teacher = arch_fn(input_dim, n_classes)
                set_seed(seed)
                train_model(teacher, train_dl, epochs=100)
                _, teacher_f1, _ = evaluate(teacher, test_dl)
                teacher_params = teacher.count_params()
                teacher_infer = measure_inference_rcu(teacher, test_dl)

                print(f"{'═'*70}")
                print(f"  {arch_name} | {ds} | seed={seed} | Teacher F1={teacher_f1:.4f} params={teacher_params:,}")

                methods = [
                    ('Taylor', lambda t, dl: generic_taylor_pruning(t, dl, keep_ratio, n_classes)),
                    ('NetSlimming', lambda t, dl: generic_net_slimming(t, dl, keep_ratio, n_classes, reg_epochs=50)),
                    ('Unstr+GFCS', lambda t, dl: _our_pipeline(t, n_classes, SPARSITY)),
                ]

                for method_name, method_fn in methods:
                    set_seed(seed)
                    try:
                        compact, rcu_prune = profile_rcu(method_fn, teacher, train_dl)
                        _, pre_f1, _ = evaluate(compact, test_dl)

                        set_seed(seed)
                        _, rcu_ft = profile_rcu(train_model, compact, train_dl, epochs=10, lr=0.01)

                        _, final_f1, _ = evaluate(compact, test_dl)
                        comp_params = compact.count_params()
                        compression = teacher_params / comp_params if comp_params > 0 else 0
                        comp_infer = measure_inference_rcu(compact, test_dl)
                        infer_speedup = teacher_infer / comp_infer if comp_infer > 0 else 0
                        delta_teacher = final_f1 - teacher_f1

                        result = {
                            'arch': arch_name, 'dataset': ds, 'seed': seed,
                            'method': method_name,
                            'teacher_f1': round(teacher_f1, 4),
                            'final_f1': round(final_f1, 4),
                            'delta_f1': round(delta_teacher, 4),
                            'teacher_params': teacher_params,
                            'compact_params': comp_params,
                            'compression': round(compression, 2),
                            'infer_speedup': round(infer_speedup, 2),
                            'rcu_prune': round(rcu_prune, 2),
                            'rcu_total': round(rcu_prune + rcu_ft, 2),
                        }
                        all_results.append(result)

                        q = '✅' if delta_teacher >= -0.05 else '❌'
                        print(f"  {q} {method_name:15s} F1={final_f1:.4f} (Δ={delta_teacher:+.4f}) "
                              f"comp={compression:.1f}× infer={infer_speedup:.2f}× "
                              f"RCU={rcu_prune:.1f}")

                    except Exception as e:
                        print(f"  ❌ {method_name:15s} FAILED: {e}")

    # ═══ Summary ═══
    from collections import defaultdict

    print(f"\n\n{'='*110}")
    print("  MULTI-ARCHITECTURE RESULTS (averaged over seeds)")
    print(f"{'='*110}\n")

    for arch_name in ARCHITECTURES:
        print(f"\n  --- {arch_name} ---")
        print(f"  {'Dataset':20s} {'Method':15s} {'F1':>6s} {'ΔF1':>7s} {'Comp×':>6s} {'Infer×':>7s} {'RCU_prune':>10s}")
        print("  " + "─" * 80)

        for ds in DATASETS:
            for mi, mn in enumerate(['Taylor', 'NetSlimming', 'Unstr+GFCS']):
                vals = [r for r in all_results if r['arch']==arch_name and r['dataset']==ds and r['method']==mn]
                if not vals: continue
                f1 = np.mean([v['final_f1'] for v in vals])
                df = np.mean([v['delta_f1'] for v in vals])
                co = np.mean([v['compression'] for v in vals])
                inf = np.mean([v['infer_speedup'] for v in vals])
                rp = np.mean([v['rcu_prune'] for v in vals])
                ds_label = ds if mi == 0 else ''
                tag = ' <<<' if mn == 'Unstr+GFCS' else ''
                print(f"  {ds_label:20s} {mn:15s} {f1:.4f} {df:+.4f} {co:6.1f} {inf:7.2f} {rp:10.1f}{tag}")

    # Cross-architecture summary
    print(f"\n\n{'='*90}")
    print("  CROSS-ARCHITECTURE: GFCS wins per architecture (best F1)")
    print(f"{'='*90}\n")

    for arch_name in ARCHITECTURES:
        wins = 0
        for ds in DATASETS:
            best_f1 = -1
            best_m = ''
            for mn in ['Taylor', 'NetSlimming', 'Unstr+GFCS']:
                vals = [r['final_f1'] for r in all_results
                        if r['arch']==arch_name and r['dataset']==ds and r['method']==mn]
                if vals:
                    avg = np.mean(vals)
                    if avg > best_f1:
                        best_f1 = avg; best_m = mn
            if best_m == 'Unstr+GFCS': wins += 1
        print(f"  {arch_name:20s} GFCS wins: {wins}/{len(DATASETS)}")

    with open('results/multi_arch_benchmark.json', 'w') as f:
        json.dump({'results': all_results}, f, indent=2)
    print(f"\n  Saved: results/multi_arch_benchmark.json")


def _our_pipeline(teacher, n_classes, sparsity):
    sparse = copy.deepcopy(teacher)
    prune_magnitude_global(sparse, sparsity)
    compact, info = generic_gfcs_convert(sparse, n_classes, use_evolution=True)
    return compact


if __name__ == '__main__':
    main()
