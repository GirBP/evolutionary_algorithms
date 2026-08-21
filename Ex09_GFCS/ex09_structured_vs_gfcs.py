#!/usr/bin/env python3
"""
Ex09: SOTA Structured Pruning vs Our Pipeline (Unstructured + GFCS)
====================================================================
Comparison on 8 datasets × 2 seeds.

SOTA Structured methods (produce compact model directly):
  1. Structured Magnitude Pruning (SMP) — L1-norm neuron removal
  2. Taylor Pruning — gradient-based neuron importance
  3. Network Slimming — L1-regularized training + prune

Our pipeline:
  4. Unstructured Magnitude → GFCS conversion

All measured with RCU profiling.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, json, time, threading, copy
import torch
import torch.nn as nn
import numpy as np

torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP, CompactMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
)
from ex09_lib.gfcs import gfcs_convert

# ═══ RCU Anchor ═══
_tls = threading.local()

def _get_anchor():
    if not hasattr(_tls, "init"):
        _tls.a = np.random.rand(1024)
        _tls.b = np.random.rand(1024)
        _tls.c = np.empty_like(_tls.a)
        _tls.loops = 50
        while True:
            t = _anchor_run(_tls.a, _tls.b, _tls.c, _tls.loops)
            if t >= 5_000_000: break
            _tls.loops *= 2
        _tls.init = True
    return _tls.a, _tls.b, _tls.c, _tls.loops

def _anchor_run(a, b, c, loops):
    s = time.thread_time_ns()
    for _ in range(loops):
        np.multiply(a, b, out=c)
        np.add(c, a, out=c)
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
    rcu = t / max((pre + post) / 2.0, 1)
    return result, rcu

def measure_inference_rcu(model, test_dl, n=100):
    model.eval()
    X = next(iter(test_dl))[0]
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
#  Method 1: Structured Magnitude Pruning (SMP)
#  Remove neurons with smallest L1-norm of incoming weights
# ═══════════════════════════════════════════
def structured_magnitude_pruning(teacher, target_ratio, n_classes):
    """Remove neurons with smallest L1-norm. target_ratio = fraction to KEEP."""
    model = copy.deepcopy(teacher)
    layers = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            layers.append((name, m))

    # For hidden layers (not input, not output)
    hiddens = []
    for i in range(1, len(layers) - 1):
        name, layer = layers[i]
        n_neurons = layer.out_features if i < len(layers) - 1 else layer.in_features
        # L1 norm of incoming weights per neuron
        importance = layer.weight.data.abs().sum(dim=1)  # [out_features]
        k = max(4, int(len(importance) * target_ratio))
        _, keep_idx = torch.topk(importance, k)
        keep_idx = keep_idx.sort().values
        hiddens.append((i, keep_idx, k))

    # Build compact model by extracting kept neurons
    hidden_sizes = []
    state = model.state_dict()

    # Layer 0 (input→hidden1): keep rows
    W0 = state['fc1.weight']
    b0 = state['fc1.bias']
    idx0 = hiddens[0][1]
    W0_new = W0[idx0]
    b0_new = b0[idx0]
    hidden_sizes.append(len(idx0))

    # Layer 1 (hidden1→hidden2): keep rows and cols
    W1 = state['fc2.weight']
    b1 = state['fc2.bias']
    idx1 = hiddens[1][1]
    W1_new = W1[idx1][:, idx0]
    b1_new = b1[idx1]
    hidden_sizes.append(len(idx1))

    # Layer 2 (hidden2→hidden3): keep rows and cols
    W2 = state['fc3.weight']
    b2 = state['fc3.bias']
    idx2 = hiddens[2][1]
    W2_new = W2[idx2][:, idx1]
    b2_new = b2[idx2]
    hidden_sizes.append(len(idx2))

    # Output layer: keep cols
    W3 = state['fc4.weight']
    b3 = state['fc4.bias']
    W3_new = W3[:, idx2]

    # Create compact model
    input_dim = W0.shape[1]
    compact = CompactMLP(input_dim=input_dim, hiddens=hidden_sizes, n_classes=n_classes)
    sd = compact.state_dict()
    keys = list(sd.keys())
    # net.0.weight, net.0.bias, net.2.weight, net.2.bias, ...
    new_weights = [W0_new, b0_new, W1_new, b1_new, W2_new, b2_new, W3_new, b3]
    for k, w in zip(keys, new_weights):
        sd[k] = w
    compact.load_state_dict(sd)
    return compact


# ═══════════════════════════════════════════
#  Method 2: Taylor Pruning
#  importance = |∂L/∂h · h| (first-order Taylor expansion)
# ═══════════════════════════════════════════
def taylor_pruning(teacher, train_dl, target_ratio, n_classes):
    """Prune neurons based on Taylor importance = |grad * activation|."""
    model = copy.deepcopy(teacher)
    model.train()

    # Collect activations and gradients
    activations = {}
    gradients = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            activations[name] = out
            out.retain_grad()
        return hook_fn

    hooks = []
    hooks.append(model.fc1.register_forward_hook(make_hook('fc1')))
    hooks.append(model.fc2.register_forward_hook(make_hook('fc2')))
    hooks.append(model.fc3.register_forward_hook(make_hook('fc3')))

    # Accumulate importance over batches
    importance = {'fc1': None, 'fc2': None, 'fc3': None}
    n_batches = 0

    criterion = nn.CrossEntropyLoss()
    for X, y in train_dl:
        model.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()

        for name in ['fc1', 'fc2', 'fc3']:
            act = activations[name]
            grad = act.grad
            if grad is not None:
                taylor = (grad * act).abs().mean(dim=0)  # per-neuron importance
                if importance[name] is None:
                    importance[name] = taylor.detach()
                else:
                    importance[name] += taylor.detach()
        n_batches += 1

    for h in hooks:
        h.remove()

    # Normalize
    for name in importance:
        if importance[name] is not None:
            importance[name] /= n_batches

    # Select top-k neurons per layer
    state = model.state_dict()
    idx0 = torch.topk(importance['fc1'], max(4, int(128 * target_ratio))).indices.sort().values
    idx1 = torch.topk(importance['fc2'], max(4, int(128 * target_ratio))).indices.sort().values
    idx2 = torch.topk(importance['fc3'], max(4, int(128 * target_ratio))).indices.sort().values

    hidden_sizes = [len(idx0), len(idx1), len(idx2)]
    input_dim = state['fc1.weight'].shape[1]

    W0_new = state['fc1.weight'][idx0]
    b0_new = state['fc1.bias'][idx0]
    W1_new = state['fc2.weight'][idx1][:, idx0]
    b1_new = state['fc2.bias'][idx1]
    W2_new = state['fc3.weight'][idx2][:, idx1]
    b2_new = state['fc3.bias'][idx2]
    W3_new = state['fc4.weight'][:, idx2]
    b3 = state['fc4.bias']

    compact = CompactMLP(input_dim=input_dim, hiddens=hidden_sizes, n_classes=n_classes)
    sd = compact.state_dict()
    keys = list(sd.keys())
    new_weights = [W0_new, b0_new, W1_new, b1_new, W2_new, b2_new, W3_new, b3]
    for k, w in zip(keys, new_weights):
        sd[k] = w
    compact.load_state_dict(sd)
    return compact


# ═══════════════════════════════════════════
#  Method 3: Network Slimming (adapted for MLP)
#  Train with L1 penalty on neuron scale → prune low-scale neurons
# ═══════════════════════════════════════════
def network_slimming(teacher, train_dl, target_ratio, n_classes, reg_epochs=50, lam=0.01):
    """Train with L1 penalty on learnable neuron scales, then prune."""
    model = copy.deepcopy(teacher)
    input_dim = model.fc1.weight.shape[1]

    # Add learnable scale parameters (BN γ substitute)
    scale1 = nn.Parameter(torch.ones(128))
    scale2 = nn.Parameter(torch.ones(128))
    scale3 = nn.Parameter(torch.ones(128))

    optimizer = torch.optim.SGD(
        list(model.parameters()) + [scale1, scale2, scale3],
        lr=0.01, momentum=0.9
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(reg_epochs):
        for X, y in train_dl:
            optimizer.zero_grad()
            # Forward with scales
            h1 = torch.relu(model.fc1(X)) * scale1
            h2 = torch.relu(model.fc2(h1)) * scale2
            h3 = torch.relu(model.fc3(h2)) * scale3
            out = model.fc4(h3)

            loss = criterion(out, y)
            # L1 penalty on scales
            l1_penalty = lam * (scale1.abs().sum() + scale2.abs().sum() + scale3.abs().sum())
            total_loss = loss + l1_penalty
            total_loss.backward()
            optimizer.step()

    # Prune: keep top-k by scale magnitude
    state = model.state_dict()
    idx0 = torch.topk(scale1.abs(), max(4, int(128 * target_ratio))).indices.sort().values
    idx1 = torch.topk(scale2.abs(), max(4, int(128 * target_ratio))).indices.sort().values
    idx2 = torch.topk(scale3.abs(), max(4, int(128 * target_ratio))).indices.sort().values

    hidden_sizes = [len(idx0), len(idx1), len(idx2)]

    # Extract and scale weights
    W0 = state['fc1.weight'][idx0] * scale1[idx0].unsqueeze(1).detach()
    b0 = state['fc1.bias'][idx0] * scale1[idx0].detach()
    W1 = state['fc2.weight'][idx1][:, idx0] * scale2[idx1].unsqueeze(1).detach()
    b1 = state['fc2.bias'][idx1] * scale2[idx1].detach()
    W2 = state['fc3.weight'][idx2][:, idx1] * scale3[idx2].unsqueeze(1).detach()
    b2 = state['fc3.bias'][idx2] * scale3[idx2].detach()
    W3 = state['fc4.weight'][:, idx2]
    b3 = state['fc4.bias']

    compact = CompactMLP(input_dim=input_dim, hiddens=hidden_sizes, n_classes=n_classes)
    sd = compact.state_dict()
    keys = list(sd.keys())
    new_weights = [W0, b0, W1, b1, W2, b2, W3, b3]
    for k, w in zip(keys, new_weights):
        sd[k] = w.detach()
    compact.load_state_dict(sd)
    return compact


# ═══════════════════════════════════════════
#  Method 4: Our pipeline (Unstructured + GFCS)
# ═══════════════════════════════════════════
def our_pipeline(teacher, n_classes, sparsity):
    """Unstructured magnitude pruning → GFCS conversion."""
    sparse = copy.deepcopy(teacher)
    prune_magnitude_global(sparse, sparsity)
    compact, info = gfcs_convert(sparse, n_classes, use_evolution=True)
    return compact


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════
DATASETS = ['moons', 'circles', 'spirals', 'blobs',
            'gaussian_quantiles', 'classification', 'highdim', 'sequence_cls']
SEEDS = [42, 123]
SPARSITIES = {
    'moons': 0.90, 'circles': 0.85, 'spirals': 0.90,
    'blobs': 0.75, 'gaussian_quantiles': 0.75,
    'classification': 0.75, 'highdim': 0.70, 'sequence_cls': 0.90,
}
# For structured: keep ratio = 1 - sparsity (approx same compression target)
KEEP_RATIOS = {ds: max(0.1, 1.0 - sp) for ds, sp in SPARSITIES.items()}

METHODS = [
    ('SMP',             lambda t, dl, nc, ds: structured_magnitude_pruning(t, KEEP_RATIOS[ds], nc)),
    ('Taylor',          lambda t, dl, nc, ds: taylor_pruning(t, dl, KEEP_RATIOS[ds], nc)),
    ('NetSlimming',     lambda t, dl, nc, ds: network_slimming(t, dl, KEEP_RATIOS[ds], nc, reg_epochs=50)),
    ('Unstr+GFCS',      lambda t, dl, nc, ds: our_pipeline(t, nc, SPARSITIES[ds])),
]


def main():
    print("Calibrating anchor...")
    for _ in range(5): get_anchor_time()
    print(f"  Anchor: {get_anchor_time()/1e6:.2f}ms\n")

    all_results = []

    for ds in DATASETS:
        for seed in SEEDS:
            set_seed(seed)
            train_dl, _, test_dl, n_classes = get_dataloaders(seed, ds, 64)
            input_dim = next(iter(test_dl))[0].shape[1]

            # Teacher
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            tp = f"data/{ds}/teacher_seed{seed}.pt"
            if os.path.exists(tp):
                teacher.load_state_dict(torch.load(tp, weights_only=True))
            else:
                os.makedirs(f"data/{ds}", exist_ok=True)
                train_model(teacher, train_dl, epochs=100)
                torch.save(teacher.state_dict(), tp)

            _, teacher_f1, _ = evaluate(teacher, test_dl)
            teacher_params = teacher.count_params()
            teacher_infer_rcu = measure_inference_rcu(teacher, test_dl)

            # Sparse baseline F1 (for our pipeline comparison)
            sparse_tmp = copy.deepcopy(teacher)
            prune_magnitude_global(sparse_tmp, SPARSITIES[ds])
            _, sparse_f1, _ = evaluate(sparse_tmp, test_dl)

            print(f"{'═'*70}")
            print(f"  {ds}, seed={seed}  Teacher F1={teacher_f1:.4f}  Sparse F1={sparse_f1:.4f}")

            for method_name, method_fn in METHODS:
                set_seed(seed)
                try:
                    # Pruning RCU
                    compact, rcu_prune = profile_rcu(method_fn, teacher, train_dl, n_classes, ds)

                    _, pre_f1, _ = evaluate(compact, test_dl)

                    # Finetune RCU
                    set_seed(seed)
                    _, rcu_ft = profile_rcu(train_model, compact, train_dl, epochs=10, lr=0.01)

                    _, final_f1, _ = evaluate(compact, test_dl)
                    compact_params = compact.count_params()
                    compression = teacher_params / compact_params if compact_params > 0 else 0
                    compact_infer_rcu = measure_inference_rcu(compact, test_dl)
                    infer_speedup = teacher_infer_rcu / compact_infer_rcu if compact_infer_rcu > 0 else 0

                    delta_f1_teacher = final_f1 - teacher_f1
                    delta_f1_sparse = final_f1 - sparse_f1

                    result = {
                        'dataset': ds, 'seed': seed, 'method': method_name,
                        'teacher_f1': round(teacher_f1, 4),
                        'sparse_f1': round(sparse_f1, 4),
                        'final_f1': round(final_f1, 4),
                        'delta_f1_teacher': round(delta_f1_teacher, 4),
                        'delta_f1_sparse': round(delta_f1_sparse, 4),
                        'teacher_params': teacher_params,
                        'compact_params': compact_params,
                        'compression': round(compression, 2),
                        'infer_speedup': round(infer_speedup, 2),
                        'rcu_prune': round(rcu_prune, 3),
                        'rcu_finetune': round(rcu_ft, 3),
                        'rcu_total': round(rcu_prune + rcu_ft, 3),
                    }
                    all_results.append(result)

                    q = '✅' if delta_f1_teacher >= -0.05 else '❌'
                    print(f"  {q} {method_name:15s} F1={final_f1:.4f} (Δteacher={delta_f1_teacher:+.4f}) "
                          f"comp={compression:.1f}× infer={infer_speedup:.2f}× "
                          f"RCU_prune={rcu_prune:.1f} RCU_tot={rcu_prune+rcu_ft:.1f}")

                except Exception as e:
                    print(f"  ❌ {method_name:15s} FAILED: {e}")
                    import traceback; traceback.print_exc()

    # ═══ Summary ═══
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        m = r['method']
        for k in ['final_f1', 'delta_f1_teacher', 'compression', 'infer_speedup',
                   'rcu_prune', 'rcu_total']:
            agg[m][k].append(r[k])

    print(f"\n\n{'='*100}")
    print("  STRUCTURED PRUNING vs UNSTRUCTURED+GFCS — Averaged across 8 datasets × 2 seeds")
    print(f"{'='*100}\n")

    print(f"{'Method':15s} {'Final F1':>9s} {'ΔF1 teach':>10s} {'Comp×':>6s} {'Infer×':>7s} "
          f"{'RCU_prune':>10s} {'RCU_total':>10s}   Type")
    print("─" * 100)

    method_names = ['SMP', 'Taylor', 'NetSlimming', 'Unstr+GFCS']
    types = {'SMP': 'Structured', 'Taylor': 'Structured', 'NetSlimming': 'Structured',
             'Unstr+GFCS': 'Unstr+Convert'}

    for m in method_names:
        if m not in agg: continue
        d = agg[m]
        f1 = np.mean(d['final_f1'])
        df1 = np.mean(d['delta_f1_teacher'])
        comp = np.mean(d['compression'])
        inf = np.mean(d['infer_speedup'])
        rp = np.mean(d['rcu_prune'])
        rt = np.mean(d['rcu_total'])
        tag = ' <<<' if m == 'Unstr+GFCS' else ''
        print(f"{m:15s} {f1:9.4f} {df1:+10.4f} {comp:6.1f} {inf:7.2f} "
              f"{rp:10.1f} {rt:10.1f}   {types[m]}{tag}")

    # Per-dataset
    print(f"\n\n{'='*100}")
    print("  PER-DATASET: Final F1 (averaged over seeds)")
    print(f"{'='*100}\n")

    print(f"{'Dataset':20s}", end='')
    for m in method_names:
        print(f"{m:>15s}", end='')
    print()
    print("─" * 80)

    for ds in DATASETS:
        print(f"{ds:20s}", end='')
        for m in method_names:
            vals = [r['final_f1'] for r in all_results if r['dataset'] == ds and r['method'] == m]
            if vals:
                avg = np.mean(vals)
                # Is this the best for this dataset?
                all_f1 = {mn: np.mean([r['final_f1'] for r in all_results
                          if r['dataset']==ds and r['method']==mn])
                          for mn in method_names
                          if any(r['dataset']==ds and r['method']==mn for r in all_results)}
                best = max(all_f1.values())
                mark = ' *' if abs(avg - best) < 0.001 else '  '
                print(f"{avg:13.4f}{mark}", end='')
            else:
                print(f"{'N/A':>15s}", end='')
        print()

    print("\n  * = best or tied for best")

    with open('results/structured_vs_gfcs.json', 'w') as f:
        json.dump({'results': all_results}, f, indent=2)
    print(f"\n  Saved: results/structured_vs_gfcs.json")


if __name__ == '__main__':
    main()
