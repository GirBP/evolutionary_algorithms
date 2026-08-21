#!/usr/bin/env python3
"""
Ex09v2: Extended Compression Benchmark
========================================
GFCS baseline + NeuronRemoval vs 3 best autosearch compression methods:
  - explore25: Evolutionary Adaptive Importance Blending (EAIB)
  - explore48: Evolutionary Prioritized Subspace Selection (EPSS)
  - explore07: Multi-Objective EA (MOEA)

All under identical Ex09 conditions:
  - SimpleMLP (128-128-128), 8 datasets × 2 seeds
  - Same pruning, finetune, RCU protocol

Metrics:
  1. Compression ratio (teacher_params / compact_params)
  2. Inference speedup (RCU_teacher / RCU_compact, 100 forward passes)
  3. RCU_method (cost of compression algorithm itself)
  4. F1 before finetune, F1 after finetune, RPR (recovery vs teacher)
"""

# ═══ MANDATORY: Disable hidden C++ threading BEFORE any imports ═══
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, json, time, threading, math, copy
import torch
torch.set_num_threads(1)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP, CompactMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
    convert_neuron_removal,
)
from ex09_lib.gfcs import gfcs_convert


# ═══════════════════════════════════════════
#  RCU Anchor (per RCU_protocol.md §4)
# ═══════════════════════════════════════════
_tls = threading.local()

def _get_local_anchor():
    if not hasattr(_tls, "initialized"):
        _tls.a = np.random.rand(1024)
        _tls.b = np.random.rand(1024)
        _tls.c = np.empty_like(_tls.a)
        _tls.loops = 50
        while True:
            t = _run_anchor_loop(_tls.a, _tls.b, _tls.c, _tls.loops)
            if t >= 5_000_000:
                break
            _tls.loops *= 2
        _tls.initialized = True
    return _tls.a, _tls.b, _tls.c, _tls.loops

def _run_anchor_loop(a, b, c, loops):
    start = time.thread_time_ns()
    for _ in range(loops):
        np.multiply(a, b, out=c)
        np.add(c, a, out=c)
    return time.thread_time_ns() - start

def get_anchor_time():
    a, b, c, loops = _get_local_anchor()
    return _run_anchor_loop(a, b, c, loops)

def profile_rcu(func, *args, **kwargs):
    anchor_pre = get_anchor_time()
    start_algo = time.thread_time_ns()
    result = func(*args, **kwargs)
    t_algo = time.thread_time_ns() - start_algo
    anchor_post = get_anchor_time()
    t_anchor_avg = (anchor_pre + anchor_post) / 2.0
    rcu = t_algo / max(t_anchor_avg, 1)
    return result, rcu, t_algo

def measure_inference_rcu(model, test_dl, n_repeats=100):
    model.eval()
    X_batch = next(iter(test_dl))[0]
    with torch.no_grad():
        for _ in range(10):
            _ = model(X_batch)
    anchor_pre = get_anchor_time()
    start = time.thread_time_ns()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = model(X_batch)
    t_algo = time.thread_time_ns() - start
    anchor_post = get_anchor_time()
    t_anchor_avg = (anchor_pre + anchor_post) / 2.0
    rcu_total = t_algo / max(t_anchor_avg, 1)
    return rcu_total / n_repeats


# ═══════════════════════════════════════════
#  ADAPTER: SimpleMLP → layer_info format
#  (same format as dense_autosearch compress_fn expects)
# ═══════════════════════════════════════════

def _get_layer_info_simplemlp(sparse_model):
    """Extract per-layer info from SimpleMLP in compress_fn format.

    SimpleMLP has fc1→fc2→fc3→fc4 (3 hidden + output).
    compress_fn expects list of dicts with W_in, W_out, spatial, n_alive
    for each HIDDEN layer (i.e., 3 entries for fc1, fc2, fc3).
    """
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
    info = []
    for i in range(len(layers) - 1):  # fc1, fc2, fc3
        W_in = layers[i].weight.data
        W_out = layers[i + 1].weight.data
        alive = max((W_in.abs().sum(dim=1) > 1e-8).sum().item(), 4)
        info.append({
            'W_in': W_in,
            'W_out': W_out,
            'spatial': 1,
            'n_alive': int(alive),
        })
    return info


def _build_compact_from_selections(sparse_model, selections, n_classes):
    """Build CompactMLP from per-layer neuron selections.

    selections: list of 3 sorted index lists (for fc1, fc2, fc3 hidden layers).
    """
    layers = [sparse_model.fc1, sparse_model.fc2, sparse_model.fc3, sparse_model.fc4]
    hiddens = tuple(len(s) for s in selections)
    compact = CompactMLP(
        input_dim=sparse_model.input_dim,
        hiddens=hiddens,
        n_classes=n_classes,
    )

    with torch.no_grad():
        prev_idx = None
        for i, sel in enumerate(selections):
            W = layers[i].weight.data
            if prev_idx is not None:
                W = W[:, prev_idx]
            # compact.net layout: Linear, ReLU, Linear, ReLU, Linear, ReLU, Linear(output)
            tgt = compact.net[i * 2]  # Linear layer
            tgt.weight.data.copy_(W[sel])
            if layers[i].bias is not None:
                tgt.bias.data.copy_(layers[i].bias.data[sel])
            prev_idx = sel

        # Output layer
        tgt_out = compact.net[-1]
        tgt_out.weight.data.copy_(layers[-1].weight.data[:, prev_idx])
        if layers[-1].bias is not None:
            tgt_out.bias.data.copy_(layers[-1].bias.data)

    return compact


# ═══════════════════════════════════════════
#  COMPRESSION METHODS (from dense_autosearch explorations)
# ═══════════════════════════════════════════

# ────── explore25: Evolutionary Adaptive Importance Blending (EAIB) ──────
def _compress_fn_explore25(layer_info):
    """Adaptive Importance Blending: evolves per-layer alpha to blend
    Flow Importance (FI = phi_in * phi_out) with Magnitude Importance (MI = phi_in + phi_out).
    Pareto-optimal across all 4 metrics.
    """
    n_layers = len(layer_info)
    if n_layers == 0:
        return []

    def compute_importance(info, alpha):
        W_in, W_out, spatial = info['W_in'], info['W_out'], info['spatial']
        n = W_in.shape[0]
        phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
        if W_out.dim() == 4:
            phi_out = W_out.abs().sum(dim=(0, 2, 3))
        elif spatial > 1:
            phi_out = torch.stack([
                W_out[:, j * spatial:(j + 1) * spatial].abs().sum() for j in range(n)
            ])
        else:
            phi_out = W_out.abs().sum(dim=0)
            if phi_out.shape[0] > n:
                phi_out = phi_out[:n]
        fi = phi_in * phi_out
        mi = phi_in + phi_out
        return alpha * fi + (1 - alpha) * mi

    def fitness(chromosome):
        ratios = chromosome[:n_layers]
        alphas = chromosome[n_layers:]
        flow_err, total_alive, total_kept = 0.0, 0, 0
        for i, info in enumerate(layer_info):
            importance = compute_importance(info, alphas[i])
            n = importance.shape[0]
            k = max(4, int(info['n_alive'] * ratios[i]))
            total_imp = importance.sum().item()
            if total_imp > 1e-12:
                _, topk = importance.topk(min(k, n))
                flow_err += 1.0 - importance[topk].sum().item() / total_imp
            total_alive += info['n_alive']
            total_kept += k
        return (1.0 - flow_err, 1.0 - total_kept / max(total_alive, 1))

    pop_size, gens, sigma = 10, 15, 0.12
    chromo_len = 2 * n_layers
    population = []
    for _ in range(pop_size):
        population.append(np.concatenate([
            np.random.uniform(0.1, 0.8, n_layers),
            np.random.uniform(0.0, 1.0, n_layers),
        ]))

    best_g, best_f = population[0].copy(), -np.inf
    for gen in range(gens):
        scored = [(g, fitness(g)) for g in population]
        front0 = []
        for i, (gi, fi) in enumerate(scored):
            dominated = False
            for j, (gj, fj) in enumerate(scored):
                if i != j and all(fj[k] >= fi[k] for k in range(2)) and any(fj[k] > fi[k] for k in range(2)):
                    dominated = True
                    break
            if not dominated:
                front0.append((gi, fi))
        for g, f in front0:
            if f[0] > best_f:
                best_f = f[0]
                best_g = g.copy()
        parents = [g for g, _ in front0]
        if len(parents) < 2:
            sorted_scored = sorted(scored, key=lambda x: x[1][0], reverse=True)
            parents = [sorted_scored[0][0]]
            parents.append(sorted_scored[1][0] if len(sorted_scored) > 1 else sorted_scored[0][0])
        sig_t = sigma * (1 - gen / gens * 0.5)
        new_pop = list(parents)
        while len(new_pop) < pop_size:
            p = parents[np.random.randint(len(parents))]
            child = p + np.random.randn(chromo_len) * sig_t
            child[:n_layers] = np.clip(child[:n_layers], 0.1, 0.8)
            child[n_layers:] = np.clip(child[n_layers:], 0.0, 1.0)
            new_pop.append(child)
        population = new_pop

    best_ratios = best_g[:n_layers]
    best_alphas = best_g[n_layers:]
    selections = []
    for i, info in enumerate(layer_info):
        importance = compute_importance(info, best_alphas[i])
        n = importance.shape[0]
        k = max(4, int(info['n_alive'] * best_ratios[i]))
        _, idx = importance.topk(min(k, n))
        selections.append(sorted(idx.cpu().tolist()))
    return selections


# ────── explore48: Evolutionary Prioritized Subspace Selection (EPSS) ──────
def _compress_fn_explore48(layer_info):
    """Prioritized Subspace Selection: extends EAIB with SVD-based subspace
    contribution score (SCS). Evolves per-layer alpha, lambda, ratios.
    Best inference speedup among all methods.
    """
    n_layers = len(layer_info)
    if n_layers == 0:
        return []

    def compute_importance(info, alpha, lambda_l):
        W_in, W_out, spatial = info['W_in'], info['W_out'], info['spatial']
        n = W_in.shape[0]
        if n == 0:
            return torch.tensor([])
        phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
        all_W_out_j_flat = []
        D_out_l = 0
        if W_out.dim() == 4:
            phi_out = W_out.abs().sum(dim=(0, 2, 3))
            D_out_l = W_out.shape[0] * W_out.shape[2] * W_out.shape[3]
            for j in range(n):
                all_W_out_j_flat.append(W_out[:, j, :, :].reshape(-1))
        elif spatial > 1:
            phi_out = torch.stack([
                W_out[:, j * spatial:(j + 1) * spatial].abs().sum() for j in range(n)
            ])
            D_out_l = W_out.shape[0] * spatial
            for j in range(n):
                all_W_out_j_flat.append(W_out[:, j * spatial:(j + 1) * spatial].reshape(-1))
        else:
            phi_out = W_out.abs().sum(dim=0)
            if phi_out.shape[0] > n:
                phi_out = phi_out[:n]
            D_out_l = W_out.shape[0]
            for j in range(n):
                all_W_out_j_flat.append(W_out[:, j])
        fi = phi_in * phi_out
        mi = phi_in + phi_out
        base_importance = alpha * fi + (1 - alpha) * mi
        SCS_j = torch.zeros(n)
        if D_out_l > 0 and len(all_W_out_j_flat) > 0:
            W_out_l_flat = torch.stack(all_W_out_j_flat)
            M = min(64, n)
            if M > 0:
                sampled_indices = torch.arange(n) if n <= M else torch.randperm(n)[:M]
                M_sample = W_out_l_flat[sampled_indices]
                _, _, V_sub_T = torch.linalg.svd(M_sample.float(), full_matrices=False)
                V_sub = V_sub_T.T
                num_comp = V_sub.shape[1]
                p = min(4, M, D_out_l, num_comp)
                if p < num_comp:
                    V_orth = V_sub[:, p:]
                    SCS_j = (W_out_l_flat.float() @ V_orth.float()).pow(2).sum(dim=1)
        return base_importance + lambda_l * SCS_j

    def fitness(chromosome):
        ratios = chromosome[:n_layers]
        alphas = chromosome[n_layers:2*n_layers]
        lambdas = chromosome[2*n_layers:]
        flow_err, total_alive, total_kept = 0.0, 0, 0
        for i, info in enumerate(layer_info):
            importance = compute_importance(info, alphas[i], lambdas[i])
            n = importance.shape[0]
            if n == 0:
                continue
            k = max(4, int(info['n_alive'] * ratios[i]))
            total_imp = importance.sum().item()
            if total_imp > 1e-12:
                _, topk = importance.topk(min(k, n))
                flow_err += 1.0 - importance[topk].sum().item() / total_imp
            total_alive += info['n_alive']
            total_kept += k
        return (1.0 - flow_err, 1.0 - total_kept / max(total_alive, 1))

    pop_size, gens, sigma = 10, 15, 0.12
    chromo_len = 3 * n_layers
    population = []
    for _ in range(pop_size):
        population.append(np.concatenate([
            np.random.uniform(0.1, 0.8, n_layers),
            np.random.uniform(0.0, 1.0, n_layers),
            np.random.uniform(0.0, 1.0, n_layers),
        ]))

    best_g, best_f = population[0].copy(), -np.inf
    for gen in range(gens):
        scored = [(g, fitness(g)) for g in population]
        front0 = []
        for i, (gi, fi) in enumerate(scored):
            dominated = False
            for j, (gj, fj) in enumerate(scored):
                if i != j and all(fj[k] >= fi[k] for k in range(2)) and any(fj[k] > fi[k] for k in range(2)):
                    dominated = True
                    break
            if not dominated:
                front0.append((gi, fi))
        for g, f in front0:
            if f[0] > best_f:
                best_f = f[0]
                best_g = g.copy()
        parents = [g for g, _ in front0]
        if len(parents) < 2:
            sorted_scored = sorted(scored, key=lambda x: x[1][0], reverse=True)
            parents = [sorted_scored[0][0]]
            parents.append(sorted_scored[1][0] if len(sorted_scored) > 1 else sorted_scored[0][0])
        sig_t = sigma * (1 - gen / gens * 0.5)
        new_pop = list(parents)
        while len(new_pop) < pop_size:
            p = parents[np.random.randint(len(parents))]
            child = p + np.random.randn(chromo_len) * sig_t
            child[:n_layers] = np.clip(child[:n_layers], 0.1, 0.8)
            child[n_layers:2*n_layers] = np.clip(child[n_layers:2*n_layers], 0.0, 1.0)
            child[2*n_layers:] = np.clip(child[2*n_layers:], 0.0, 1.0)
            new_pop.append(child)
        population = new_pop

    best_ratios = best_g[:n_layers]
    best_alphas = best_g[n_layers:2*n_layers]
    best_lambdas = best_g[2*n_layers:]
    selections = []
    for i, info in enumerate(layer_info):
        importance = compute_importance(info, best_alphas[i], best_lambdas[i])
        n = importance.shape[0]
        if n == 0:
            selections.append([])
            continue
        k = max(4, int(info['n_alive'] * best_ratios[i]))
        _, idx = importance.topk(min(k, n))
        selections.append(sorted(idx.cpu().tolist()))
    return selections


# ────── explore07: Multi-Objective EA (MOEA) ──────
def _compress_fn_explore07(layer_info):
    """Multi-objective EA with NSGA-II-style non-dominated sorting.
    Evolves per-layer keep ratios optimizing flow retention AND compression.
    Best compression ratio among all methods.
    """
    n_layers = len(layer_info)
    if n_layers == 0:
        return []

    def compute_phi(info):
        W_in, W_out, spatial = info['W_in'], info['W_out'], info['spatial']
        n = W_in.shape[0]
        phi_in = W_in.abs().reshape(n, -1).sum(dim=1)
        if W_out.dim() == 4:
            phi_out = W_out.abs().sum(dim=(0, 2, 3))
        elif spatial > 1:
            phi_out = torch.stack([
                W_out[:, j * spatial:(j + 1) * spatial].abs().sum() for j in range(n)
            ])
        else:
            phi_out = W_out.abs().sum(dim=0)
            if phi_out.shape[0] > n:
                phi_out = phi_out[:n]
        return phi_in * phi_out

    def fitness(ratios):
        flow_err, total_alive, total_kept = 0.0, 0, 0
        for i, info in enumerate(layer_info):
            phi = compute_phi(info)
            n = phi.shape[0]
            k = max(4, int(info['n_alive'] * ratios[i]))
            total_flow = phi.sum().item()
            if total_flow > 1e-12:
                _, topk = phi.topk(min(k, n))
                flow_err += 1.0 - phi[topk].sum().item() / total_flow
            total_alive += info['n_alive']
            total_kept += k
        return (1.0 - flow_err, 1.0 - total_kept / max(total_alive, 1))

    def is_dominated(obj1, obj2):
        better = False
        for i in range(len(obj1)):
            if obj2[i] < obj1[i]:
                return False
            if obj2[i] > obj1[i]:
                better = True
        return better

    pop_size, gens, sigma = 10, 15, 0.12
    population = [np.random.uniform(0.1, 0.8, n_layers) for _ in range(pop_size)]
    best_g, best_f = population[0].copy(), -np.inf

    for gen in range(gens):
        scored = [(g, fitness(g)) for g in population]

        # Non-dominated sort (front 0 only)
        front0 = []
        for i, (gi, fi) in enumerate(scored):
            dominated = False
            for j, (gj, fj) in enumerate(scored):
                if i != j and is_dominated(fi, fj):
                    dominated = True
                    break
            if not dominated:
                front0.append((gi, fi))

        for g, f in front0:
            if f[0] > best_f:
                best_f = f[0]
                best_g = g.copy()

        parents = [g for g, _ in front0]
        if len(parents) < 2:
            sorted_scored = sorted(scored, key=lambda x: x[1][0], reverse=True)
            parents = [sorted_scored[0][0]]
            parents.append(sorted_scored[1][0] if len(sorted_scored) > 1 else sorted_scored[0][0])

        sig_t = sigma * (1 - gen / gens * 0.5)
        new_pop = list(parents)
        while len(new_pop) < pop_size:
            p = parents[np.random.randint(len(parents))]
            child = np.clip(p + np.random.randn(n_layers) * sig_t, 0.1, 0.8)
            new_pop.append(child)
        population = new_pop

    # Final selection with best ratios
    selections = []
    for i, info in enumerate(layer_info):
        phi = compute_phi(info)
        n = phi.shape[0]
        k = max(4, int(info['n_alive'] * best_g[i]))
        _, idx = phi.topk(min(k, n))
        selections.append(sorted(idx.cpu().tolist()))
    return selections


# ═══════════════════════════════════════════
#  Conversion wrappers (unified interface)
# ═══════════════════════════════════════════

def do_gfcs(sparse_model, n_classes, **kw):
    compact, _ = gfcs_convert(sparse_model, n_classes, use_evolution=True)
    return compact

def do_neuron_removal(sparse_model, n_classes, **kw):
    return convert_neuron_removal(sparse_model, n_classes)

def do_autosearch_method(sparse_model, n_classes, compress_fn_ref, **kw):
    """Generic wrapper for autosearch compress_fn methods."""
    layer_info = _get_layer_info_simplemlp(sparse_model)
    selections = compress_fn_ref(layer_info)
    return _build_compact_from_selections(sparse_model, selections, n_classes)

def do_finetune(compact, train_dl, epochs=10, lr=0.01):
    train_model(compact, train_dl, epochs=epochs, lr=lr)
    return compact


# ═══════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════
DATASETS = ['moons', 'circles', 'spirals', 'blobs',
            'gaussian_quantiles', 'classification',
            'highdim', 'sequence_cls']
SEEDS = [42, 123]
SPARSITIES = {
    'moons': 0.90, 'circles': 0.85, 'spirals': 0.90,
    'blobs': 0.75, 'gaussian_quantiles': 0.75,
    'classification': 0.75, 'highdim': 0.70, 'sequence_cls': 0.90,
}

METHODS = {
    'GFCS':        lambda sm, nc: do_gfcs(sm, nc),
    'NeuronRem':   lambda sm, nc: do_neuron_removal(sm, nc),
    'EAIB_e25':    lambda sm, nc: do_autosearch_method(sm, nc, _compress_fn_explore25),
    'EPSS_e48':    lambda sm, nc: do_autosearch_method(sm, nc, _compress_fn_explore48),
    'MOEA_e07':    lambda sm, nc: do_autosearch_method(sm, nc, _compress_fn_explore07),
}

METHOD_LABELS = {
    'GFCS':      'GFCS (baseline)',
    'NeuronRem': 'Neuron Removal',
    'EAIB_e25':  'EAIB (explore25)',
    'EPSS_e48':  'EPSS (explore48)',
    'MOEA_e07':  'MOEA (explore07)',
}

def count_flops(model):
    total = 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            total += 2 * p.numel()
    return total


# ═══════════════════════════════════════════
#  Main benchmark
# ═══════════════════════════════════════════
def main():
    print("=" * 80)
    print("  Ex09v2: Extended Compression Benchmark")
    print("  5 methods × 8 datasets × 2 seeds = 80 measurements")
    print("=" * 80)

    print("\nCalibrating RCU anchor...")
    for _ in range(3):
        get_anchor_time()
    t_anchor = get_anchor_time()
    print(f"  Anchor time: {t_anchor/1e6:.2f}ms ({_tls.loops} loops)\n")

    os.makedirs('results', exist_ok=True)
    all_results = []

    for ds in DATASETS:
        for seed in SEEDS:
            print(f"\n{'═'*70}")
            print(f"  {ds}, seed={seed}")
            print(f"{'═'*70}")

            set_seed(seed)
            train_dl, val_dl, test_dl, n_classes = get_dataloaders(seed, ds, 64)

            # Teacher
            teacher_path = f"data/{ds}/teacher_seed{seed}.pt"
            input_dim = next(iter(test_dl))[0].shape[1]
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            if os.path.exists(teacher_path):
                teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            else:
                os.makedirs(f"data/{ds}", exist_ok=True)
                train_model(teacher, train_dl, epochs=100)
                torch.save(teacher.state_dict(), teacher_path)

            _, teacher_f1, _ = evaluate(teacher, test_dl)
            teacher_params = teacher.count_params()
            teacher_flops = count_flops(teacher)
            teacher_infer_rcu = measure_inference_rcu(teacher, test_dl)

            # Prune
            sparse = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse.load_state_dict({k: v.clone() for k, v in teacher.state_dict().items()})
            sp = SPARSITIES.get(ds, 0.80)
            prune_magnitude_global(sparse, sp)
            actual_sp = get_sparsity(sparse)
            _, sparse_f1, _ = evaluate(sparse, test_dl)

            print(f"  Teacher: F1={teacher_f1:.4f}  params={teacher_params:,}")
            print(f"  Sparse:  F1={sparse_f1:.4f}  sparsity={actual_sp:.0%}")
            print(f"  Teacher infer RCU: {teacher_infer_rcu:.6f}/pass")

            for method_key, method_fn in METHODS.items():
                set_seed(seed)

                try:
                    # Clone sparse
                    sparse_copy = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
                    sparse_copy.load_state_dict({k: v.clone() for k, v in sparse.state_dict().items()})
                    for name, p in sparse_copy.named_parameters():
                        if 'weight' in name:
                            p.data *= (p.data != 0).float()

                    # ── RCU: Conversion ──
                    compact, rcu_conv, t_conv_ns = profile_rcu(
                        method_fn, sparse_copy, n_classes
                    )

                    # Pre-finetune eval
                    _, pre_f1, _ = evaluate(compact, test_dl)

                    # ── RCU: Finetune ──
                    set_seed(seed)
                    _, rcu_ft, t_ft_ns = profile_rcu(
                        do_finetune, compact, train_dl, epochs=10, lr=0.01
                    )

                    rcu_total = rcu_conv + rcu_ft

                    # Final eval
                    _, final_f1, final_acc = evaluate(compact, test_dl)
                    compact_params = compact.count_params()
                    compact_flops = count_flops(compact)
                    compression = teacher_params / compact_params if compact_params > 0 else 0
                    flop_speedup = teacher_flops / compact_flops if compact_flops > 0 else 0
                    delta_f1 = final_f1 - sparse_f1
                    recovery = final_f1 / teacher_f1 if teacher_f1 > 0 else 0

                    # Inference speedup
                    compact_infer_rcu = measure_inference_rcu(compact, test_dl)
                    infer_speedup = teacher_infer_rcu / compact_infer_rcu if compact_infer_rcu > 0 else 0

                    result = {
                        'dataset': ds, 'seed': seed, 'method': method_key,
                        'sparsity': round(actual_sp, 3),
                        'teacher_f1': round(teacher_f1, 4),
                        'sparse_f1': round(sparse_f1, 4),
                        'pre_ft_f1': round(pre_f1, 4),
                        'final_f1': round(final_f1, 4),
                        'delta_f1': round(delta_f1, 4),
                        'recovery': round(recovery, 4),
                        'teacher_params': teacher_params,
                        'compact_params': compact_params,
                        'compression': round(compression, 2),
                        'flop_speedup': round(flop_speedup, 2),
                        'teacher_infer_rcu': round(teacher_infer_rcu, 6),
                        'compact_infer_rcu': round(compact_infer_rcu, 6),
                        'infer_speedup': round(infer_speedup, 2),
                        'rcu_conversion': round(rcu_conv, 3),
                        'rcu_finetune': round(rcu_ft, 3),
                        'rcu_total': round(rcu_total, 3),
                        't_conv_ms': round(t_conv_ns / 1e6, 2),
                        't_ft_ms': round(t_ft_ns / 1e6, 2),
                    }
                    all_results.append(result)

                    q = '✅' if delta_f1 >= -0.03 else '❌'
                    print(f"    {q} {METHOD_LABELS[method_key]:25s} "
                          f"F1={final_f1:.4f} (Δ={delta_f1:+.4f})  "
                          f"RCU_conv={rcu_conv:7.2f}  "
                          f"comp={compression:.1f}×  "
                          f"infer={infer_speedup:.2f}×  "
                          f"params={compact_params:,}")

                except Exception as e:
                    import traceback
                    print(f"    ❌ {METHOD_LABELS[method_key]:25s} FAILED: {e}")
                    traceback.print_exc()
                    all_results.append({
                        'dataset': ds, 'seed': seed, 'method': method_key,
                        'error': str(e)
                    })

    # ═══════════════════════════════════════════
    #  Summary Tables
    # ═══════════════════════════════════════════
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if 'error' in r:
            continue
        key = r['method']
        for k in ['final_f1', 'delta_f1', 'compression', 'flop_speedup',
                   'recovery', 'rcu_conversion', 'rcu_finetune', 'rcu_total',
                   'compact_params', 'infer_speedup', 'pre_ft_f1']:
            agg[key][k].append(r[k])

    method_order = ['GFCS', 'NeuronRem', 'EAIB_e25', 'EPSS_e48', 'MOEA_e07']

    print(f"\n\n{'='*120}")
    print("  Ex09v2 FULL BENCHMARK — 5 methods × 8 datasets × 2 seeds (averaged)")
    print(f"{'='*120}\n")

    print(f"{'Method':25s} {'F1':\u003e6s} {'ΔF1':\u003e7s} {'pre_F1':\u003e7s} {'RPR':\u003e6s} "
          f"{'Comp×':\u003e6s} {'Infer×':\u003e7s} {'Params':\u003e8s} "
          f"{'RCU_conv':\u003e8s} {'RCU_tot':\u003e8s}  Q")
    print("─" * 120)

    for mk in method_order:
        if mk not in agg:
            continue
        d = agg[mk]
        f1 = np.mean(d['final_f1'])
        df1 = np.mean(d['delta_f1'])
        pf1 = np.mean(d['pre_ft_f1'])
        rec = np.mean(d['recovery'])
        comp = np.mean(d['compression'])
        isp = np.mean(d['infer_speedup'])
        params = int(np.mean(d['compact_params']))
        rc = np.mean(d['rcu_conversion'])
        rt = np.mean(d['rcu_total'])
        n_ok = sum(1 for x in d['delta_f1'] if x >= -0.03)
        q = f'{n_ok}/{len(d["delta_f1"])}'
        marker = ' ◄' if mk == 'GFCS' else ''
        print(f"{METHOD_LABELS[mk]:25s} {f1:.4f} {df1:+.4f} {pf1:.4f} {rec:.4f} "
              f"{comp:6.1f} {isp:7.2f} {params:\u003e8,} "
              f"{rc:8.2f} {rt:8.2f}  {q}{marker}")

    # ── Per-dataset breakdown ──
    print(f"\n\n{'='*120}")
    print("  PER-DATASET BREAKDOWN (averaged across seeds)")
    print(f"{'='*120}\n")

    ds_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_results:
        if 'error' in r:
            continue
        ds_agg[r['dataset']][r['method']]['final_f1'].append(r['final_f1'])
        ds_agg[r['dataset']][r['method']]['compression'].append(r['compression'])
        ds_agg[r['dataset']][r['method']]['infer_speedup'].append(r['infer_speedup'])
        ds_agg[r['dataset']][r['method']]['rcu_conversion'].append(r['rcu_conversion'])
        ds_agg[r['dataset']][r['method']]['compact_params'].append(r['compact_params'])

    print(f"{'Dataset':20s}", end='')
    for mk in method_order:
        short = mk[:8]
        print(f"  {'F1':>6s}|{'Comp':>5s}|{'Inf×':>5s}", end='')
    print()
    print(f"{'':20s}", end='')
    for mk in method_order:
        print(f"  {mk[:8]:>6s}|{'':>5s}|{'':>5s}", end='')
    print()
    print("─" * 120)

    for ds in DATASETS:
        print(f"{ds:20s}", end='')
        for mk in method_order:
            if mk in ds_agg[ds]:
                dd = ds_agg[ds][mk]
                f1 = np.mean(dd['final_f1'])
                comp = np.mean(dd['compression'])
                isp = np.mean(dd['infer_speedup'])
                print(f"  {f1:.4f}|{comp:5.1f}|{isp:5.2f}", end='')
            else:
                print(f"  {'N/A':>6s}|{'N/A':>5s}|{'N/A':>5s}", end='')
        print()

    # ── Winner per dataset ──
    print(f"\n\n{'='*80}")
    print("  WINNER PER DATASET (highest F1 among quality-passing methods)")
    print(f"{'='*80}\n")

    for ds in DATASETS:
        candidates = []
        for mk in method_order:
            if mk in ds_agg[ds]:
                f1 = np.mean(ds_agg[ds][mk]['final_f1'])
                df1_vals = []
                for r in all_results:
                    if r.get('dataset') == ds and r.get('method') == mk and 'error' not in r:
                        df1_vals.append(r['delta_f1'])
                avg_df = np.mean(df1_vals) if df1_vals else -1
                if avg_df >= -0.03:
                    candidates.append((mk, f1, np.mean(ds_agg[ds][mk]['compression'])))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            winner = candidates[0]
            print(f"  {ds:20s} → {METHOD_LABELS[winner[0]]:25s} F1={winner[1]:.4f}  comp={winner[2]:.1f}×")

    # ── Pareto analysis ──
    print(f"\n\n{'='*80}")
    print("  PARETO DOMINANCE ANALYSIS")
    print(f"{'='*80}\n")

    for mi, mk_i in enumerate(method_order):
        if mk_i not in agg:
            continue
        wins_over = []
        for mj, mk_j in enumerate(method_order):
            if mi == mj or mk_j not in agg:
                continue
            # mk_i dominates mk_j if better or equal on ALL, strictly better on at least one
            f1_i, f1_j = np.mean(agg[mk_i]['final_f1']), np.mean(agg[mk_j]['final_f1'])
            rc_i, rc_j = np.mean(agg[mk_i]['rcu_conversion']), np.mean(agg[mk_j]['rcu_conversion'])
            co_i, co_j = np.mean(agg[mk_i]['compression']), np.mean(agg[mk_j]['compression'])
            is_i, is_j = np.mean(agg[mk_i]['infer_speedup']), np.mean(agg[mk_j]['infer_speedup'])

            ge_f1 = f1_i >= f1_j
            ge_rc = rc_i <= rc_j  # lower RCU = better
            ge_co = co_i >= co_j
            ge_is = is_i >= is_j

            all_ge = ge_f1 and ge_rc and ge_co and ge_is
            strict = (f1_i > f1_j or rc_i < rc_j or co_i > co_j or is_i > is_j)
            if all_ge and strict:
                wins_over.append(mk_j)

        if wins_over:
            print(f"  {METHOD_LABELS[mk_i]:25s} Pareto-dominates: {', '.join(METHOD_LABELS[w] for w in wins_over)}")
        else:
            print(f"  {METHOD_LABELS[mk_i]:25s} Pareto-non-dominated")

    # Save
    with open('results/ex09v2_benchmark.json', 'w') as f:
        json.dump({'results': all_results, 'method_labels': METHOD_LABELS}, f, indent=2)
    print(f"\n  Saved: results/ex09v2_benchmark.json")


if __name__ == '__main__':
    main()
