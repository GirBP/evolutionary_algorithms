#!/usr/bin/env python3
"""
Ex09: Full Benchmark with RCU Profiling
========================================
6 methods × 8 datasets × 2 seeds
Scoring: RCU (Relative Compute Units) per RCU_protocol.md

RCU = Thread_Time(Algorithm) / Average_Thread_Time(Anchor)
"""

# ═══ MANDATORY: Disable hidden C++ threading BEFORE any imports ═══
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys, json, time, threading
import torch
torch.set_num_threads(1)

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP, CompactMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
    convert_neuron_removal, convert_svd_compression,
    convert_knowledge_distill, convert_weight_redistribution,
)
from ex09_lib.evomerge import evomerge
from ex09_lib.gfcs import gfcs_convert

# ═══════════════════════════════════════════
#  RCU Anchor (per RCU_protocol.md §4)
# ═══════════════════════════════════════════
_tls = threading.local()

def _get_local_anchor():
    """Thread-local L1-bound arrays (~24 KB total)."""
    if not hasattr(_tls, "initialized"):
        _tls.a = np.random.rand(1024)
        _tls.b = np.random.rand(1024)
        _tls.c = np.empty_like(_tls.a)
        # Auto-calibrate to ~5-10ms
        _tls.loops = 50
        while True:
            t = _run_anchor_loop(_tls.a, _tls.b, _tls.c, _tls.loops)
            if t >= 5_000_000:  # 5ms in ns
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
    """Execute func and return (result, rcu_cost)."""
    anchor_pre = get_anchor_time()
    start_algo = time.thread_time_ns()
    result = func(*args, **kwargs)
    t_algo = time.thread_time_ns() - start_algo
    anchor_post = get_anchor_time()
    t_anchor_avg = (anchor_pre + anchor_post) / 2.0
    rcu = t_algo / max(t_anchor_avg, 1)
    return result, rcu, t_algo

# ═══════════════════════════════════════════
#  Conversion wrappers
# ═══════════════════════════════════════════
def do_conversion(method, sparse_model, train_dl, n_classes, sparsity):
    if method == 'neuron_removal':
        return convert_neuron_removal(sparse_model, n_classes)
    elif method == 'svd_compression':
        rank_ratio = max(0.1, 1.0 - sparsity)
        return convert_svd_compression(sparse_model, rank_ratio, n_classes)
    elif method == 'knowledge_distill':
        return convert_knowledge_distill(sparse_model, train_dl, n_classes, epochs=30)
    elif method == 'weight_redistribution':
        return convert_weight_redistribution(sparse_model, n_classes)
    elif method == 'evomerge':
        compact, _ = evomerge(sparse_model, train_dl, n_classes,
                              pop_size=20, generations=30,
                              min_ratio=0.1, max_ratio=0.8)
        return compact
    elif method == 'gfcs':
        compact, _ = gfcs_convert(sparse_model, n_classes, use_evolution=True)
        return compact

def do_finetune(compact, train_dl, epochs=10, lr=0.01):
    train_model(compact, train_dl, epochs=epochs, lr=lr)
    return compact

# ═══════════════════════════════════════════
#  Main benchmark
# ═══════════════════════════════════════════
DATASETS = ['moons', 'circles', 'spirals', 'blobs',
            'gaussian_quantiles', 'classification',
            'highdim', 'sequence_cls']
SEEDS = [42, 123]
METHODS = ['neuron_removal', 'svd_compression', 'knowledge_distill',
           'weight_redistribution', 'evomerge', 'gfcs']
SPARSITIES = {
    'moons': 0.90, 'circles': 0.85, 'spirals': 0.90,
    'blobs': 0.75, 'gaussian_quantiles': 0.75,
    'classification': 0.75, 'highdim': 0.70, 'sequence_cls': 0.90,
}

def count_flops(model):
    total = 0
    for name, p in model.named_parameters():
        if 'weight' in name:
            total += 2 * p.numel()
    return total

def main():
    # Warmup anchor
    print("Calibrating RCU anchor...")
    for _ in range(3):
        get_anchor_time()
    t_anchor = get_anchor_time()
    print(f"  Anchor time: {t_anchor/1e6:.2f}ms ({_tls.loops} loops)")

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

            # Prune
            sparse = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse.load_state_dict({k: v.clone() for k, v in teacher.state_dict().items()})
            sp = SPARSITIES.get(ds, 0.80)
            prune_magnitude_global(sparse, sp)
            actual_sp = get_sparsity(sparse)
            _, sparse_f1, _ = evaluate(sparse, test_dl)

            print(f"  Teacher: F1={teacher_f1:.4f}  params={teacher_params:,}")
            print(f"  Sparse:  F1={sparse_f1:.4f}  sparsity={actual_sp:.0%}")

            for method in METHODS:
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
                        do_conversion, method, sparse_copy, train_dl, n_classes, actual_sp
                    )

                    # Pre-finetune
                    _, pre_f1, _ = evaluate(compact, test_dl)

                    # ── RCU: Finetune ──
                    set_seed(seed)
                    _, rcu_ft, t_ft_ns = profile_rcu(
                        do_finetune, compact, train_dl, epochs=10, lr=0.01
                    )

                    # Total RCU = conversion + finetune
                    rcu_total = rcu_conv + rcu_ft

                    # Final eval
                    _, final_f1, final_acc = evaluate(compact, test_dl)
                    compact_params = compact.count_params()
                    compact_flops = count_flops(compact)
                    compression = teacher_params / compact_params if compact_params > 0 else 0
                    flop_speedup = teacher_flops / compact_flops if compact_flops > 0 else 0
                    delta_f1 = final_f1 - sparse_f1
                    recovery = final_f1 / teacher_f1 if teacher_f1 > 0 else 0

                    result = {
                        'dataset': ds, 'seed': seed, 'method': method,
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
                        'rcu_conversion': round(rcu_conv, 3),
                        'rcu_finetune': round(rcu_ft, 3),
                        'rcu_total': round(rcu_total, 3),
                        't_conv_ms': round(t_conv_ns / 1e6, 2),
                        't_ft_ms': round(t_ft_ns / 1e6, 2),
                    }
                    all_results.append(result)

                    q = '' if delta_f1 >= -0.03 else ''
                    print(f"    {q} {method:25s} F1={final_f1:.4f} (Δ={delta_f1:+.4f})  "
                          f"RCU={rcu_total:7.2f}  "
                          f"[conv={rcu_conv:.2f} ft={rcu_ft:.2f}]  "
                          f"comp={compression:.1f}×")

                except Exception as e:
                    print(f"     {method:25s} FAILED: {e}")
                    all_results.append({
                        'dataset': ds, 'seed': seed, 'method': method,
                        'error': str(e)
                    })

    # ═══════════════════════════════════════════
    #  Summary Table
    # ═══════════════════════════════════════════
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if 'error' in r:
            continue
        key = (r['dataset'], r['method'])
        for k in ['final_f1', 'delta_f1', 'compression', 'flop_speedup',
                   'recovery', 'rcu_conversion', 'rcu_finetune', 'rcu_total']:
            agg[key][k].append(r[k])

    print(f"\n\n{'='*110}")
    print("  FULL BENCHMARK WITH RCU — 6 methods × 8 datasets × 2 seeds (averaged)")
    print(f"{'='*110}\n")

    print(f"{'Dataset':20s} {'Method':25s} {'F1':>6s} {'ΔF1':>7s} {'Comp×':>6s} "
          f"{'FLOP×':>6s} {'RCU_conv':>8s} {'RCU_ft':>8s} {'RCU_tot':>8s}  Q")
    print("─" * 110)

    for ds in DATASETS:
        for mi, method in enumerate(METHODS):
            key = (ds, method)
            if key not in agg:
                continue
            d = agg[key]
            f1 = np.mean(d['final_f1'])
            df1 = np.mean(d['delta_f1'])
            comp = np.mean(d['compression'])
            flop = np.mean(d['flop_speedup'])
            rc = np.mean(d['rcu_conversion'])
            rf = np.mean(d['rcu_finetune'])
            rt = np.mean(d['rcu_total'])
            q = '' if df1 >= -0.03 else ''
            ds_label = ds if mi == 0 else ''
            marker = ' ◄' if method == 'gfcs' else ''
            print(f"{ds_label:20s} {method:25s} {f1:.4f} {df1:+.4f} {comp:6.1f} "
                  f"{flop:6.1f} {rc:8.2f} {rf:8.2f} {rt:8.2f}  {q}{marker}")
        print()

    # ── RCU Winner per dataset (lowest RCU among quality-passing methods) ──
    print(f"\n{'='*70}")
    print("  WINNER PER DATASET (lowest RCU_total, among ΔF1 ≥ −0.03)")
    print(f"{'='*70}\n")

    gfcs_wins = 0
    for ds in DATASETS:
        candidates = []
        for method in METHODS:
            key = (ds, method)
            if key not in agg:
                continue
            df1 = np.mean(agg[key]['delta_f1'])
            rcu = np.mean(agg[key]['rcu_total'])
            if df1 >= -0.03:
                candidates.append((method, rcu, df1))

        if candidates:
            candidates.sort(key=lambda x: x[1])  # sort by RCU ascending
            winner = candidates[0]
            is_gfcs = winner[0] == 'gfcs'
            if is_gfcs:
                gfcs_wins += 1
            # Find GFCS rank
            gfcs_entry = [c for c in candidates if c[0] == 'gfcs']
            gfcs_rank = next((i+1 for i, c in enumerate(candidates) if c[0] == 'gfcs'), '?')
            gfcs_rcu = gfcs_entry[0][1] if gfcs_entry else 0

            marker = ' ' if is_gfcs else ''
            print(f"  {ds:20s} → {winner[0]:25s} RCU={winner[1]:7.2f}  "
                  f"(GFCS: rank {gfcs_rank}/{len(candidates)}, RCU={gfcs_rcu:.2f}){marker}")

    print(f"\n  GFCS RCU-wins: {gfcs_wins}/{len(DATASETS)}")

    # ── Pareto analysis: quality × efficiency ──
    print(f"\n{'='*70}")
    print("  PARETO ANALYSIS: ΔF1 ≥ −0.03 AND lowest RCU_conversion")
    print("  (finetune cost identical for all → compare conversion only)")
    print(f"{'='*70}\n")

    gfcs_conv_wins = 0
    for ds in DATASETS:
        candidates = []
        for method in METHODS:
            key = (ds, method)
            if key not in agg:
                continue
            df1 = np.mean(agg[key]['delta_f1'])
            rc = np.mean(agg[key]['rcu_conversion'])
            comp = np.mean(agg[key]['compression'])
            if df1 >= -0.03:
                candidates.append((method, rc, df1, comp))

        if candidates:
            candidates.sort(key=lambda x: x[1])
            winner = candidates[0]
            is_gfcs = winner[0] == 'gfcs'
            if is_gfcs:
                gfcs_conv_wins += 1
            gfcs_entry = [c for c in candidates if c[0] == 'gfcs']
            gfcs_rank = next((i+1 for i, c in enumerate(candidates) if c[0] == 'gfcs'), '?')
            gfcs_rc = gfcs_entry[0][1] if gfcs_entry else 0

            marker = ' ' if is_gfcs else ''
            print(f"  {ds:20s} → {winner[0]:25s} RCU_conv={winner[1]:7.3f}  "
                  f"(GFCS: rank {gfcs_rank}, RCU_conv={gfcs_rc:.3f}){marker}")

    print(f"\n  GFCS conversion-RCU wins: {gfcs_conv_wins}/{len(DATASETS)}")

    # Save
    with open('results/full_benchmark_rcu.json', 'w') as f:
        json.dump({'results': all_results}, f, indent=2)
    print(f"\n  Saved: results/full_benchmark_rcu.json")


if __name__ == '__main__':
    main()
