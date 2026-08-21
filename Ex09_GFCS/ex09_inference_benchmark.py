#!/usr/bin/env python3
"""
Ex09: Inference RCU Benchmark
==============================
Measures the RCU cost of a single forward pass for:
  - Teacher (original dense)
  - Sparse (pruned)
  - Compact models from each of 6 methods

RCU = Thread_Time(inference) / Avg_Thread_Time(Anchor)
"""

# ═══ MANDATORY: Disable hidden C++ threading ═══
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
    set_seed, get_dataloaders, SimpleMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
    convert_neuron_removal, convert_svd_compression,
    convert_knowledge_distill, convert_weight_redistribution,
)
from ex09_lib.evomerge import evomerge
from ex09_lib.gfcs import gfcs_convert

# ═══ RCU Anchor ═══
_tls = threading.local()

def _get_local_anchor():
    if not hasattr(_tls, "initialized"):
        _tls.a = np.random.rand(1024)
        _tls.b = np.random.rand(1024)
        _tls.c = np.empty_like(_tls.a)
        _tls.loops = 50
        while True:
            t = _run_anchor(_tls.a, _tls.b, _tls.c, _tls.loops)
            if t >= 5_000_000:
                break
            _tls.loops *= 2
        _tls.initialized = True
    return _tls.a, _tls.b, _tls.c, _tls.loops

def _run_anchor(a, b, c, loops):
    s = time.thread_time_ns()
    for _ in range(loops):
        np.multiply(a, b, out=c)
        np.add(c, a, out=c)
    return time.thread_time_ns() - s

def get_anchor_time():
    a, b, c, loops = _get_local_anchor()
    return _run_anchor(a, b, c, loops)

def measure_inference_rcu(model, test_dl, n_repeats=100):
    """Measure inference RCU: cost of n_repeats forward passes."""
    model.eval()
    X_batch = next(iter(test_dl))[0]

    # Warmup
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
    rcu_per_pass = rcu_total / n_repeats

    return rcu_per_pass, t_algo / n_repeats  # rcu_per_pass, ns_per_pass

# ═══ Config ═══
DATASETS = ['moons', 'circles', 'spirals', 'blobs',
            'gaussian_quantiles', 'classification', 'highdim', 'sequence_cls']
SEEDS = [42, 123]
METHODS = ['neuron_removal', 'svd_compression', 'knowledge_distill',
           'weight_redistribution', 'evomerge', 'gfcs']
SPARSITIES = {
    'moons': 0.90, 'circles': 0.85, 'spirals': 0.90,
    'blobs': 0.75, 'gaussian_quantiles': 0.75,
    'classification': 0.75, 'highdim': 0.70, 'sequence_cls': 0.90,
}

def do_conversion(method, sparse_model, train_dl, n_classes, sparsity):
    if method == 'neuron_removal':
        return convert_neuron_removal(sparse_model, n_classes)
    elif method == 'svd_compression':
        return convert_svd_compression(sparse_model, max(0.1, 1.0 - sparsity), n_classes)
    elif method == 'knowledge_distill':
        return convert_knowledge_distill(sparse_model, train_dl, n_classes, epochs=30)
    elif method == 'weight_redistribution':
        return convert_weight_redistribution(sparse_model, n_classes)
    elif method == 'evomerge':
        compact, _ = evomerge(sparse_model, train_dl, n_classes, pop_size=20, generations=30)
        return compact
    elif method == 'gfcs':
        compact, _ = gfcs_convert(sparse_model, n_classes, use_evolution=True)
        return compact


def main():
    print("Calibrating RCU anchor...")
    for _ in range(5):
        get_anchor_time()
    print(f"  Anchor: {get_anchor_time()/1e6:.2f}ms ({_tls.loops} loops)\n")

    all_results = []

    for ds in DATASETS:
        for seed in SEEDS:
            set_seed(seed)
            train_dl, _, test_dl, n_classes = get_dataloaders(seed, ds, 64)
            input_dim = next(iter(test_dl))[0].shape[1]

            # Teacher
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            teacher_path = f"data/{ds}/teacher_seed{seed}.pt"
            if os.path.exists(teacher_path):
                teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            else:
                os.makedirs(f"data/{ds}", exist_ok=True)
                train_model(teacher, train_dl, epochs=100)
                torch.save(teacher.state_dict(), teacher_path)

            teacher_rcu, teacher_ns = measure_inference_rcu(teacher, test_dl)

            # Sparse
            sparse = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse.load_state_dict({k: v.clone() for k, v in teacher.state_dict().items()})
            sp = SPARSITIES.get(ds, 0.80)
            prune_magnitude_global(sparse, sp)
            actual_sp = get_sparsity(sparse)
            _, sparse_f1, _ = evaluate(sparse, test_dl)

            sparse_rcu, sparse_ns = measure_inference_rcu(sparse, test_dl)

            print(f"{'═'*70}")
            print(f"  {ds}, seed={seed}  (sparsity={actual_sp:.0%})")
            print(f"  Teacher inference: {teacher_rcu:.6f} RCU  ({teacher_ns/1e3:.1f}μs)")
            print(f"  Sparse inference:  {sparse_rcu:.6f} RCU  ({sparse_ns/1e3:.1f}μs)")

            for method in METHODS:
                set_seed(seed)
                try:
                    sparse_copy = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
                    sparse_copy.load_state_dict({k: v.clone() for k, v in sparse.state_dict().items()})
                    for _, p in sparse_copy.named_parameters():
                        if 'weight' in _.split('.')[-1]:
                            p.data *= (p.data != 0).float()

                    compact = do_conversion(method, sparse_copy, train_dl, n_classes, actual_sp)

                    # Finetune
                    set_seed(seed)
                    train_model(compact, train_dl, epochs=10, lr=0.01)
                    _, final_f1, _ = evaluate(compact, test_dl)
                    delta_f1 = final_f1 - sparse_f1

                    # Inference RCU
                    compact_rcu, compact_ns = measure_inference_rcu(compact, test_dl)
                    compact_params = compact.count_params()

                    # Speedups
                    speedup_vs_teacher = teacher_rcu / compact_rcu if compact_rcu > 0 else 0
                    speedup_vs_sparse = sparse_rcu / compact_rcu if compact_rcu > 0 else 0

                    result = {
                        'dataset': ds, 'seed': seed, 'method': method,
                        'teacher_rcu': round(teacher_rcu, 6),
                        'sparse_rcu': round(sparse_rcu, 6),
                        'compact_rcu': round(compact_rcu, 6),
                        'speedup_vs_teacher': round(speedup_vs_teacher, 2),
                        'speedup_vs_sparse': round(speedup_vs_sparse, 2),
                        'compact_params': compact_params,
                        'final_f1': round(final_f1, 4),
                        'delta_f1': round(delta_f1, 4),
                    }
                    all_results.append(result)

                    q = '' if delta_f1 >= -0.03 else ''
                    print(f"  {q} {method:25s} RCU={compact_rcu:.6f}  "
                          f"speed vs teacher={speedup_vs_teacher:.2f}×  "
                          f"vs sparse={speedup_vs_sparse:.2f}×  "
                          f"F1={final_f1:.4f}")

                except Exception as e:
                    print(f"   {method:25s} FAILED: {e}")

    # ═══ Summary ═══
    from collections import defaultdict
    agg = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        m = r['method']
        agg[m]['compact_rcu'].append(r['compact_rcu'])
        agg[m]['speedup_vs_teacher'].append(r['speedup_vs_teacher'])
        agg[m]['speedup_vs_sparse'].append(r['speedup_vs_sparse'])
        agg[m]['delta_f1'].append(r['delta_f1'])
        agg[m]['compact_params'].append(r['compact_params'])

    # Also get teacher/sparse averages
    teacher_rcus = list(set((r['dataset'], r['seed'], r['teacher_rcu']) for r in all_results))
    sparse_rcus = list(set((r['dataset'], r['seed'], r['sparse_rcu']) for r in all_results))
    avg_teacher_rcu = np.mean([x[2] for x in teacher_rcus])
    avg_sparse_rcu = np.mean([x[2] for x in sparse_rcus])

    print(f"\n\n{'='*100}")
    print("  INFERENCE RCU BENCHMARK — Averaged across 8 datasets × 2 seeds")
    print(f"{'='*100}\n")

    print(f"  Teacher (original):   {avg_teacher_rcu:.6f} RCU/pass  (baseline)")
    print(f"  Sparse (pruned):      {avg_sparse_rcu:.6f} RCU/pass  (same arch, zero weights)\n")

    print(f"{'Method':25s} {'Infer RCU':>10s} {'vs Teacher':>11s} {'vs Sparse':>11s} "
          f"{'Params':>8s} {'ΔF1':>7s} {'Reliable':>8s}")
    print("─" * 90)

    for m in METHODS:
        d = agg[m]
        rcu = np.mean(d['compact_rcu'])
        vt = np.mean(d['speedup_vs_teacher'])
        vs = np.mean(d['speedup_vs_sparse'])
        params = int(np.mean(d['compact_params']))
        df1 = np.mean(d['delta_f1'])
        # Reliability
        per_ds = defaultdict(list)
        for r in all_results:
            if r['method'] == m:
                per_ds[r['dataset']].append(r['delta_f1'])
        rel = sum(1 for vals in per_ds.values() if np.mean(vals) >= -0.03)
        marker = ' ◄' if m == 'gfcs' else ''
        print(f"{m:25s} {rcu:10.6f} {vt:10.2f}× {vs:10.2f}× "
              f"{params:>8,} {df1:+7.4f} {rel}/8{marker}")

    # Per dataset: inference speedup
    print(f"\n\n{'='*100}")
    print("  INFERENCE SPEEDUP vs TEACHER — per dataset (тільки методи з ΔF1 ≥ −0.03)")
    print(f"{'='*100}\n")

    SHORT = {'neuron_removal':'NR', 'svd_compression':'SVD', 'knowledge_distill':'KD',
             'weight_redistribution':'WR', 'evomerge':'Evo', 'gfcs':'GFCS'}

    print(f"{'Dataset':20s}", end='')
    for m in METHODS:
        print(f'{SHORT[m]:>8s}', end='')
    print()
    print("─" * 68)

    for ds in DATASETS:
        print(f"{ds:20s}", end='')
        for m in METHODS:
            vals_sp = [r['speedup_vs_teacher'] for r in all_results
                       if r['dataset'] == ds and r['method'] == m]
            vals_df = [r['delta_f1'] for r in all_results
                       if r['dataset'] == ds and r['method'] == m]
            if vals_sp:
                avg_sp = np.mean(vals_sp)
                fail = np.mean(vals_df) < -0.03
                mark = '' if fail else '×'
                print(f'{avg_sp:6.2f}{mark} ', end='')
            else:
                print(f'   N/A  ', end='')
        print()

    # Save
    with open('results/inference_rcu.json', 'w') as f:
        json.dump({'results': all_results}, f, indent=2)
    print(f"\n  Saved: results/inference_rcu.json")


if __name__ == '__main__':
    main()
