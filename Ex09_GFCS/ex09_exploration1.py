#!/usr/bin/env python3
"""
Ex09 Exploration 1: Hypothesis + Code — GFCS on Tier 1
========================================================
Protocol §3: Test the GFCS operator on Tier 1 (moons, circles) × 2 seeds.

Metrics:
  - speedup = T(sparse) / T(compact)  [real inference time]
  - F1 recovery: Q(f_C) ≥ Q(f_S) − 0.02
  - Compression ratio: teacher_params / compact_params

Also compares against existing baselines (neuron_removal, evomerge).
"""
import sys, os, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP, CompactMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
    convert_neuron_removal,
)
from ex09_lib.gfcs import gfcs_convert


def measure_inference_time(model, test_dl, n_repeats=50):
    """Measure real inference time (RCU)."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for X, _ in test_dl:
            _ = model(X)
            break
    
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            for X, _ in test_dl:
                _ = model(X)
        times.append(time.perf_counter() - t0)
    
    return np.median(times)


def run_tier1():
    """Run Exploration 1 on Tier 1 datasets."""
    datasets = ['moons', 'circles']
    seeds = [42, 123]
    ft_epochs = 10
    ft_lr = 0.01
    
    all_results = []
    
    for dataset in datasets:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}, Seed: {seed}")
            print(f"{'='*60}")
            
            set_seed(seed)
            train_dl, val_dl, test_dl, n_classes = get_dataloaders(seed, dataset, batch_size=64)
            
            # Load pre-trained teacher
            teacher_path = f"data/{dataset}/teacher_seed{seed}.pt"
            input_dim = 2  # moons, circles are 2D
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            _, teacher_f1, teacher_acc = evaluate(teacher, test_dl)
            teacher_params = teacher.count_params()
            print(f"  Teacher: F1={teacher_f1:.4f}, acc={teacher_acc:.4f}, params={teacher_params:,}")
            
            # Load sparse model  
            # Find the sparse model file
            sparse_files = [f for f in os.listdir(f"data/{dataset}") 
                          if f.startswith(f'sparse_seed{seed}')]
            if not sparse_files:
                print(f"  ERROR: No sparse model found for {dataset} seed {seed}")
                continue
            
            sparse_path = f"data/{dataset}/{sparse_files[0]}"
            sparse_model = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse_model.load_state_dict(torch.load(sparse_path, weights_only=True))
            actual_sp = get_sparsity(sparse_model)
            _, sparse_f1, sparse_acc = evaluate(sparse_model, test_dl)
            sparse_nz = sparse_model.count_nonzero()
            print(f"  Sparse: F1={sparse_f1:.4f}, sp={actual_sp:.1%}, nonzero={sparse_nz:,}")
            
            # Measure sparse inference time
            t_sparse = measure_inference_time(sparse_model, test_dl)
            print(f"  Sparse inference time: {t_sparse*1000:.2f}ms")
            
            # ─── Method 1: GFCS (our new operator) ───
            print(f"\n  --- GFCS (Gradient-Flow Connectivity Synthesis) ---")
            t0 = time.time()
            compact_gfcs, gfcs_info = gfcs_convert(sparse_model, n_classes=n_classes)
            convert_time_gfcs = time.time() - t0
            
            _, pre_ft_f1_gfcs, _ = evaluate(compact_gfcs, test_dl)
            print(f"  GFCS pre-finetune: F1={pre_ft_f1_gfcs:.4f}")
            print(f"  GFCS merged hiddens: {gfcs_info['merged_hiddens']}")
            print(f"  GFCS compression: ×{gfcs_info['compression']:.1f}")
            
            # Fine-tune
            t1 = time.time()
            train_model(compact_gfcs, train_dl, epochs=ft_epochs, lr=ft_lr)
            ft_time_gfcs = time.time() - t1
            
            _, final_f1_gfcs, final_acc_gfcs = evaluate(compact_gfcs, test_dl)
            compact_params_gfcs = compact_gfcs.count_params()
            t_compact_gfcs = measure_inference_time(compact_gfcs, test_dl)
            speedup_gfcs = t_sparse / t_compact_gfcs
            
            delta_f1_gfcs = final_f1_gfcs - sparse_f1
            quality_ok = delta_f1_gfcs >= -0.02
            speedup_ok = speedup_gfcs > 1.0
            
            flag = '✅' if (quality_ok and speedup_ok) else '❌'
            print(f"  {flag} GFCS: F1={final_f1_gfcs:.4f} (Δ={delta_f1_gfcs:+.4f}) "
                  f"speedup={speedup_gfcs:.2f}× params={compact_params_gfcs:,} "
                  f"t_convert={convert_time_gfcs:.2f}s")
            
            result_gfcs = {
                'dataset': dataset, 'seed': seed, 'method': 'gfcs',
                'teacher_f1': teacher_f1, 'sparse_f1': sparse_f1,
                'pre_ft_f1': pre_ft_f1_gfcs, 'final_f1': final_f1_gfcs,
                'delta_f1': delta_f1_gfcs,
                'teacher_params': teacher_params, 'sparse_nonzero': sparse_nz,
                'compact_params': compact_params_gfcs,
                'compression': teacher_params / compact_params_gfcs,
                'speedup': speedup_gfcs,
                'convert_time': convert_time_gfcs,
                'finetune_time': ft_time_gfcs,
                'merged_hiddens': gfcs_info['merged_hiddens'],
                'quality_ok': quality_ok, 'speedup_ok': speedup_ok,
                'sparsity': actual_sp,
                't_sparse_ms': t_sparse * 1000,
                't_compact_ms': t_compact_gfcs * 1000,
            }
            all_results.append(result_gfcs)
            
            # ─── Baseline: Neuron Removal ───
            print(f"\n  --- Baseline: Neuron Removal ---")
            t0 = time.time()
            compact_nr = convert_neuron_removal(sparse_model, n_classes=n_classes)
            convert_time_nr = time.time() - t0
            
            _, pre_ft_f1_nr, _ = evaluate(compact_nr, test_dl)
            train_model(compact_nr, train_dl, epochs=ft_epochs, lr=ft_lr)
            _, final_f1_nr, _ = evaluate(compact_nr, test_dl)
            compact_params_nr = compact_nr.count_params()
            t_compact_nr = measure_inference_time(compact_nr, test_dl)
            speedup_nr = t_sparse / t_compact_nr
            delta_f1_nr = final_f1_nr - sparse_f1
            
            flag = '✅' if (delta_f1_nr >= -0.02 and speedup_nr > 1.0) else '❌'
            print(f"  {flag} NeuronRemoval: F1={final_f1_nr:.4f} (Δ={delta_f1_nr:+.4f}) "
                  f"speedup={speedup_nr:.2f}× params={compact_params_nr:,}")
            
            result_nr = {
                'dataset': dataset, 'seed': seed, 'method': 'neuron_removal',
                'teacher_f1': teacher_f1, 'sparse_f1': sparse_f1,
                'pre_ft_f1': pre_ft_f1_nr, 'final_f1': final_f1_nr,
                'delta_f1': delta_f1_nr,
                'teacher_params': teacher_params, 'sparse_nonzero': sparse_nz,
                'compact_params': compact_params_nr,
                'compression': teacher_params / compact_params_nr,
                'speedup': speedup_nr,
                'convert_time': convert_time_nr,
                'sparsity': actual_sp,
                't_sparse_ms': t_sparse * 1000,
                't_compact_ms': t_compact_nr * 1000,
            }
            all_results.append(result_nr)
    
    # ─── Summary ───
    print(f"\n{'='*80}")
    print("EXPLORATION 1 SUMMARY — Tier 1 (moons, circles)")
    print(f"{'='*80}")
    print(f"{'Dataset':10s} {'Seed':>5s} {'Method':20s} {'F1':>6s} {'ΔF1':>7s} {'Speedup':>8s} {'Compress':>9s} {'Status':>7s}")
    print('-' * 80)
    
    for r in all_results:
        status = '✅' if r.get('quality_ok', r.get('delta_f1', 0) >= -0.02) and r.get('speedup_ok', r.get('speedup', 0) > 1.0) else '❌'
        print(f"{r['dataset']:10s} {r['seed']:5d} {r['method']:20s} "
              f"{r['final_f1']:6.4f} {r.get('delta_f1', 0):+7.4f} "
              f"{r.get('speedup', 0):7.2f}× "
              f"{r['compression']:8.1f}× {status:>7s}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    out_path = 'results/exploration1_tier1.json'
    with open(out_path, 'w') as f:
        json.dump({
            'exploration': 1,
            'tier': 1,
            'datasets': datasets,
            'seeds': seeds,
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    
    return all_results


if __name__ == '__main__':
    results = run_tier1()
