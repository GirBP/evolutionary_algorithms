#!/usr/bin/env python3
"""
Ex09 Exploration 2: Scaling Tier 2 — GFCS on spirals, blobs
=============================================================
Protocol §4: Progressive scaling. 
Test GFCS on Tier 2 datasets after passing Tier 1.

Break criteria §4.3:
  - speedup ≤ 1.0
  - Q(f_C) < Q(f_S) − 0.05
  - Divergence between seeds > 0.1 F1

Uses the BEST available sparse model per dataset/seed.
"""
import sys, os, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP, CompactMLP,
    train_model, evaluate, get_sparsity,
    convert_neuron_removal,
)
from ex09_lib.gfcs import gfcs_convert


def measure_inference_time(model, test_dl, n_repeats=50):
    """Measure real inference time (RCU)."""
    model.eval()
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


def find_best_sparse(dataset, seed):
    """Find the sparse model with highest sparsity for a dataset/seed."""
    data_dir = f"data/{dataset}"
    sparse_files = [f for f in os.listdir(data_dir) 
                    if f.startswith(f'sparse_seed{seed}_sp')]
    if not sparse_files:
        return None, None
    
    # Parse sparsity from filename and pick highest
    def get_sp(f):
        try:
            return float(f.split('_sp')[1].replace('.pt', ''))
        except:
            return 0
    
    sparse_files.sort(key=get_sp, reverse=True)
    best = sparse_files[0]
    sp = get_sp(best)
    return f"{data_dir}/{best}", sp


def run_tier2():
    """Run Exploration 2 on Tier 2 datasets (spirals, blobs)."""
    datasets_config = {
        'spirals': {'input_dim': 2},
        'blobs': {'input_dim': 2},
    }
    seeds = [42, 123]
    ft_epochs = 10
    ft_lr = 0.01
    
    all_results = []
    break_detected = False
    
    for dataset, dcfg in datasets_config.items():
        input_dim = dcfg['input_dim']
        
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset}, Seed: {seed}")
            print(f"{'='*60}")
            
            set_seed(seed)
            train_dl, val_dl, test_dl, n_classes = get_dataloaders(seed, dataset, batch_size=64)
            
            # Load teacher
            teacher_path = f"data/{dataset}/teacher_seed{seed}.pt"
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            _, teacher_f1, teacher_acc = evaluate(teacher, test_dl)
            teacher_params = teacher.count_params()
            print(f"  Teacher: F1={teacher_f1:.4f}, acc={teacher_acc:.4f}, params={teacher_params:,}")
            
            # Find best sparse model
            sparse_path, nominal_sp = find_best_sparse(dataset, seed)
            if sparse_path is None:
                print(f"  ERROR: No sparse model found")
                continue
            
            sparse_model = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse_model.load_state_dict(torch.load(sparse_path, weights_only=True))
            actual_sp = get_sparsity(sparse_model)
            _, sparse_f1, sparse_acc = evaluate(sparse_model, test_dl)
            sparse_nz = sparse_model.count_nonzero()
            print(f"  Sparse: F1={sparse_f1:.4f}, sp={actual_sp:.1%}, nonzero={sparse_nz:,}")
            print(f"  Sparse file: {os.path.basename(sparse_path)}")
            
            # Measure sparse inference time
            t_sparse = measure_inference_time(sparse_model, test_dl)
            print(f"  Sparse inference time: {t_sparse*1000:.2f}ms")
            
            # ─── GFCS ───
            print(f"\n  --- GFCS ---")
            t0 = time.time()
            compact_gfcs, gfcs_info = gfcs_convert(sparse_model, n_classes=n_classes)
            convert_time = time.time() - t0
            
            _, pre_ft_f1, _ = evaluate(compact_gfcs, test_dl)
            print(f"  GFCS pre-finetune: F1={pre_ft_f1:.4f}")
            print(f"  GFCS merged hiddens: {gfcs_info['merged_hiddens']}")
            print(f"  GFCS compression: ×{gfcs_info['compression']:.1f}")
            
            # Fine-tune
            t1 = time.time()
            train_model(compact_gfcs, train_dl, epochs=ft_epochs, lr=ft_lr)
            ft_time = time.time() - t1
            
            _, final_f1, final_acc = evaluate(compact_gfcs, test_dl)
            compact_params = compact_gfcs.count_params()
            t_compact = measure_inference_time(compact_gfcs, test_dl)
            speedup = t_sparse / t_compact
            delta_f1 = final_f1 - sparse_f1
            
            quality_ok = delta_f1 >= -0.05  # §4.3 threshold for Tier 2
            speedup_ok = speedup > 1.0
            
            flag = '✅' if (quality_ok and speedup_ok) else '❌'
            print(f"  {flag} GFCS: F1={final_f1:.4f} (Δ={delta_f1:+.4f}) "
                  f"speedup={speedup:.2f}× params={compact_params:,} "
                  f"t_convert={convert_time:.2f}s")
            
            # Check break criteria
            if not quality_ok:
                print(f"  ⚠️ BREAK CRITERION: Q(f_C) < Q(f_S) − 0.05")
                break_detected = True
            if not speedup_ok:
                print(f"  ⚠️ BREAK CRITERION: speedup ≤ 1.0")
                break_detected = True
            
            result = {
                'dataset': dataset, 'seed': seed, 'method': 'gfcs',
                'teacher_f1': teacher_f1, 'sparse_f1': sparse_f1,
                'pre_ft_f1': pre_ft_f1, 'final_f1': final_f1,
                'delta_f1': delta_f1,
                'teacher_params': teacher_params, 'sparse_nonzero': sparse_nz,
                'compact_params': compact_params,
                'compression': teacher_params / compact_params,
                'speedup': speedup,
                'convert_time': convert_time, 'finetune_time': ft_time,
                'merged_hiddens': gfcs_info['merged_hiddens'],
                'quality_ok': quality_ok, 'speedup_ok': speedup_ok,
                'sparsity': actual_sp,
                't_sparse_ms': t_sparse * 1000,
                't_compact_ms': t_compact * 1000,
            }
            all_results.append(result)
            
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
            
            flag_nr = '✅' if (delta_f1_nr >= -0.05 and speedup_nr > 1.0) else '❌'
            print(f"  {flag_nr} NeuronRemoval: F1={final_f1_nr:.4f} (Δ={delta_f1_nr:+.4f}) "
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
    
    # ─── Cross-seed divergence check §4.3 ───
    print(f"\n{'='*60}")
    print("CROSS-SEED DIVERGENCE CHECK")
    print(f"{'='*60}")
    for dataset in datasets_config:
        gfcs_results = [r for r in all_results 
                       if r['dataset'] == dataset and r['method'] == 'gfcs']
        if len(gfcs_results) >= 2:
            f1s = [r['final_f1'] for r in gfcs_results]
            divergence = abs(f1s[0] - f1s[1])
            status = '✅' if divergence <= 0.1 else '❌ BREAK: divergence > 0.1'
            print(f"  {dataset}: seed42 F1={f1s[0]:.4f}, seed123 F1={f1s[1]:.4f}, "
                  f"divergence={divergence:.4f} {status}")
            if divergence > 0.1:
                break_detected = True
    
    # ─── Summary ───
    print(f"\n{'='*80}")
    print("EXPLORATION 2 SUMMARY — Tier 2 (spirals, blobs)")
    print(f"{'='*80}")
    print(f"{'Dataset':10s} {'Seed':>5s} {'Method':20s} {'Sp':>5s} {'F1':>6s} {'ΔF1':>7s} "
          f"{'Speedup':>8s} {'Compress':>9s} {'Status':>7s}")
    print('-' * 85)
    
    for r in all_results:
        status = '✅' if r.get('quality_ok', True) and r.get('speedup_ok', True) else '❌'
        if r['method'] == 'neuron_removal':
            status = '✅' if (r['delta_f1'] >= -0.05 and r.get('speedup', 0) > 1.0) else '❌'
        print(f"{r['dataset']:10s} {r['seed']:5d} {r['method']:20s} "
              f"{r['sparsity']:4.0%} {r['final_f1']:6.4f} {r['delta_f1']:+7.4f} "
              f"{r.get('speedup', 0):7.2f}× "
              f"{r['compression']:8.1f}× {status:>7s}")
    
    if break_detected:
        print(f"\n⚠️  BREAK DETECTED — may need Exploration 3 (Impossibility) or Reframing")
    else:
        print(f"\n✅ ALL TIER 2 PASSED — ready for Exploration 2 Tier 3")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    out_path = 'results/exploration2_tier2.json'
    with open(out_path, 'w') as f:
        json.dump({
            'exploration': 2,
            'tier': 2,
            'datasets': list(datasets_config.keys()),
            'seeds': seeds,
            'break_detected': break_detected,
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    
    return all_results, break_detected


if __name__ == '__main__':
    results, broke = run_tier2()
