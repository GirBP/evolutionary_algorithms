#!/usr/bin/env python3
"""
Ex09 Exploration 2: Scaling Tier 3 — GFCS on gaussian_quantiles, classification
=================================================================================
Protocol §4.2: Tier 3 — more classes, lower sparsity.

Break criteria §4.3:
  - speedup ≤ 1.0
  - Q(f_C) < Q(f_S) − 0.05
  - Divergence between seeds > 0.1 F1
"""
import sys, os, json, time
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP,
    train_model, evaluate, get_sparsity,
    convert_neuron_removal,
)
from ex09_lib.gfcs import gfcs_convert


def measure_inference_time(model, test_dl, n_repeats=50):
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
    data_dir = f"data/{dataset}"
    sparse_files = [f for f in os.listdir(data_dir) 
                    if f.startswith(f'sparse_seed{seed}_sp')]
    if not sparse_files:
        return None, None
    def get_sp(f):
        try:
            return float(f.split('_sp')[1].replace('.pt', ''))
        except:
            return 0
    sparse_files.sort(key=get_sp, reverse=True)
    best = sparse_files[0]
    return f"{data_dir}/{best}", get_sp(best)


def run_tier3():
    datasets_config = {
        'gaussian_quantiles': {'input_dim': 2},
        'classification': {'input_dim': 2},
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
            
            teacher_path = f"data/{dataset}/teacher_seed{seed}.pt"
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            _, teacher_f1, _ = evaluate(teacher, test_dl)
            teacher_params = teacher.count_params()
            print(f"  Teacher: F1={teacher_f1:.4f}, params={teacher_params:,}")
            
            sparse_path, nominal_sp = find_best_sparse(dataset, seed)
            if sparse_path is None:
                print(f"  ERROR: No sparse model found")
                continue
            
            sparse_model = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse_model.load_state_dict(torch.load(sparse_path, weights_only=True))
            actual_sp = get_sparsity(sparse_model)
            _, sparse_f1, _ = evaluate(sparse_model, test_dl)
            sparse_nz = sparse_model.count_nonzero()
            print(f"  Sparse: F1={sparse_f1:.4f}, sp={actual_sp:.1%}, nonzero={sparse_nz:,}")
            
            t_sparse = measure_inference_time(sparse_model, test_dl)
            
            # ─── GFCS ───
            print(f"\n  --- GFCS ---")
            t0 = time.time()
            compact_gfcs, gfcs_info = gfcs_convert(sparse_model, n_classes=n_classes)
            convert_time = time.time() - t0
            
            _, pre_ft_f1, _ = evaluate(compact_gfcs, test_dl)
            print(f"  GFCS pre-finetune: F1={pre_ft_f1:.4f}")
            print(f"  GFCS merged hiddens: {gfcs_info['merged_hiddens']}")
            
            train_model(compact_gfcs, train_dl, epochs=ft_epochs, lr=ft_lr)
            _, final_f1, _ = evaluate(compact_gfcs, test_dl)
            compact_params = compact_gfcs.count_params()
            t_compact = measure_inference_time(compact_gfcs, test_dl)
            speedup = t_sparse / t_compact
            delta_f1 = final_f1 - sparse_f1
            quality_ok = delta_f1 >= -0.05
            speedup_ok = speedup > 1.0
            
            flag = '✅' if (quality_ok and speedup_ok) else '❌'
            print(f"  {flag} GFCS: F1={final_f1:.4f} (Δ={delta_f1:+.4f}) "
                  f"speedup={speedup:.2f}× compress=×{teacher_params/compact_params:.1f}")
            
            if not quality_ok:
                print(f"  ⚠️ BREAK: quality")
                break_detected = True
            if not speedup_ok:
                print(f"  ⚠️ BREAK: speedup")
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
                'convert_time': convert_time,
                'merged_hiddens': gfcs_info['merged_hiddens'],
                'quality_ok': quality_ok, 'speedup_ok': speedup_ok,
                'sparsity': actual_sp,
            }
            all_results.append(result)
            
            # ─── Baseline: NeuronRemoval ───
            compact_nr = convert_neuron_removal(sparse_model, n_classes=n_classes)
            train_model(compact_nr, train_dl, epochs=ft_epochs, lr=ft_lr)
            _, final_f1_nr, _ = evaluate(compact_nr, test_dl)
            compact_params_nr = compact_nr.count_params()
            t_compact_nr = measure_inference_time(compact_nr, test_dl)
            speedup_nr = t_sparse / t_compact_nr
            delta_f1_nr = final_f1_nr - sparse_f1
            flag_nr = '✅' if (delta_f1_nr >= -0.05 and speedup_nr > 1.0) else '❌'
            print(f"  {flag_nr} NR: F1={final_f1_nr:.4f} (Δ={delta_f1_nr:+.4f}) "
                  f"speedup={speedup_nr:.2f}× compress=×{teacher_params/compact_params_nr:.1f}")
            
            all_results.append({
                'dataset': dataset, 'seed': seed, 'method': 'neuron_removal',
                'teacher_f1': teacher_f1, 'sparse_f1': sparse_f1,
                'final_f1': final_f1_nr, 'delta_f1': delta_f1_nr,
                'compact_params': compact_params_nr,
                'compression': teacher_params / compact_params_nr,
                'speedup': speedup_nr, 'sparsity': actual_sp,
            })
    
    # Cross-seed divergence
    print(f"\n{'='*60}")
    print("CROSS-SEED DIVERGENCE CHECK")
    print(f"{'='*60}")
    for dataset in datasets_config:
        gfcs_r = [r for r in all_results if r['dataset'] == dataset and r['method'] == 'gfcs']
        if len(gfcs_r) >= 2:
            f1s = [r['final_f1'] for r in gfcs_r]
            div = abs(f1s[0] - f1s[1])
            st = '✅' if div <= 0.1 else '❌ BREAK'
            print(f"  {dataset}: F1={f1s[0]:.4f} vs {f1s[1]:.4f}, div={div:.4f} {st}")
            if div > 0.1:
                break_detected = True
    
    # Summary
    print(f"\n{'='*80}")
    print("EXPLORATION 2 — Tier 3 SUMMARY (gaussian_quantiles, classification)")
    print(f"{'='*80}")
    print(f"{'Dataset':20s} {'Seed':>5s} {'Method':15s} {'Sp':>5s} {'F1':>6s} {'ΔF1':>7s} {'Spdup':>6s} {'Cmpr':>6s}")
    print('-' * 80)
    for r in all_results:
        print(f"{r['dataset']:20s} {r['seed']:5d} {r['method']:15s} "
              f"{r['sparsity']:4.0%} {r['final_f1']:6.4f} {r['delta_f1']:+7.4f} "
              f"{r['speedup']:5.2f}× {r['compression']:5.1f}×")
    
    status = '⚠️ BREAK DETECTED' if break_detected else '✅ ALL PASSED'
    print(f"\n{status}")
    
    out_path = 'results/exploration2_tier3.json'
    os.makedirs('results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'exploration': 2, 'tier': 3, 'break_detected': break_detected,
                   'results': all_results}, f, indent=2, default=str)
    print(f"Saved: {out_path}")
    return all_results, break_detected


if __name__ == '__main__':
    run_tier3()
