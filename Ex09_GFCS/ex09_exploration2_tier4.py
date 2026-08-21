#!/usr/bin/env python3
"""
Ex09 Exploration 2: Scaling Tier 4 — GFCS on highdim (50D), sequence_cls (16D)
================================================================================
Protocol §4.2: Tier 4 — different connectivity, high-dimensional inputs.
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


# Dataset configs (input_dim must match architecture)
DATASETS = {
    'highdim': {'input_dim': 50},
    'sequence_cls': {'input_dim': 16},
}

def run_tier4():
    seeds = [42, 123]
    ft_epochs = 10
    ft_lr = 0.01
    all_results = []
    break_detected = False
    
    for dataset, dcfg in DATASETS.items():
        input_dim = dcfg['input_dim']
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset} ({input_dim}D), Seed: {seed}")
            print(f"{'='*60}")
            
            set_seed(seed)
            train_dl, val_dl, test_dl, n_classes = get_dataloaders(seed, dataset, batch_size=64)
            
            teacher_path = f"data/{dataset}/teacher_seed{seed}.pt"
            if not os.path.exists(teacher_path):
                print(f"  ERROR: Teacher not found at {teacher_path}")
                continue
                
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            _, teacher_f1, _ = evaluate(teacher, test_dl)
            teacher_params = teacher.count_params()
            print(f"  Teacher: F1={teacher_f1:.4f}, params={teacher_params:,}")
            
            sparse_path, _ = find_best_sparse(dataset, seed)
            if sparse_path is None:
                print(f"  ERROR: No sparse model")
                continue
            
            sparse_model = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse_model.load_state_dict(torch.load(sparse_path, weights_only=True))
            actual_sp = get_sparsity(sparse_model)
            _, sparse_f1, _ = evaluate(sparse_model, test_dl)
            sparse_nz = sparse_model.count_nonzero()
            print(f"  Sparse: F1={sparse_f1:.4f}, sp={actual_sp:.1%}, nnz={sparse_nz:,}")
            
            t_sparse = measure_inference_time(sparse_model, test_dl)
            
            # GFCS
            print(f"\n  --- GFCS ---")
            t0 = time.time()
            compact_gfcs, gfcs_info = gfcs_convert(sparse_model, n_classes=n_classes)
            convert_time = time.time() - t0
            
            _, pre_ft_f1, _ = evaluate(compact_gfcs, test_dl)
            print(f"  pre-FT F1={pre_ft_f1:.4f}, hiddens={gfcs_info['merged_hiddens']}")
            
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
            
            if not quality_ok or not speedup_ok:
                break_detected = True
                if not quality_ok:
                    print("  ⚠️ BREAK: quality")
                if not speedup_ok:
                    print("  ⚠️ BREAK: speedup")
            
            all_results.append({
                'dataset': dataset, 'seed': seed, 'method': 'gfcs',
                'teacher_f1': teacher_f1, 'sparse_f1': sparse_f1,
                'pre_ft_f1': pre_ft_f1, 'final_f1': final_f1,
                'delta_f1': delta_f1, 'compact_params': compact_params,
                'compression': teacher_params / compact_params,
                'speedup': speedup, 'quality_ok': quality_ok,
                'speedup_ok': speedup_ok, 'sparsity': actual_sp,
                'merged_hiddens': gfcs_info['merged_hiddens'],
                'convert_time': convert_time,
            })
            
            # Baseline
            compact_nr = convert_neuron_removal(sparse_model, n_classes)
            train_model(compact_nr, train_dl, epochs=ft_epochs, lr=ft_lr)
            _, f1_nr, _ = evaluate(compact_nr, test_dl)
            p_nr = compact_nr.count_params()
            t_nr = measure_inference_time(compact_nr, test_dl)
            sp_nr = t_sparse / t_nr
            d_nr = f1_nr - sparse_f1
            flag_nr = '✅' if (d_nr >= -0.05 and sp_nr > 1.0) else '❌'
            print(f"  {flag_nr} NR:   F1={f1_nr:.4f} (Δ={d_nr:+.4f}) "
                  f"speedup={sp_nr:.2f}× compress=×{teacher_params/p_nr:.1f}")
            all_results.append({
                'dataset': dataset, 'seed': seed, 'method': 'neuron_removal',
                'sparse_f1': sparse_f1, 'final_f1': f1_nr, 'delta_f1': d_nr,
                'compact_params': p_nr, 'compression': teacher_params/p_nr,
                'speedup': sp_nr, 'sparsity': actual_sp,
            })
    
    # Cross-seed divergence
    print(f"\n{'='*60}")
    print("CROSS-SEED DIVERGENCE")
    for ds in DATASETS:
        gr = [r for r in all_results if r['dataset'] == ds and r['method'] == 'gfcs']
        if len(gr) >= 2:
            d = abs(gr[0]['final_f1'] - gr[1]['final_f1'])
            st = '✅' if d <= 0.1 else '❌ BREAK'
            print(f"  {ds}: {gr[0]['final_f1']:.4f} vs {gr[1]['final_f1']:.4f} div={d:.4f} {st}")
            if d > 0.1:
                break_detected = True
    
    print(f"\n{'='*80}")
    print("TIER 4 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':15s} {'Seed':>5s} {'Method':10s} {'Sp':>5s} {'F1':>6s} {'ΔF1':>7s} {'Spdup':>6s} {'Cmpr':>6s}")
    print('-' * 70)
    for r in all_results:
        print(f"{r['dataset']:15s} {r['seed']:5d} {r['method']:10s} "
              f"{r['sparsity']:4.0%} {r['final_f1']:6.4f} {r['delta_f1']:+7.4f} "
              f"{r['speedup']:5.2f}× {r['compression']:5.1f}×")
    
    status = '⚠️ BREAK' if break_detected else '✅ ALL TIERS PASSED → SYNTHESIS'
    print(f"\n{status}")
    
    out = 'results/exploration2_tier4.json'
    os.makedirs('results', exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'exploration': 2, 'tier': 4, 'break_detected': break_detected,
                   'results': all_results}, f, indent=2, default=str)
    print(f"Saved: {out}")
    return all_results, break_detected

if __name__ == '__main__':
    run_tier4()
