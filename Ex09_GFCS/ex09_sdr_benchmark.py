#!/usr/bin/env python3
"""
Ex09 SDR Benchmark: SDR-Magnitude vs SDR-EvoSF vs GFCS vs Neuron Removal
==========================================================================
Compares Sparse-to-Dense Restructuring (SDR) with existing conversion methods.

Goal: Prove that EA-optimized pruning ratios produce BETTER compact topology
than Magnitude-based ratios at the SAME model size.

Metrics:
  - F1 macro (quality)
  - Actual params (real model size, not nominal sparsity)
  - Inference time (wall-clock CPU)
  - FLOP count
  - Compression ratio (teacher_params / compact_params)

Datasets: all 8 from Ex09 protocol
Seeds: 42, 123
"""
import sys, os, json, time, csv
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP, CompactMLP,
    train_model, evaluate, get_sparsity,
    convert_neuron_removal,
)
from ex09_lib.gfcs import gfcs_convert
from ex09_lib.sdr import sdr_convert
from ex09_synthesis import measure_inference_time, count_flops, DATASETS, find_best_sparse


def run_sdr_benchmark():
    """Run SDR benchmark on all datasets × seeds."""
    seeds = [42, 123]
    ft_epochs = 10
    ft_lr = 0.01
    kd_epochs = 20
    
    all_results = []
    
    print("=" * 100)
    print("  SDR BENCHMARK: SDR-Magnitude vs SDR-EvoSF vs GFCS vs NeuronRemoval")
    print("=" * 100)
    
    for dataset, dcfg in DATASETS.items():
        input_dim = dcfg['input_dim']
        
        for seed in seeds:
            print(f"\n{'─'*80}")
            print(f"  {dataset} (Tier {dcfg['tier']}, {input_dim}D), seed={seed}")
            print(f"{'─'*80}")
            
            set_seed(seed)
            train_dl, val_dl, test_dl, n_classes = get_dataloaders(
                seed, dataset, batch_size=64
            )
            
            # ─── Load teacher ───
            teacher_path = f"data/{dataset}/teacher_seed{seed}.pt"
            if not os.path.exists(teacher_path):
                print(f"  ⚠️  Teacher not found — skipping")
                continue
            teacher = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            _, teacher_f1, _ = evaluate(teacher, test_dl)
            teacher_params = teacher.count_params()
            teacher_flops = count_flops(teacher, input_dim)
            
            # ─── Load sparse model ───
            sparse_path = find_best_sparse(dataset, seed)
            if sparse_path is None:
                print(f"  ⚠️  Sparse model not found — skipping")
                continue
            
            sparse_model = SimpleMLP(input_dim=input_dim, n_classes=n_classes)
            sparse_model.load_state_dict(torch.load(sparse_path, weights_only=True))
            actual_sp = get_sparsity(sparse_model)
            _, sparse_f1, _ = evaluate(sparse_model, test_dl)
            t_sparse_ms = measure_inference_time(sparse_model, test_dl, n_repeats=50)
            
            print(f"  Teacher: F1={teacher_f1:.4f}  params={teacher_params:,}")
            print(f"  Sparse:  F1={sparse_f1:.4f}  sp={actual_sp:.0%}  t={t_sparse_ms:.3f}ms")
            
            # ═══════════════════════════════════════
            #  Method 1: Neuron Removal (baseline)
            # ═══════════════════════════════════════
            set_seed(seed)
            t0 = time.time()
            nr_model = convert_neuron_removal(sparse_model, n_classes)
            train_model(nr_model, train_dl, epochs=ft_epochs, lr=ft_lr)
            nr_time = time.time() - t0
            _, nr_f1, _ = evaluate(nr_model, test_dl)
            nr_params = nr_model.count_params()
            nr_flops = count_flops(nr_model, input_dim)
            t_nr_ms = measure_inference_time(nr_model, test_dl, n_repeats=50)
            
            nr_result = {
                'dataset': dataset, 'seed': seed, 'method': 'NeuronRemoval',
                'f1': round(nr_f1, 4), 'delta_f1': round(nr_f1 - sparse_f1, 4),
                'params': nr_params, 'flops': nr_flops,
                'compression': round(teacher_params / max(nr_params, 1), 1),
                'speedup_real': round(t_sparse_ms / t_nr_ms, 2),
                'speedup_flops': round(teacher_flops / max(nr_flops, 1), 1),
                'time_s': round(nr_time, 3),
                'hiddens': str([m.out_features for m in nr_model.net 
                              if isinstance(m, torch.nn.Linear)][:-1]),
            }
            all_results.append(nr_result)
            print(f"\n  NR:        F1={nr_f1:.4f} (Δ={nr_f1-sparse_f1:+.4f})  "
                  f"params={nr_params:,}  comp={nr_result['compression']}×  "
                  f"speed={nr_result['speedup_real']}×")
            
            # ═══════════════════════════════════════
            #  Method 2: GFCS (existing)
            # ═══════════════════════════════════════
            set_seed(seed)
            t0 = time.time()
            gfcs_model, gfcs_info = gfcs_convert(sparse_model, n_classes=n_classes)
            train_model(gfcs_model, train_dl, epochs=ft_epochs, lr=ft_lr)
            gfcs_time = time.time() - t0
            _, gfcs_f1, _ = evaluate(gfcs_model, test_dl)
            gfcs_params = gfcs_model.count_params()
            gfcs_flops = count_flops(gfcs_model, input_dim)
            t_gfcs_ms = measure_inference_time(gfcs_model, test_dl, n_repeats=50)
            
            gfcs_result = {
                'dataset': dataset, 'seed': seed, 'method': 'GFCS',
                'f1': round(gfcs_f1, 4), 'delta_f1': round(gfcs_f1 - sparse_f1, 4),
                'params': gfcs_params, 'flops': gfcs_flops,
                'compression': round(teacher_params / max(gfcs_params, 1), 1),
                'speedup_real': round(t_sparse_ms / t_gfcs_ms, 2),
                'speedup_flops': round(teacher_flops / max(gfcs_flops, 1), 1),
                'time_s': round(gfcs_time, 3),
                'hiddens': str(gfcs_info['merged_hiddens']),
            }
            all_results.append(gfcs_result)
            print(f"  GFCS:      F1={gfcs_f1:.4f} (Δ={gfcs_f1-sparse_f1:+.4f})  "
                  f"params={gfcs_params:,}  comp={gfcs_result['compression']}×  "
                  f"speed={gfcs_result['speedup_real']}×")
            
            # ═══════════════════════════════════════
            #  Method 3: SDR-Magnitude (our baseline)
            # ═══════════════════════════════════════
            set_seed(seed)
            t0 = time.time()
            sdr_mag_model, sdr_mag_info = sdr_convert(
                sparse_model, teacher, train_dl, n_classes=n_classes,
                method='magnitude', kd_epochs=kd_epochs, ft_epochs=ft_epochs, ft_lr=ft_lr
            )
            sdr_mag_time = time.time() - t0
            _, sdr_mag_f1, _ = evaluate(sdr_mag_model, test_dl)
            sdr_mag_params = sdr_mag_info['compact_params']
            sdr_mag_flops = count_flops(sdr_mag_model, input_dim)
            t_sdr_mag_ms = measure_inference_time(sdr_mag_model, test_dl, n_repeats=50)
            
            sdr_mag_result = {
                'dataset': dataset, 'seed': seed, 'method': 'SDR-Magnitude',
                'f1': round(sdr_mag_f1, 4), 'delta_f1': round(sdr_mag_f1 - sparse_f1, 4),
                'params': sdr_mag_params, 'flops': sdr_mag_flops,
                'compression': round(teacher_params / max(sdr_mag_params, 1), 1),
                'speedup_real': round(t_sparse_ms / t_sdr_mag_ms, 2),
                'speedup_flops': round(teacher_flops / max(sdr_mag_flops, 1), 1),
                'time_s': round(sdr_mag_time, 3),
                'hiddens': str(sdr_mag_info['compact_hiddens']),
                'ratios': str(sdr_mag_info['ratios']),
            }
            all_results.append(sdr_mag_result)
            print(f"  SDR-Mag:   F1={sdr_mag_f1:.4f} (Δ={sdr_mag_f1-sparse_f1:+.4f})  "
                  f"params={sdr_mag_params:,}  comp={sdr_mag_result['compression']}×  "
                  f"speed={sdr_mag_result['speedup_real']}×  "
                  f"hiddens={sdr_mag_info['compact_hiddens']}")
            
            # ═══════════════════════════════════════
            #  Method 4: SDR-EvoSF (our method) ← KEY
            # ═══════════════════════════════════════
            set_seed(seed)
            t0 = time.time()
            sdr_evo_model, sdr_evo_info = sdr_convert(
                sparse_model, teacher, train_dl, n_classes=n_classes,
                method='evo', kd_epochs=kd_epochs, ft_epochs=ft_epochs, ft_lr=ft_lr
            )
            sdr_evo_time = time.time() - t0
            _, sdr_evo_f1, _ = evaluate(sdr_evo_model, test_dl)
            sdr_evo_params = sdr_evo_info['compact_params']
            sdr_evo_flops = count_flops(sdr_evo_model, input_dim)
            t_sdr_evo_ms = measure_inference_time(sdr_evo_model, test_dl, n_repeats=50)
            
            sdr_evo_result = {
                'dataset': dataset, 'seed': seed, 'method': 'SDR-EvoSF',
                'f1': round(sdr_evo_f1, 4), 'delta_f1': round(sdr_evo_f1 - sparse_f1, 4),
                'params': sdr_evo_params, 'flops': sdr_evo_flops,
                'compression': round(teacher_params / max(sdr_evo_params, 1), 1),
                'speedup_real': round(t_sparse_ms / t_sdr_evo_ms, 2),
                'speedup_flops': round(teacher_flops / max(sdr_evo_flops, 1), 1),
                'time_s': round(sdr_evo_time, 3),
                'hiddens': str(sdr_evo_info['compact_hiddens']),
                'ratios': str(sdr_evo_info['ratios']),
            }
            all_results.append(sdr_evo_result)
            
            # Highlight if SDR-EvoSF beats SDR-Magnitude
            evo_wins = sdr_evo_f1 > sdr_mag_f1
            win_marker = " ★ WINS" if evo_wins else ""
            print(f"  SDR-Evo:   F1={sdr_evo_f1:.4f} (Δ={sdr_evo_f1-sparse_f1:+.4f})  "
                  f"params={sdr_evo_params:,}  comp={sdr_evo_result['compression']}×  "
                  f"speed={sdr_evo_result['speedup_real']}×  "
                  f"hiddens={sdr_evo_info['compact_hiddens']}{win_marker}")
    
    # ═══════════════════════════════════════
    #  Summary Table
    # ═══════════════════════════════════════
    print(f"\n\n{'='*100}")
    print("  SUMMARY: SDR-EvoSF vs SDR-Magnitude — Head-to-Head")
    print(f"{'='*100}")
    
    methods_of_interest = ['SDR-Magnitude', 'SDR-EvoSF']
    evo_wins = 0
    mag_wins = 0
    total_comparisons = 0
    
    datasets_seen = set()
    for r in all_results:
        if r['method'] == 'SDR-EvoSF':
            ds_key = (r['dataset'], r['seed'])
            if ds_key in datasets_seen:
                continue
            datasets_seen.add(ds_key)
            
            # Find corresponding Magnitude result
            mag_r = next((x for x in all_results 
                         if x['dataset'] == r['dataset'] and x['seed'] == r['seed'] 
                         and x['method'] == 'SDR-Magnitude'), None)
            if mag_r is None:
                continue
            
            total_comparisons += 1
            if r['f1'] > mag_r['f1']:
                evo_wins += 1
                marker = "★ EVO"
            else:
                mag_wins += 1
                marker = "  MAG"
            
            print(f"  {marker}  {r['dataset']:20s} s{r['seed']}  "
                  f"Mag F1={mag_r['f1']:.4f} ({mag_r['params']}p)  vs  "
                  f"Evo F1={r['f1']:.4f} ({r['params']}p)  "
                  f"ΔF1={r['f1']-mag_r['f1']:+.4f}")
    
    print(f"\n  SDR-EvoSF wins: {evo_wins}/{total_comparisons}")
    print(f"  SDR-Magnitude wins: {mag_wins}/{total_comparisons}")
    
    if evo_wins > mag_wins:
        print("  ✅ HYPOTHESIS CONFIRMED: EA finds better topology than Magnitude")
    else:
        print("  ⚠️  HYPOTHESIS NOT CONFIRMED: Need investigation")
    
    # ═══════════════════════════════════════
    #  Full comparison table
    # ═══════════════════════════════════════
    print(f"\n\n{'='*100}")
    print("  ALL METHODS — Averaged by Dataset")
    print(f"{'='*100}")
    
    print(f"\n{'Dataset':20s} {'Method':16s} {'F1':<8s} {'ΔF1':<8s} {'Params':>8s} "
          f"{'Comp×':>6s} {'Real×':>6s} {'FLOP×':>6s}")
    print('─' * 90)
    
    # Group by dataset
    ds_methods = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        ds_methods[r['dataset']][r['method']].append(r)
    
    for ds in DATASETS:
        first = True
        for method in ['NeuronRemoval', 'GFCS', 'SDR-Magnitude', 'SDR-EvoSF']:
            results = ds_methods.get(ds, {}).get(method, [])
            if not results:
                continue
            avg_f1 = np.mean([r['f1'] for r in results])
            avg_df1 = np.mean([r['delta_f1'] for r in results])
            avg_params = int(np.mean([r['params'] for r in results]))
            avg_comp = np.mean([r['compression'] for r in results])
            avg_real = np.mean([r['speedup_real'] for r in results])
            avg_flop = np.mean([r['speedup_flops'] for r in results])
            
            ds_label = ds if first else ""
            first = False
            print(f"{ds_label:20s} {method:16s} {avg_f1:.4f}  {avg_df1:+.4f}  "
                  f"{avg_params:>8,}  {avg_comp:5.1f}× {avg_real:5.2f}× {avg_flop:5.1f}×")
        if not first:
            print()
    
    # ═══════════════════════════════════════
    #  Save results
    # ═══════════════════════════════════════
    os.makedirs('results', exist_ok=True)
    
    out_json = 'results/sdr_benchmark.json'
    with open(out_json, 'w') as f:
        json.dump({
            'experiment': 'Ex09_SDR',
            'description': 'SDR-Magnitude vs SDR-EvoSF vs GFCS vs NeuronRemoval',
            'evo_wins': evo_wins,
            'mag_wins': mag_wins,
            'total_comparisons': total_comparisons,
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved: {out_json}")
    
    out_csv = 'results/sdr_benchmark.csv'
    fieldnames = ['dataset', 'seed', 'method', 'f1', 'delta_f1', 'params', 
                  'flops', 'compression', 'speedup_real', 'speedup_flops', 
                  'time_s', 'hiddens', 'ratios']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';', extrasaction='ignore')
        w.writeheader()
        w.writerows(all_results)
    print(f"  Saved: {out_csv}")
    
    return all_results


if __name__ == '__main__':
    run_sdr_benchmark()
