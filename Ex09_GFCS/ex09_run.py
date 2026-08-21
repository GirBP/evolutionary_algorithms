#!/usr/bin/env python3
"""
Ex09: Sparse→Dense Conversion Benchmark
========================================
Runs the full pipeline:
  1. Load/train teacher
  2. Prune at each sparsity
  3. Convert sparse→compact dense (4 methods)
  4. Fine-tune compact model
  5. Evaluate and compare

Usage:
  python3 ex09_run.py                    # default (moons)
  python3 ex09_run.py -d circles         # other datasets
  python3 ex09_run.py -d moons --quick   # fast test
"""
import sys, os, json, time, argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP, CompactMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
    convert_neuron_removal, convert_svd_compression,
    convert_knowledge_distill, convert_weight_redistribution,
    CONVERTERS,
)
from ex09_lib.config_experiment import CONFIG
from ex09_lib.evomerge import evomerge


def run_experiment(config):
    dataset = config['dataset']
    seeds = config['seeds']
    sparsities = config['sparsities']
    ft_epochs = config['finetune_epochs']
    ft_lr = config['finetune_lr']

    print(f"\n{'='*70}")
    print(f"Ex09: Sparse→Dense Conversion — {dataset}")
    print(f"Seeds: {seeds}, Sparsities: {[f'{s:.0%}' for s in sparsities]}")
    print(f"{'='*70}\n")

    all_results = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        set_seed(seed)
        train_dl, val_dl, test_dl, n_classes = get_dataloaders(seed, dataset, config['batch_size'])

        # 1. Train teacher (or load)
        teacher_path = f"data/{dataset}/teacher_seed{seed}.pt"
        os.makedirs(f"data/{dataset}", exist_ok=True)

        if os.path.exists(teacher_path):
            teacher = SimpleMLP(n_classes=n_classes)
            teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
            print(f"  Loaded teacher from {teacher_path}")
        else:
            teacher = SimpleMLP(n_classes=n_classes)
            train_model(teacher, train_dl, epochs=config['epochs_pretrain'])
            torch.save(teacher.state_dict(), teacher_path)
            print(f"  Trained & saved teacher to {teacher_path}")

        _, teacher_f1, teacher_acc = evaluate(teacher, test_dl)
        teacher_params = teacher.count_params()
        print(f"  Teacher: F1={teacher_f1:.4f} acc={teacher_acc:.4f} params={teacher_params:,}")

        for sp in sparsities:
            print(f"\n  Sparsity {sp:.0%}:")

            # 2. Prune
            sparse_model = SimpleMLP(n_classes=n_classes)
            sparse_model.load_state_dict({k: v.clone() for k, v in teacher.state_dict().items()})
            prune_magnitude_global(sparse_model, sp)
            actual_sp = get_sparsity(sparse_model)
            _, sparse_f1, _ = evaluate(sparse_model, test_dl)
            sparse_nz = sparse_model.count_nonzero()
            print(f"    Sparse: F1={sparse_f1:.4f} asp={actual_sp:.1%} nonzero={sparse_nz:,}")

            # 3. Convert with each method
            for conv_name in config.get('conversion_methods', CONVERTERS.keys()):
                t0 = time.time()

                if conv_name == 'knowledge_distill':
                    compact = convert_knowledge_distill(sparse_model, train_dl, n_classes)
                elif conv_name == 'svd_compression':
                    # rank ratio proportional to density
                    rank_ratio = max(0.1, 1.0 - sp)
                    compact = convert_svd_compression(sparse_model, rank_ratio, n_classes)
                elif conv_name == 'neuron_removal':
                    compact = convert_neuron_removal(sparse_model, n_classes)
                elif conv_name == 'weight_redistribution':
                    compact = convert_weight_redistribution(sparse_model, n_classes)
                elif conv_name == 'evomerge':
                    compact, merge_info = evomerge(
                        sparse_model, train_dl, n_classes,
                        pop_size=20, generations=30,
                        min_ratio=0.1, max_ratio=0.8,
                    )
                else:
                    continue

                convert_time = time.time() - t0

                # Pre-finetune eval
                _, pre_ft_f1, _ = evaluate(compact, test_dl)

                # 4. Fine-tune compact model
                t1 = time.time()
                train_model(compact, train_dl, epochs=ft_epochs, lr=ft_lr)
                ft_time = time.time() - t1

                # 5. Evaluate
                _, final_f1, final_acc = evaluate(compact, test_dl)
                compact_params = compact.count_params()
                compression = teacher_params / compact_params

                total_time = convert_time + ft_time

                result = {
                    'seed': seed,
                    'sparsity': sp,
                    'method': conv_name,
                    'teacher_f1': teacher_f1,
                    'sparse_f1': sparse_f1,
                    'pre_ft_f1': pre_ft_f1,
                    'final_f1': final_f1,
                    'final_acc': final_acc,
                    'teacher_params': teacher_params,
                    'sparse_nonzero': sparse_nz,
                    'compact_params': compact_params,
                    'compression': compression,
                    'convert_time': convert_time,
                    'finetune_time': ft_time,
                    'total_time': total_time,
                    'f1_recovery': final_f1 / teacher_f1 if teacher_f1 > 0 else 0,
                }
                all_results.append(result)

                recovery_pct = result['f1_recovery'] * 100
                flag = '' if recovery_pct >= 95 else ('' if recovery_pct >= 80 else '')
                print(f"    {flag} {conv_name:25s} F1={final_f1:.4f} (recover={recovery_pct:.1f}%) "
                      f"params={compact_params:>6,} (×{compression:.1f}) t={total_time:.1f}s")

    return all_results


def save_results(results, dataset):
    os.makedirs('results', exist_ok=True)
    os.makedirs(f'data/{dataset}', exist_ok=True)

    # JSON
    out_path = f'data/{dataset}/results_s2d.json'
    with open(out_path, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Summary table
    import csv
    csv_path = f'results/{dataset}_s2d_summary.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys(), delimiter=';')
        w.writeheader()
        w.writerows(results)
    print(f"  Saved: {csv_path}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY — {dataset}")
    print(f"{'='*80}")
    methods = sorted(set(r['method'] for r in results))
    sps = sorted(set(r['sparsity'] for r in results))

    print(f"\n{'Method':25s}", end='')
    for sp in sps:
        print(f" | {sp:.0%}:F1  rec%  ×comp", end='')
    print()
    print('-' * (25 + len(sps) * 22))

    for m in methods:
        print(f"{m:25s}", end='')
        for sp in sps:
            mrs = [r for r in results if r['method'] == m and r['sparsity'] == sp]
            if mrs:
                avg_f1 = np.mean([r['final_f1'] for r in mrs])
                avg_rec = np.mean([r['f1_recovery'] for r in mrs]) * 100
                avg_comp = np.mean([r['compression'] for r in mrs])
                print(f" | {avg_f1:.3f} {avg_rec:5.1f} {avg_comp:6.1f}", end='')
            else:
                print(f" |   —     —      —  ", end='')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ex09: Sparse→Dense')
    parser.add_argument('-d', '--dataset', default='moons',
                        choices=['moons', 'circles', 'spirals', 'blobs'])
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    config = CONFIG.copy()
    config['dataset'] = args.dataset
    if args.quick:
        config['seeds'] = [42]
        config['sparsities'] = [0.50, 0.90]
        config['epochs_pretrain'] = 30
        config['finetune_epochs'] = 5

    results = run_experiment(config)
    save_results(results, args.dataset)
    print("\nDone!")
