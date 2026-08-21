#!/usr/bin/env python3
"""
Ex09 Pre-protocol: Train teachers, prune, save masks.
Knuth Protocol §1 — prepare fixed input data BEFORE research begins.

For each (dataset × seed):
  1. Train SimpleMLP teacher → save teacher_seed{s}.pt
  2. Prune at target sparsity (90%) → check F1 ≥ 0.80
     If F1 < 0.80 → reduce sparsity by 5% steps until F1 ≥ 0.80
  3. Save sparse model + mask
  4. Log baseline metrics
"""
import sys, os, json, torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ex09_lib.core import (
    set_seed, get_dataloaders, SimpleMLP,
    train_model, evaluate, prune_magnitude_global, get_sparsity,
)
from ex09_lib.config_experiment import CONFIG


def prepare_dataset(dataset, seed, target_sp, min_f1):
    """Train teacher, find valid sparsity, save everything."""
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}, Seed: {seed}")
    print(f"{'='*60}")

    set_seed(seed)
    base_dir = f"data/{dataset}"
    os.makedirs(base_dir, exist_ok=True)

    # 1. Train teacher
    train_dl, val_dl, test_dl, n_classes = get_dataloaders(seed, dataset, CONFIG['batch_size'])
    teacher_path = f"{base_dir}/teacher_seed{seed}.pt"

    if os.path.exists(teacher_path):
        teacher = SimpleMLP(n_classes=n_classes)
        teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
        print(f"  Loaded teacher: {teacher_path}")
    else:
        teacher = SimpleMLP(n_classes=n_classes)
        train_model(teacher, train_dl, epochs=CONFIG['epochs_pretrain'])
        torch.save(teacher.state_dict(), teacher_path)
        print(f"  Trained teacher → {teacher_path}")

    _, teacher_f1, teacher_acc = evaluate(teacher, test_dl)
    print(f"  Teacher: F1={teacher_f1:.4f}, acc={teacher_acc:.4f}, params={teacher.count_params():,}")

    # 2. Find valid sparsity
    sparsity = target_sp
    while sparsity >= 0.30:
        set_seed(seed)
        sparse = SimpleMLP(n_classes=n_classes)
        sparse.load_state_dict({k: v.clone() for k, v in teacher.state_dict().items()})
        prune_magnitude_global(sparse, sparsity)

        actual_sp = get_sparsity(sparse)
        _, sparse_f1, sparse_acc = evaluate(sparse, test_dl)

        print(f"  Sparsity {sparsity:.0%}: F1={sparse_f1:.4f}, actual={actual_sp:.1%}")

        if sparse_f1 >= min_f1:
            break
        else:
            print(f"    F1 < {min_f1} → reducing sparsity")
            sparsity -= 0.05

    if sparse_f1 < min_f1:
        print(f"   Could not find valid sparsity ≥ {min_f1} F1. Using {sparsity:.0%}")

    # 3. Save sparse model + mask
    sparse_path = f"{base_dir}/sparse_seed{seed}_sp{sparsity:.2f}.pt"
    torch.save(sparse.state_dict(), sparse_path)

    # Extract and save mask
    mask = {}
    for name, param in sparse.named_parameters():
        if param.dim() >= 2:
            mask[name] = (param.data != 0).float()
    mask_path = f"{base_dir}/mask_seed{seed}_sp{sparsity:.2f}.pt"
    torch.save(mask, mask_path)

    print(f"  Saved: {sparse_path}")
    print(f"  Saved: {mask_path}")

    # 4. Return metrics
    return {
        'dataset': dataset,
        'seed': seed,
        'teacher_f1': teacher_f1,
        'teacher_acc': teacher_acc,
        'teacher_params': teacher.count_params(),
        'target_sparsity': target_sp,
        'actual_sparsity': sparsity,
        'actual_sparsity_measured': actual_sp,
        'sparse_f1': sparse_f1,
        'sparse_acc': sparse_acc,
        'sparse_nonzero': sparse.count_nonzero(),
        'n_classes': n_classes,
        'teacher_path': teacher_path,
        'sparse_path': sparse_path,
        'mask_path': mask_path,
    }


def main():
    datasets = CONFIG['datasets']
    seeds = CONFIG['seeds']
    target_sp = CONFIG['target_sparsity']
    min_f1 = CONFIG['min_f1_weighted']

    print("Ex09 Pre-protocol: Preparing input data")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {seeds}")
    print(f"Target sparsity: {target_sp:.0%}, min F1: {min_f1}")

    all_results = []
    for dataset in datasets:
        for seed in seeds:
            result = prepare_dataset(dataset, seed, target_sp, min_f1)
            all_results.append(result)

    # Save summary
    os.makedirs('data', exist_ok=True)
    summary_path = 'data/preprotocol_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({'results': all_results}, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("PRE-PROTOCOL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':20s} {'Seed':>5s} {'Teacher F1':>10s} {'Sparsity':>10s} {'Sparse F1':>10s} {'NonZero':>8s} {'':>3s}")
    print('-' * 80)
    for r in all_results:
        ok = '' if r['sparse_f1'] >= min_f1 else ''
        print(f"{r['dataset']:20s} {r['seed']:5d} {r['teacher_f1']:10.4f} "
              f"{r['actual_sparsity']:10.0%} {r['sparse_f1']:10.4f} "
              f"{r['sparse_nonzero']:8,d} {ok:>3s}")

    print(f"\nSaved: {summary_path}")
    print("Pre-protocol complete. Ready to begin research.")


if __name__ == '__main__':
    main()
