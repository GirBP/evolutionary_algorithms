#!/usr/bin/env python3
"""
Ex09 Practical Demo: GFCS + SDR on FashionMNIST
=================================================
Demonstrates sparse→dense conversion on a REAL image classification task.

Pipeline:
  1. Train large MLP on FashionMNIST (784→256→128→64→10, ~235K params)
  2. Unstructured magnitude pruning at 80%, 85%, 90% sparsity
  3. Apply 4 conversion methods: GFCS, SDR-Mag, SDR-Evo, SDR-GFCS
  4. Measure: accuracy (F1), real inference time, model size, compression

All timings normalized to RCU (Relative Compute Units) via T_etalon.

RCU metrics:
  - RCU_train: time to train the teacher network
  - RCU_prune: time for pruning + conversion
  - RCU_infer: time for single-batch inference (normalized)
"""
import sys, os, time, json, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import measure_etalon
from ex09_lib.core import CompactMLP, set_seed
from ex09_lib.gfcs import gfcs_convert
from ex09_lib.sdr import sdr_convert


# ═══════════════════════════════════════════
#  FashionMNIST Data Loading
# ═══════════════════════════════════════════

def get_fashionmnist_loaders(batch_size=128, subset_ratio=1.0, seed=42):
    """
    Load FashionMNIST, flatten to 784-dim vectors.
    Returns train_dl, val_dl, test_dl, n_classes=10.
    """
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_full = datasets.FashionMNIST(
        root='data/fashionmnist', train=True, download=True, transform=transform
    )
    test_ds = datasets.FashionMNIST(
        root='data/fashionmnist', train=False, download=True, transform=transform
    )
    
    # Flatten images
    def to_flat_tensors(dataset):
        X = dataset.data.float().view(-1, 784) / 255.0
        # Normalize
        X = (X - 0.2860) / 0.3530
        y = dataset.targets
        return X, y
    
    X_train_full, y_train_full = to_flat_tensors(train_full)
    X_test, y_test = to_flat_tensors(test_ds)
    
    # Subset if needed
    if subset_ratio < 1.0:
        np.random.seed(seed)
        n = int(len(X_train_full) * subset_ratio)
        idx = np.random.choice(len(X_train_full), n, replace=False)
        X_train_full = X_train_full[idx]
        y_train_full = y_train_full[idx]
    
    # Split train into train + val (90/10)
    n_train = int(len(X_train_full) * 0.9)
    X_train, X_val = X_train_full[:n_train], X_train_full[n_train:]
    y_train, y_val = y_train_full[:n_train], y_train_full[n_train:]
    
    train_dl = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    val_dl = DataLoader(
        TensorDataset(X_val, y_val), batch_size=batch_size
    )
    test_dl = DataLoader(
        TensorDataset(X_test, y_test), batch_size=batch_size
    )
    
    print(f"  FashionMNIST: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return train_dl, val_dl, test_dl, 10


# ═══════════════════════════════════════════
#  Large MLP (uses SimpleMLP interface)
# ═══════════════════════════════════════════

class LargeMLP(nn.Module):
    """Large MLP for FashionMNIST: 784→256→128→64→10.
    
    Uses same interface as SimpleMLP (fc1, fc2, fc3, fc4) so GFCS/SDR work.
    """
    def __init__(self, input_dim=784, hidden=256, n_classes=10):
        super().__init__()
        # 4 layers: 784→256→128→64→10
        self.fc1 = nn.Linear(input_dim, hidden)       # 784×256 = 200,960
        self.fc2 = nn.Linear(hidden, hidden // 2)      # 256×128 = 32,896
        self.fc3 = nn.Linear(hidden // 2, hidden // 4) # 128×64  = 8,256
        self.fc4 = nn.Linear(hidden // 4, n_classes)   # 64×10   = 650
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden = hidden  # used by SDR to derive topology

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def count_nonzero(self):
        return sum((p.data != 0).sum().item() for p in self.parameters())


# ═══════════════════════════════════════════
#  Training & Evaluation
# ═══════════════════════════════════════════

def train_model(model, train_dl, epochs=15, lr=1e-3, verbose=True):
    """Train model with Adam, return training time."""
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    for ep in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for X, y in train_dl:
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"    epoch {ep+1}/{epochs}: loss={total_loss/len(train_dl):.4f}  "
                  f"acc={correct/total:.4f}")
    
    return model


def evaluate(model, test_dl):
    """Evaluate model, return (loss, f1_macro, accuracy)."""
    from sklearn.metrics import f1_score
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for X, y in test_dl:
            logits = model(X)
            total_loss += F.cross_entropy(logits, y, reduction='sum').item()
            all_preds.extend(logits.argmax(1).tolist())
            all_labels.extend(y.tolist())
    
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(all_labels), f1, acc


def measure_inference_rcu(model, test_dl, T_etalon, n_repeats=50):
    """Measure inference time in RCU."""
    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            for X, _ in test_dl:
                _ = model(X)
    
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        with torch.no_grad():
            for X, _ in test_dl:
                _ = model(X)
        times.append(time.perf_counter() - t0)
    
    median_s = np.median(times)
    return median_s / T_etalon  # in RCU


# ═══════════════════════════════════════════
#  Magnitude Pruning
# ═══════════════════════════════════════════

def magnitude_prune(model, sparsity):
    """Apply global unstructured magnitude pruning."""
    import copy
    pruned = copy.deepcopy(model)
    
    # Collect all weights
    all_weights = torch.cat([p.data.abs().flatten() for p in pruned.parameters()])
    threshold = torch.quantile(all_weights, sparsity)
    
    # Apply mask
    with torch.no_grad():
        for p in pruned.parameters():
            mask = p.data.abs() >= threshold
            p.data *= mask.float()
    
    return pruned


# ═══════════════════════════════════════════
#  Main Experiment
# ═══════════════════════════════════════════

def run_practical_demo():
    """Full practical demonstration on FashionMNIST."""
    
    seeds = [42, 123]
    sparsities = [0.80, 0.85, 0.90]
    train_epochs = 20
    kd_epochs = 20
    ft_epochs = 10
    
    print("=" * 110)
    print("  PRACTICAL DEMO: Sparse→Dense on FashionMNIST (784→256→128→64→10)")
    print("=" * 110)
    
    # ─── Measure RCU etalon ───
    print("\n  Measuring T_etalon...")
    T_etalon = measure_etalon(n_runs=10, seed=42)
    print(f"  T_etalon = {T_etalon:.6f} s")
    
    all_results = []
    
    for seed in seeds:
        set_seed(seed)
        print(f"\n{'='*110}")
        print(f"  SEED = {seed}")
        print(f"{'='*110}")
        
        # ─── Load data ───
        train_dl, val_dl, test_dl, n_classes = get_fashionmnist_loaders(
            batch_size=128, subset_ratio=1.0, seed=seed
        )
        
        # ─── Train teacher ───
        print(f"\n  Training teacher MLP (784→256→128→64→10)...")
        teacher = LargeMLP(input_dim=784, hidden=256, n_classes=10)
        teacher_params = teacher.count_params()
        
        t0 = time.perf_counter()
        teacher = train_model(teacher, train_dl, epochs=train_epochs, lr=1e-3)
        rcu_train = (time.perf_counter() - t0) / T_etalon
        
        _, teacher_f1, teacher_acc = evaluate(teacher, test_dl)
        rcu_infer_teacher = measure_inference_rcu(teacher, test_dl, T_etalon)
        
        print(f"  Teacher: F1={teacher_f1:.4f}  acc={teacher_acc:.4f}  "
              f"params={teacher_params:,}  RCU_train={rcu_train:.1f}  "
              f"RCU_infer={rcu_infer_teacher:.4f}")
        
        for sparsity in sparsities:
            print(f"\n  {'─'*90}")
            print(f"  Sparsity = {sparsity:.0%}")
            print(f"  {'─'*90}")
            
            # ─── Prune ───
            t0 = time.perf_counter()
            sparse_model = magnitude_prune(teacher, sparsity)
            rcu_prune = (time.perf_counter() - t0) / T_etalon
            
            _, sparse_f1, sparse_acc = evaluate(sparse_model, test_dl)
            sparse_nnz = sparse_model.count_nonzero()
            rcu_infer_sparse = measure_inference_rcu(sparse_model, test_dl, T_etalon)
            
            print(f"  Sparse:    F1={sparse_f1:.4f}  acc={sparse_acc:.4f}  "
                  f"nnz={sparse_nnz:,}  RCU_prune={rcu_prune:.4f}  "
                  f"RCU_infer={rcu_infer_sparse:.4f}")
            
            base_row = {
                'seed': seed, 'sparsity': sparsity,
                'teacher_f1': round(teacher_f1, 4),
                'teacher_acc': round(teacher_acc, 4),
                'teacher_params': teacher_params,
                'rcu_train': round(rcu_train, 2),
                'sparse_f1': round(sparse_f1, 4),
                'sparse_acc': round(sparse_acc, 4),
                'sparse_nnz': sparse_nnz,
                'rcu_prune': round(rcu_prune, 4),
                'rcu_infer_teacher': round(rcu_infer_teacher, 4),
                'rcu_infer_sparse': round(rcu_infer_sparse, 4),
            }
            
            # ─── Apply conversion methods ───
            methods = [
                ('GFCS', 'gfcs_merge'),
                ('SDR-Mag', 'magnitude'),
                ('SDR-Evo', 'evo'),
                ('SDR-GFCS', 'gfcs'),
            ]
            
            for method_name, method_key in methods:
                set_seed(seed)
                t0 = time.perf_counter()
                
                if method_key == 'gfcs_merge':
                    # Pure GFCS: algebraic merge + finetune
                    compact, info = gfcs_convert(sparse_model, n_classes=n_classes)
                    from ex09_lib.core import train_model as core_train
                    core_train(compact, train_dl, epochs=ft_epochs, lr=0.01)
                    compact_hiddens = info['merged_hiddens']
                else:
                    # SDR variants: topology + KD
                    compact, info = sdr_convert(
                        sparse_model, teacher, train_dl, n_classes=n_classes,
                        method=method_key, kd_epochs=kd_epochs, ft_epochs=ft_epochs
                    )
                    compact_hiddens = info['compact_hiddens']
                
                rcu_convert = (time.perf_counter() - t0) / T_etalon
                
                _, compact_f1, compact_acc = evaluate(compact, test_dl)
                compact_params = compact.count_params()
                rcu_infer_compact = measure_inference_rcu(compact, test_dl, T_etalon)
                
                compression = teacher_params / max(compact_params, 1)
                delta_f1 = compact_f1 - sparse_f1
                speedup = rcu_infer_sparse / rcu_infer_compact if rcu_infer_compact > 0 else 0
                
                flag = '✅' if delta_f1 >= -0.02 else '❌'
                print(f"  {method_name:10s}: F1={compact_f1:.4f} (Δ={delta_f1:+.4f})  "
                      f"acc={compact_acc:.4f}  params={compact_params:>6,}  "
                      f"comp={compression:4.1f}×  RCU_conv={rcu_convert:6.1f}  "
                      f"RCU_inf={rcu_infer_compact:.4f}  speed={speedup:.2f}×  "
                      f"h={compact_hiddens}  {flag}")
                
                row = {**base_row}
                row.update({
                    'method': method_name,
                    'compact_f1': round(compact_f1, 4),
                    'compact_acc': round(compact_acc, 4),
                    'compact_params': compact_params,
                    'compact_hiddens': str(compact_hiddens),
                    'compression': round(compression, 1),
                    'delta_f1': round(delta_f1, 4),
                    'rcu_convert': round(rcu_convert, 2),
                    'rcu_infer_compact': round(rcu_infer_compact, 4),
                    'speedup_rcu': round(speedup, 2),
                })
                all_results.append(row)
    
    # ═══════════════════════════════════════
    #  Summary Tables
    # ═══════════════════════════════════════
    print(f"\n\n{'='*110}")
    print("  SUMMARY TABLE: F1 by (Sparsity × Method), averaged across seeds")
    print(f"{'='*110}")
    
    from collections import defaultdict
    summary = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        key = (r['sparsity'], r['method'])
        summary[key]['f1'].append(r['compact_f1'])
        summary[key]['params'].append(r['compact_params'])
        summary[key]['comp'].append(r['compression'])
        summary[key]['rcu_conv'].append(r['rcu_convert'])
        summary[key]['rcu_inf'].append(r['rcu_infer_compact'])
        summary[key]['speed'].append(r['speedup_rcu'])
    
    print(f"\n  {'Sp':>4s}  {'Method':12s}  {'F1':>7s}  {'Params':>8s}  {'Comp×':>6s}  "
          f"{'RCU_conv':>8s}  {'RCU_inf':>8s}  {'Speedup':>7s}")
    print("  " + "─" * 80)
    
    for sp in sparsities:
        for method in ['GFCS', 'SDR-Mag', 'SDR-Evo', 'SDR-GFCS']:
            key = (sp, method)
            if key not in summary:
                continue
            s = summary[key]
            print(f"  {sp:4.0%}  {method:12s}  {np.mean(s['f1']):7.4f}  "
                  f"{int(np.mean(s['params'])):>8,}  {np.mean(s['comp']):5.1f}×  "
                  f"{np.mean(s['rcu_conv']):8.1f}  {np.mean(s['rcu_inf']):8.4f}  "
                  f"{np.mean(s['speed']):6.2f}×")
        print()
    
    # ─── RCU Budget table ───
    print(f"\n{'='*110}")
    print("  RCU BUDGET TABLE: Total cost of each approach (train + prune + convert + infer)")
    print(f"{'='*110}")
    
    print(f"\n  {'Sp':>4s}  {'Method':12s}  {'RCU_train':>10s}  {'RCU_prune':>10s}  "
          f"{'RCU_conv':>10s}  {'RCU_infer':>10s}  {'TOTAL':>10s}")
    print("  " + "─" * 80)
    
    for sp in sparsities:
        for method in ['GFCS', 'SDR-Mag', 'SDR-Evo', 'SDR-GFCS']:
            rows = [r for r in all_results if r['sparsity'] == sp and r['method'] == method]
            if not rows:
                continue
            r_train = np.mean([r['rcu_train'] for r in rows])
            r_prune = np.mean([r['rcu_prune'] for r in rows])
            r_conv = np.mean([r['rcu_convert'] for r in rows])
            r_inf = np.mean([r['rcu_infer_compact'] for r in rows])
            total = r_train + r_prune + r_conv
            print(f"  {sp:4.0%}  {method:12s}  {r_train:10.1f}  {r_prune:10.4f}  "
                  f"{r_conv:10.1f}  {r_inf:10.4f}  {total:10.1f}")
        print()
    
    # ─── Save results ───
    os.makedirs('results', exist_ok=True)
    out_path = 'results/practical_fashionmnist.json'
    with open(out_path, 'w') as f:
        json.dump({
            'experiment': 'Ex09_Practical_FashionMNIST',
            'model': 'LargeMLP 784→256→128→64→10',
            'dataset': 'FashionMNIST (60K train, 10K test)',
            'T_etalon': T_etalon,
            'results': all_results,
        }, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    
    out_csv = 'results/practical_fashionmnist.csv'
    if all_results:
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_results[0].keys(), delimiter=';')
            w.writeheader()
            w.writerows(all_results)
        print(f"  Saved: {out_csv}")
    
    return all_results


if __name__ == '__main__':
    run_practical_demo()
