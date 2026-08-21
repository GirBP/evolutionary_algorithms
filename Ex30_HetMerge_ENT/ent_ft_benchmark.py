#!/usr/bin/env python3
"""ENT-FT: ENT with post-merge FC fine-tuning.

Extension of ENT that adds a calibration step:
1. ENT topology search (channel masks via EA)  
2. Freeze conv backbone (preserves parent features)
3. Fine-tune FC on validation set (LogReg or SGD, few epochs)

This addresses ENT's key weakness: FC routing uses parent weights
that were trained on original features, not merged features.
"""

import time, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from pathlib import Path

# Самодостатні визначення (SmallCNN та покласова оцінка —
# якого немає в публічному репозиторії) — архітектура та функція оцінки
# ідентичні оригіналу, обчислення не змінені.
class SmallCNN(nn.Module):
    """Мінімальна CNN для MNIST/FashionMNIST. ~1.3K параметрів для n_out=2."""
    def __init__(self, n_out=2):
        super().__init__()
        self.n_out = n_out
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AvgPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, n_out)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.gap(x).flatten(1)
        return self.fc(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def eval_per_class_accuracy(model, xs, ys, all_classes):
    """Точність по класах. ys — оригінальні мітки класів."""
    model.eval()
    with torch.no_grad():
        logits = model(xs)
        preds = logits.argmax(1)

    accs = {}
    for c in all_classes:
        mask = ys == c
        if mask.sum() == 0:
            accs[c] = 0.0
        else:
            accs[c] = (preds[mask] == c).float().mean().item()
    return accs


# Шляхи — відносні від кореня зони (Ex30_HetMerge_ENT/).
RESULTS_DIR = Path(__file__).resolve().parent / "results_full_benchmark"
PARENTS_DIR = RESULTS_DIR / "parents"
SEEDS = [42, 123]


def load_scenario(sc_key, seed):
    d = PARENTS_DIR / sc_key / f"s{seed}"
    sdA = torch.load(d / "parentA.pt", map_location='cpu', weights_only=True)
    sdB = torch.load(d / "parentB.pt", map_location='cpu', weights_only=True)
    data = torch.load(d / "valset.pt", weights_only=True)
    config = torch.load(d / "config.pt", weights_only=True)
    return sdA, sdB, data['val_xs'], data['val_ys'], data['test_xs'], data['test_ys'], config


# ══════════════════════════════════════════
# ENT-FT: ENT + FC calibration
# ══════════════════════════════════════════

def ent_ft_merge(weights_A, weights_B, val_xs, val_ys, config,
                 ft_epochs=30, ft_lr=0.01, use_logreg=True):
    """ENT with post-merge FC fine-tuning.
    
    Phase 1: EA finds optimal channel masks (same as ENT)
    Phase 2: Freeze conv backbone, fine-tune FC on val set
    """
    ALL_CLS = config['ALL_CLS']
    n_out = config['n_out']
    CLS_A, CLS_B = config['CLS_A'], config['CLS_B']
    remapA, remapB = config['remapA'], config['remapB']
    
    n_ch1 = weights_A['conv1.weight'].shape[0]
    n_ch2 = weights_A['conv2.weight'].shape[0]
    mapped_ys = torch.tensor([ALL_CLS.index(y.item()) for y in val_ys])

    def build_merged_model(ch):
        """Build merged SmallCNN from chromosome. Returns model (not state_dict)."""
        mA1, mB1 = ch['mask1A'], ch['mask1B']
        mA2, mB2 = ch['mask2A'], ch['mask2B']
        
        iA1, iB1 = np.where(mA1)[0], np.where(mB1)[0]
        iA2, iB2 = np.where(mA2)[0], np.where(mB2)[0]
        nA1, nB1 = len(iA1), len(iB1)
        nA2, nB2 = len(iA2), len(iB2)
        if nA1+nB1 < 1 or nA2+nB2 < 1:
            return None, {}
        
        sd = OrderedDict()
        # Conv1: concat
        sd['conv1.weight'] = torch.cat([weights_A['conv1.weight'][iA1],
                                         weights_B['conv1.weight'][iB1]], dim=0)
        sd['conv1.bias'] = torch.cat([weights_A['conv1.bias'][iA1],
                                       weights_B['conv1.bias'][iB1]])
        # Conv2: block-diagonal
        n_in2, n_out2 = nA1+nB1, nA2+nB2
        w2 = torch.zeros(n_out2, n_in2, 3, 3)
        w2[:nA2, :nA1] = weights_A['conv2.weight'][np.ix_(iA2, iA1)]
        w2[nA2:, nA1:] = weights_B['conv2.weight'][np.ix_(iB2, iB1)]
        sd['conv2.weight'] = w2
        sd['conv2.bias'] = torch.cat([weights_A['conv2.bias'][iA2],
                                       weights_B['conv2.bias'][iB2]])
        
        # FC: initialize with ENT routing (will be overwritten in Phase 2)
        route = ch.get('route', np.zeros(n_out))
        fc_w = torch.zeros(n_out, n_out2)
        fc_b = torch.zeros(n_out)
        for c in ALL_CLS:
            idx = ALL_CLS.index(c)
            alpha = 1.0 / (1.0 + np.exp(-route[idx]))
            if c in CLS_A and c in CLS_B:
                rA, rB = remapA[c], remapB[c]
                fc_w[idx, :nA2] = alpha * weights_A['fc.weight'][rA][iA2]
                fc_w[idx, nA2:] = (1-alpha) * weights_B['fc.weight'][rB][iB2]
                fc_b[idx] = alpha * weights_A['fc.bias'][rA] + (1-alpha) * weights_B['fc.bias'][rB]
            elif c in CLS_A:
                rA = remapA[c]
                fc_w[idx, :nA2] = weights_A['fc.weight'][rA][iA2]
                fc_b[idx] = weights_A['fc.bias'][rA]
            elif c in CLS_B:
                rB = remapB[c]
                fc_w[idx, nA2:] = weights_B['fc.weight'][rB][iB2]
                fc_b[idx] = weights_B['fc.bias'][rB]
        sd['fc.weight'] = fc_w
        sd['fc.bias'] = fc_b
        
        model = SmallCNN(n_out=n_out)
        model.conv1 = nn.Conv2d(1, nA1+nB1, 3, padding=1)
        model.conv2 = nn.Conv2d(nA1+nB1, nA2+nB2, 3, padding=1)
        model.fc = nn.Linear(nA2+nB2, n_out)
        model.load_state_dict(sd)
        model.eval()
        
        info = {'nA1': nA1, 'nB1': nB1, 'nA2': nA2, 'nB2': nB2,
                'iA2': iA2, 'iB2': iB2}
        return model, info

    # ── Phase 1: EA topology search (simplified, larger budget) ──
    def fitness(ch):
        model, _ = build_merged_model(ch)
        if model is None: return -1.0
        with torch.no_grad():
            preds = model(val_xs).argmax(1)
        acc = (preds == mapped_ys).float().mean().item()
        per_cls = []
        for c_idx in range(n_out):
            mask = mapped_ys == c_idx
            if mask.sum() > 0:
                per_cls.append((preds[mask] == c_idx).float().mean().item())
        min_c = min(per_cls) if per_cls else 0.0
        total = 2*(n_ch1+n_ch2)
        used = ch['mask1A'].sum()+ch['mask1B'].sum()+ch['mask2A'].sum()+ch['mask2B'].sum()
        return 0.4*acc + 0.4*min_c + 0.1*np.mean(per_cls) + 0.1*(1-used/total)

    POP, GENS = 20, 30
    pop = []
    for _ in range(POP):
        ch = {'mask1A': np.random.random(n_ch1)>0.3, 'mask1B': np.random.random(n_ch1)>0.3,
              'mask2A': np.random.random(n_ch2)>0.3, 'mask2B': np.random.random(n_ch2)>0.3,
              'route': np.random.randn(n_out)*1.5}
        for mk in ['mask1A','mask1B','mask2A','mask2B']:
            if ch[mk].sum()==0: ch[mk][0]=True
        pop.append(ch)
    
    # Seed configurations
    pop[0] = {'mask1A': np.ones(n_ch1,dtype=bool), 'mask1B': np.ones(n_ch1,dtype=bool),
              'mask2A': np.ones(n_ch2,dtype=bool), 'mask2B': np.ones(n_ch2,dtype=bool),
              'route': np.zeros(n_out)}
    if CLS_A != CLS_B:
        route_s = np.zeros(n_out)
        for c in CLS_A: route_s[ALL_CLS.index(c)] = 3.0
        for c in CLS_B: route_s[ALL_CLS.index(c)] = -3.0
        pop[1] = {'mask1A': np.ones(n_ch1,dtype=bool), 'mask1B': np.ones(n_ch1,dtype=bool),
                  'mask2A': np.ones(n_ch2,dtype=bool), 'mask2B': np.ones(n_ch2,dtype=bool),
                  'route': route_s}

    best_fit, best_ch = -1, None
    for gen in range(GENS):
        fits = [fitness(ch) for ch in pop]
        gi = np.argmax(fits)
        if fits[gi] > best_fit:
            best_fit = fits[gi]
            best_ch = {k: v.copy() for k,v in pop[gi].items()}
        new_pop = [{k: v.copy() for k,v in best_ch.items()}]
        while len(new_pop) < POP:
            ti = np.random.choice(len(pop), 3, replace=False)
            parent = pop[ti[np.argmax([fits[i] for i in ti])]]
            child = {k: v.copy() for k,v in parent.items()}
            child['route'] = child['route'] + np.random.randn(n_out)*0.3
            pf = max(0.02, 0.06 - gen*0.001)
            for mk in ['mask1A','mask1B','mask2A','mask2B']:
                flip = np.random.random(len(child[mk])) < pf
                child[mk][flip] = ~child[mk][flip]
                if child[mk].sum()==0: child[mk][np.random.randint(len(child[mk]))]=True
            new_pop.append(child)
        pop = new_pop
    
    print(f"    Phase 1 (EA): best fitness = {best_fit:.4f}")

    # ── Phase 2: FC calibration ──
    model, info = build_merged_model(best_ch)
    if model is None:
        # fallback
        merged = OrderedDict()
        for k in weights_A: merged[k] = 0.5 * weights_A[k] + 0.5 * weights_B[k]
        return merged
    
    # Freeze conv backbone
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    
    # Extract features from frozen backbone
    model.eval()
    with torch.no_grad():
        x = model.pool(F.relu(model.conv1(val_xs)))
        x = model.pool(F.relu(model.conv2(x)))
        features = model.gap(x).flatten(1)  # (N, n_features)
    
    n_features = features.shape[1]
    
    if ft_epochs == 0 and not use_logreg:
        # No fine-tuning — return pure ENT model
        print(f"    Phase 2: SKIPPED (pure ENT)")
        return model
    
    if use_logreg:
        # Approach A: sklearn LogisticRegression (fast, regularized)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features.numpy())
        
        # Per-class balanced weighting
        class_counts = {}
        for y in mapped_ys.numpy():
            class_counts[y] = class_counts.get(y, 0) + 1
        sample_weights = np.array([1.0 / class_counts[y] for y in mapped_ys.numpy()])
        sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
        
        clf = LogisticRegression(
            max_iter=500, C=1.0,
            solver='lbfgs', class_weight='balanced'
        )
        clf.fit(X_scaled, mapped_ys.numpy(), sample_weight=sample_weights)
        
        # Replace FC with LogReg weights
        # Need to account for scaler: w_new = w_lr / scale, b_new = b_lr - (w_lr * mean / scale)
        scale = torch.tensor(scaler.scale_, dtype=torch.float32)
        mean = torch.tensor(scaler.mean_, dtype=torch.float32)
        
        w_lr = torch.tensor(clf.coef_, dtype=torch.float32)  # (n_out, n_features)
        b_lr = torch.tensor(clf.intercept_, dtype=torch.float32)  # (n_out,)
        
        # Fuse scaler into weights: y = w_lr @ ((x - mean) / scale) + b_lr
        #                           = (w_lr / scale) @ x + (b_lr - w_lr @ mean / scale)
        w_fused = w_lr / scale.unsqueeze(0)
        b_fused = b_lr - (w_lr * mean.unsqueeze(0) / scale.unsqueeze(0)).sum(1)
        
        with torch.no_grad():
            model.fc.weight.copy_(w_fused)
            model.fc.bias.copy_(b_fused)
        
        print(f"    Phase 2 (LogReg): calibrated FC on {len(val_xs)} samples")
    
    else:
        # Approach B: SGD fine-tuning with per-class balanced loss
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        
        optimizer = torch.optim.Adam([model.fc.weight, model.fc.bias], lr=ft_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)
        
        # Per-class weights for balanced loss
        class_counts = torch.bincount(mapped_ys, minlength=n_out).float()
        class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
        class_weights = class_weights / class_weights.sum() * n_out
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        for epoch in range(ft_epochs):
            logits = model.fc(features)
            loss = criterion(logits, mapped_ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        print(f"    Phase 2 (SGD): {ft_epochs} epochs, final loss = {loss.item():.4f}")
    
    model.eval()
    return model


# ══════════════════════════════════════════
# BENCHMARK
# ══════════════════════════════════════════

def evaluate_model(model, test_xs, test_ys, config):
    ALL_CLS = config['ALL_CLS']
    model.eval()
    accs = eval_per_class_accuracy(model, test_xs, test_ys, ALL_CLS)
    vals = [accs[c] for c in ALL_CLS]
    mn, mx = min(vals), max(vals)
    return {
        'accuracy': round(float(np.mean(vals)), 4),
        'balance': round(float(mn/mx) if mx>0 else 0, 4),
        'min_class': round(float(mn), 4),
        'n_ok': sum(1 for a in vals if a > 0.1),
        'n_total': len(ALL_CLS),
        'n_params': sum(p.numel() for p in model.parameters()),
        'per_class': {c: round(float(accs[c]), 4) for c in ALL_CLS},
    }


def run_comparison():
    scenarios = ['homo_mnist_10', 'homo_fmnist_10', 'homo_mnist_4',
                 'same_mnist_10', 'same_fmnist_10']
    
    print("=" * 80)
    print("  ENT vs ENT-FT (LogReg) vs ENT-FT (SGD) — Full Comparison")
    print("=" * 80)
    
    all_results = []
    
    for sc_key in scenarios:
        print(f"\n{'='*60}")
        print(f"  SCENARIO: {sc_key}")
        print(f"{'='*60}")
        
        # Load parent accuracies for retention calc
        d = PARENTS_DIR / sc_key / f"s42"
        config_check = torch.load(d / "config.pt", weights_only=True)
        parent_accs = config_check.get('parent_accs', {})
        
        for seed in SEEDS:
            sdA, sdB, val_xs, val_ys, test_xs, test_ys, config = load_scenario(sc_key, seed)
            ALL_CLS = config['ALL_CLS']
            
            methods = [
                ('ENT', {'ft_epochs': 0, 'use_logreg': False}),
                ('ENT-FT-LogReg', {'use_logreg': True}),
                ('ENT-FT-SGD30', {'use_logreg': False, 'ft_epochs': 30, 'ft_lr': 0.01}),
            ]
            
            for method_name, kwargs in methods:
                print(f"\n  {method_name} | {sc_key} s={seed}")
                t0 = time.time()
                
                model = ent_ft_merge(sdA, sdB, val_xs, val_ys, config, **kwargs)
                
                elapsed = time.time() - t0
                
                if isinstance(model, dict) or isinstance(model, OrderedDict):
                    print(f"    FALLBACK (returned dict)")
                    continue
                
                result = evaluate_model(model, test_xs, test_ys, config)
                result['method'] = method_name
                result['scenario'] = sc_key
                result['seed'] = seed
                result['time_s'] = round(elapsed, 1)
                all_results.append(result)
                
                # Print per-class with retention
                print(f"    acc={result['accuracy']:.3f} bal={result['balance']:.3f} "
                      f"ok={result['n_ok']}/{result['n_total']} "
                      f"min={result['min_class']:.3f} ({elapsed:.1f}s)")
                print(f"    Per-class: ", end="")
                for c in ALL_CLS:
                    a = result['per_class'][c]
                    pa = parent_accs.get(c, 0)
                    ret = f"({a/pa*100:.0f}%)" if pa > 0 else ""
                    print(f"c{c}={a:.3f}{ret} ", end="")
                print()
    
    # ── Summary tables ──
    import pandas as pd
    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_DIR / "ent_ft_results.tsv", sep='\t', index=False)
    
    print("\n\n" + "=" * 90)
    print("  SUMMARY — Average across seeds")
    print("=" * 90)
    
    for sc_key in scenarios:
        sc_df = df[df['scenario'] == sc_key]
        if len(sc_df) == 0: continue
        print(f"\n  ── {sc_key} ──")
        print(f"  {'Method':<20} {'Acc':>6} {'Balance':>8} {'Min':>7} {'OK':>5} {'Params':>7} {'Time':>6}")
        print(f"  {'-'*20} {'-'*6} {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*6}")
        
        agg = sc_df.groupby('method').agg({
            'accuracy': 'mean', 'balance': 'mean', 'min_class': 'mean',
            'n_ok': 'mean', 'n_params': 'mean', 'time_s': 'mean',
        }).sort_values('accuracy', ascending=False)
        
        for method, row in agg.iterrows():
            print(f"  {method:<20} {row['accuracy']:>6.3f} {row['balance']:>8.3f} "
                  f"{row['min_class']:>7.3f} {row['n_ok']:>4.0f}/{sc_df['n_total'].iloc[0]} "
                  f"{int(row['n_params']):>7d} {row['time_s']:>5.1f}s")

    print(f"\n  Results saved to: {RESULTS_DIR / 'ent_ft_results.tsv'}")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    run_comparison()
