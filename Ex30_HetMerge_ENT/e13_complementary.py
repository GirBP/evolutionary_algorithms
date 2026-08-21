#!/usr/bin/env python3
"""
E13 [Hypothesis] — Complementary knowledge merging.
=====================================================
The ULTIMATE test: can PCMA merge models trained on DIFFERENT data?

Setup:
  Model A: trained on digits 0-4 (class-split, NOT data-split)
  Model B: trained on digits 5-9
  Merged model should classify ALL digits 0-9

This tests true knowledge transfer — not just mixing parameters,
but combining complementary classification capabilities.

Also test: overlapping splits
  Model A: digits 0-6
  Model B: digits 3-9
  Overlap: digits 3-6

Fidelity: L0 (MNIST subset, 1 seed)
"""

import numpy as np
import torch
import torch.nn as nn
import time, json, sys

sys.path.insert(0, '/Users/bibo/Desktop/cs_dev/Ex30_HetMerge')

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def load_mnist():
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
    te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
    X_tr = torch.stack([tr[i][0] for i in range(len(tr))])
    y_tr = torch.tensor([tr[i][1] for i in range(len(tr))])
    X_te = torch.stack([te[i][0] for i in range(2000)])
    y_te = torch.tensor([te[i][1] for i in range(2000)])
    return X_tr, y_tr, X_te, y_te


class MLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch)-2: layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers); self.arch = arch
    def forward(self, x): return self.net(x)


def ev(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def per_class_acc(model, X, y, classes=range(10)):
    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1)
    accs = {}
    for c in classes:
        mask = (y == c)
        if mask.sum() > 0:
            accs[c] = (preds[mask] == c).float().mean().item()
        else:
            accs[c] = 0.0
    return accs


def train_on_classes(arch, X, y, classes, epochs=15, lr=0.003):
    """Train model on subset of classes only."""
    mask = torch.zeros(len(y), dtype=torch.bool)
    for c in classes:
        mask |= (y == c)
    X_sub = X[mask]
    y_sub = y[mask]
    
    # Take max 5000 samples
    if len(X_sub) > 5000:
        idx = torch.randperm(len(X_sub))[:5000]
        X_sub, y_sub = X_sub[idx], y_sub[idx]
    
    model = MLP(arch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        l = nn.CrossEntropyLoss()(model(X_sub), y_sub)
        opt.zero_grad(); l.backward(); opt.step()
    return model


# ─── PCMA merge (inline) ────────────────────────────────────────────

def pcma_merge(mA, mB, X_cal, X_val, y_val, maxiter=35, popsize=14):
    import cma
    archA = mA.arch
    n_hidden = len(archA) - 2
    
    actsA, actsB = [], []
    with torch.no_grad():
        hA, hB = X_cal, X_cal
        for ma, mb in zip(mA.net, mB.net):
            hA, hB = ma(hA), mb(hB)
            if isinstance(ma, nn.ReLU):
                actsA.append(hA.numpy().copy())
                actsB.append(hB.numpy().copy())
    
    svd_list, s_inits = [], []
    for i in range(n_hidden):
        dmin = min(archA[i+1], mB.arch[i+1])
        C = actsA[i].T @ actsB[i]
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_init = S[:dmin] / (S[0]+1e-10)
        svd_list.append((U[:,:dmin], Vt[:dmin,:]))
        s_inits.append(s_init)
    
    s0 = np.concatenate(s_inits)
    nd_s = len(s0)
    n_lay = n_hidden + 1
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    
    x0 = np.concatenate([s0, [0.5]*n_lay])
    nd = len(x0)
    
    def build(s_vec, alphas):
        maps = []; off = 0
        for i in range(n_hidden):
            U, Vt = svd_list[i]; d = U.shape[1]
            maps.append(U @ np.diag(s_vec[off:off+d]) @ Vt); off += d
        params = []
        for li in range(n_lay):
            a = np.clip(alphas[li], 0.05, 0.95)
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            if li==0: wm=maps[0]@wb; bm=maps[0]@bb
            elif li<n_hidden: wm=maps[li]@wb@maps[li-1].T; bm=maps[li]@bb
            else: wm=wb@maps[-1].T; bm=bb
            params.append(a*wa+(1-a)*wm); params.append(a*ba+(1-a)*bm)
        merged = MLP(archA)
        with torch.no_grad():
            for p, v in zip(merged.parameters(), params):
                p.copy_(torch.tensor(v, dtype=torch.float32))
        return merged
    
    def fitness(x):
        try:
            m = build(x[:nd_s], x[nd_s:].tolist())
            return -ev(m, X_val, y_val)
        except: return 1.0
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': SEED,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd_s+[0.1]*n_lay, [3]*nd_s+[0.9]*n_lay],
        'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best = np.array(es.result.xbest)
    merged = build(best[:nd_s], best[nd_s:].tolist())
    alphas = best[nd_s:].tolist()
    return merged, alphas


# ─── Main ────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E13 [Hypothesis] Complementary knowledge merging")
    print("=" * 75)
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    
    # Val split from training data (for CMA fitness)
    idx = torch.randperm(len(X_tr), generator=torch.Generator().manual_seed(0))
    X_val = X_tr[idx[50000:55000]]
    y_val = y_tr[idx[50000:55000]]
    X_cal = X_tr[idx[:3000]]
    
    archA = [784, 128, 64, 10]
    archB = [784, 256, 128, 10]
    
    scenarios = [
        ("Disjoint_0-4_vs_5-9", list(range(5)), list(range(5,10))),
        ("Overlap_0-6_vs_3-9", list(range(7)), list(range(3,10))),
        ("Minimal_0-1_vs_8-9", [0,1], [8,9]),
    ]
    
    results = []
    for name, classes_A, classes_B in scenarios:
        print(f"\n{'━' * 75}")
        print(f"  {name}")
        print(f"  A trains on: {classes_A}")
        print(f"  B trains on: {classes_B}")
        print(f"{'━' * 75}")
        
        mA = train_on_classes(archA, X_tr, y_tr, classes_A)
        mB = train_on_classes(archB, X_tr, y_tr, classes_B)
        
        # Per-class performance of parents
        pcA = per_class_acc(mA, X_te, y_te)
        pcB = per_class_acc(mB, X_te, y_te)
        
        allA = ev(mA, X_te, y_te)
        allB = ev(mB, X_te, y_te)
        
        print(f"  A overall: {allA:.3f}  B overall: {allB:.3f}")
        
        # Merge
        merged, alphas = pcma_merge(mA, mB, X_cal, X_val, y_val)
        pcM = per_class_acc(merged, X_te, y_te)
        allM = ev(merged, X_te, y_te)
        
        # Knowledge transfer metrics
        a_only = [c for c in classes_A if c not in classes_B]
        b_only = [c for c in classes_B if c not in classes_A]
        overlap = [c for c in classes_A if c in classes_B]
        all_classes = sorted(set(classes_A + classes_B))
        
        # Can merged model classify A-only classes?
        a_only_acc = np.mean([pcM[c] for c in a_only]) if a_only else 0
        b_only_acc = np.mean([pcM[c] for c in b_only]) if b_only else 0
        overlap_acc = np.mean([pcM[c] for c in overlap]) if overlap else 0
        
        # Theoretical best: best of A and B for each class
        best_per_class = {c: max(pcA[c], pcB[c]) for c in range(10)}
        theoretical_best = np.mean([best_per_class[c] for c in all_classes])
        merged_on_known = np.mean([pcM[c] for c in all_classes])
        knowledge_transfer = merged_on_known / theoretical_best if theoretical_best > 0 else 0
        
        print(f"  Merged overall: {allM:.3f}")
        print(f"  α = {[round(float(a),2) for a in alphas]}")
        
        print(f"\n  Per-class:")
        print(f"  {'Class':>6s} {'A':>6s} {'B':>6s} {'Merged':>7s} {'Source':>8s}")
        for c in range(10):
            src = "A" if c in a_only else ("B" if c in b_only else ("A∩B" if c in overlap else "?"))
            marker = "✅" if pcM[c] >= 0.5 * max(pcA[c], pcB[c]) else "❌"
            print(f"  {c:>6d} {pcA[c]:>6.3f} {pcB[c]:>6.3f} {pcM[c]:>7.3f} {src:>8s} {marker}")
        
        print(f"\n  Knowledge transfer:")
        if a_only: print(f"    A-only classes {a_only}: acc={a_only_acc:.3f}")
        if b_only: print(f"    B-only classes {b_only}: acc={b_only_acc:.3f}")
        if overlap: print(f"    Overlap classes {overlap}: acc={overlap_acc:.3f}")
        print(f"    Known-class accuracy: {merged_on_known:.3f} / {theoretical_best:.3f} = {knowledge_transfer:.3f}")
        
        results.append({
            'scenario': name, 'classes_A': classes_A, 'classes_B': classes_B,
            'acc_A': round(allA,4), 'acc_B': round(allB,4), 'merged': round(allM,4),
            'a_only_acc': round(a_only_acc,4), 'b_only_acc': round(b_only_acc,4),
            'overlap_acc': round(overlap_acc,4),
            'knowledge_transfer': round(knowledge_transfer,4),
            'per_class': {str(c): round(pcM[c],4) for c in range(10)},
        })
    
    elapsed = time.time() - t0
    print(f"\n{'=' * 75}")
    print("  SUMMARY")
    print(f"{'=' * 75}")
    for r in results:
        print(f"  {r['scenario']:<25s}: A={r['acc_A']:.3f} B={r['acc_B']:.3f} "
              f"M={r['merged']:.3f} KT={r['knowledge_transfer']:.3f}")
    print(f"  Time: {elapsed:.1f}s")
    
    with open("results_e13.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
