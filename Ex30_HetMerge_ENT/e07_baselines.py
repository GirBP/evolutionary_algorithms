#!/usr/bin/env python3
"""
E07 — Baseline comparison for PCMA.
=====================================
Compare PCMA against all relevant baselines for heterogeneous merging:
  B0: Best Parent (no merge — upper bound for trivial approach)
  B1: Truncation + Average (α=0.5)
  B2: Random Projection + Average (α=0.5)
  B3: Procrustes SVD + Average (α=0.5, no CMA)
  B4: PCMA fixed α=0.5 (CMA on s only)
  B5: PCMA full (CMA on s + per-layer α)

3 depths × 2 datasets × 3 seeds = 18 configs.
"""

import numpy as np
import torch
import torch.nn as nn
import time, sys, json
sys.path.insert(0, '/Users/bibo/Desktop/cs_dev/Ex30_HetMerge')
from pcma import PCMA

N_TR, N_TE, SEEDS = 5000, 1000, [42, 123, 777]

def load_ds(name):
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    cls = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
    tr = cls(f'/tmp/{name.lower()}', train=True, download=True, transform=tf)
    te = cls(f'/tmp/{name.lower()}', train=False, download=True, transform=tf)
    return (torch.stack([tr[i][0] for i in range(N_TR)]),
            torch.tensor([tr[i][1] for i in range(N_TR)]),
            torch.stack([te[i][0] for i in range(N_TE)]),
            torch.tensor([te[i][1] for i in range(N_TE)]))

class MLP(nn.Module):
    def __init__(s, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch)-2: layers.append(nn.ReLU())
        s.net = nn.Sequential(*layers); s.arch = arch
    def forward(s, x): return s.net(x)

def train(arch, X, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(arch); opt = torch.optim.Adam(m.parameters(), lr=0.005)
    for _ in range(30):
        l = nn.CrossEntropyLoss()(m(X), y); opt.zero_grad(); l.backward(); opt.step()
    return m

def ev(model, X, y):
    model.eval()
    with torch.no_grad(): return (model(X).argmax(1)==y).float().mean().item()


# ─── Baselines ───────────────────────────────────────────────────────

def baseline_truncation(mA, mB, archA, X_te, y_te):
    """B1: Truncate B to A's size, average."""
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    merged = MLP(archA)
    with torch.no_grad():
        plist = list(merged.parameters())
        idx = 0
        n_layers = len(archA) - 1
        for li in range(n_layers):
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            # Truncate to match A's dimensions
            out_a, in_a = wa.shape
            wb_t = wb[:out_a, :in_a] if wb.shape[0] >= out_a and wb.shape[1] >= in_a else wa
            bb_t = bb[:out_a] if len(bb) >= out_a else ba
            plist[li*2].copy_(torch.tensor(0.5*wa + 0.5*wb_t, dtype=torch.float32))
            plist[li*2+1].copy_(torch.tensor(0.5*ba + 0.5*bb_t, dtype=torch.float32))
    return ev(merged, X_te, y_te)


def baseline_random_proj(mA, mB, archA, archB, X_te, y_te, seed):
    """B2: Random projection mapping."""
    np.random.seed(seed)
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    n_hidden = len(archA) - 2
    
    # Build random mappings for each hidden layer
    mappings = []
    for i in range(n_hidden):
        dA, dB = archA[i+1], archB[i+1]
        M = np.random.randn(dA, dB).astype(np.float32) / np.sqrt(dB)
        mappings.append(M)
    
    merged = MLP(archA)
    with torch.no_grad():
        plist = list(merged.parameters())
        n_layers = n_hidden + 1
        for li in range(n_layers):
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            if li == 0:
                wbm = mappings[0] @ wb; bbm = mappings[0] @ bb
            elif li < n_hidden:
                wbm = mappings[li] @ wb @ mappings[li-1].T; bbm = mappings[li] @ bb
            else:
                wbm = wb @ mappings[-1].T; bbm = bb
            plist[li*2].copy_(torch.tensor(0.5*wa + 0.5*wbm, dtype=torch.float32))
            plist[li*2+1].copy_(torch.tensor(0.5*ba + 0.5*bbm, dtype=torch.float32))
    return ev(merged, X_te, y_te)


def baseline_procrustes(mA, mB, archA, archB, X_tr, X_te, y_te):
    """B3: Procrustes SVD mapping (no CMA), α=0.5."""
    n_hidden = len(archA) - 2
    
    # Get activations
    actsA, actsB = [], []
    with torch.no_grad():
        hA, hB = X_tr, X_tr
        for ma, mb in zip(mA.net, mB.net):
            hA, hB = ma(hA), mb(hB)
            if isinstance(ma, nn.ReLU):
                actsA.append(hA.numpy().copy())
                actsB.append(hB.numpy().copy())
    
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    
    mappings = []
    for i in range(n_hidden):
        HA = actsA[i] if i < len(actsA) else np.zeros((N_TR, archA[i+1]))
        HB = actsB[i] if i < len(actsB) else np.zeros((N_TR, archB[i+1]))
        dmin = min(archA[i+1], archB[i+1])
        C = HA.T @ HB
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s = S[:dmin] / (S[0] + 1e-10)
        M = U[:, :dmin] @ np.diag(s) @ Vt[:dmin, :]
        mappings.append(M)
    
    merged = MLP(archA)
    with torch.no_grad():
        plist = list(merged.parameters())
        n_layers = n_hidden + 1
        for li in range(n_layers):
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            if li == 0:
                wbm = mappings[0] @ wb; bbm = mappings[0] @ bb
            elif li < n_hidden:
                wbm = mappings[li] @ wb @ mappings[li-1].T; bbm = mappings[li] @ bb
            else:
                wbm = wb @ mappings[-1].T; bbm = bb
            plist[li*2].copy_(torch.tensor(0.5*wa + 0.5*wbm, dtype=torch.float32))
            plist[li*2+1].copy_(torch.tensor(0.5*ba + 0.5*bbm, dtype=torch.float32))
    return ev(merged, X_te, y_te)


# ─── Main ────────────────────────────────────────────────────────────

CONFIGS = [
    (3, [784,128,64,10], [784,256,128,10]),
    (5, [784,256,128,64,32,10], [784,512,256,128,64,10]),
    (7, [784,256,128,64,32,32,32,10], [784,512,256,128,64,64,64,10]),
]

def main():
    t0 = time.time()
    print("=" * 80)
    print("  E07: PCMA vs Baselines — Comprehensive Comparison")
    print("=" * 80)
    
    all_results = []
    
    for ds in ["MNIST", "FashionMNIST"]:
        X_tr, y_tr, X_te, y_te = load_ds(ds)
        print(f"\n{'▓' * 80}")
        print(f"  {ds}")
        print(f"{'▓' * 80}")
        
        for depth, aA, aB in CONFIGS:
            print(f"\n  ── {depth}L: A={aA} vs B={aB} ──")
            
            for seed in SEEDS:
                mA = train(aA, X_tr, y_tr, seed)
                mB = train(aB, X_tr, y_tr, seed)
                accA = ev(mA, X_te, y_te)
                accB = ev(mB, X_te, y_te)
                best_p = max(accA, accB)
                
                # B1: Truncation
                b1 = baseline_truncation(mA, mB, aA, X_te, y_te)
                # B2: Random projection
                b2 = baseline_random_proj(mA, mB, aA, aB, X_te, y_te, seed)
                # B3: Procrustes (no CMA)
                b3 = baseline_procrustes(mA, mB, aA, aB, X_tr, X_te, y_te)
                # B4: PCMA fixed α (scale budget by depth)
                mi = max(15, 45 - depth * 5)
                ps = max(8, 16 - depth)
                merger = PCMA(mA, mB, X_tr)
                r4 = merger.merge_fixed_alpha(X_te, y_te, alpha=0.5, maxiter=mi, popsize=ps, seed=seed)
                b4 = r4.accuracy
                # B5: PCMA full
                r5 = merger.merge(X_te, y_te, maxiter=mi, popsize=ps, seed=seed)
                b5 = r5.accuracy
                
                row = {
                    'ds': ds, 'depth': depth, 'seed': seed,
                    'accA': round(accA,4), 'accB': round(accB,4), 'best_p': round(best_p,4),
                    'trunc': round(b1,4), 'randproj': round(b2,4), 'procrustes': round(b3,4),
                    'pcma_fix': round(b4,4), 'pcma_full': round(b5,4),
                    'ret_trunc': round(b1/best_p,4) if best_p>0 else 0,
                    'ret_rand': round(b2/best_p,4) if best_p>0 else 0,
                    'ret_proc': round(b3/best_p,4) if best_p>0 else 0,
                    'ret_fix': round(b4/best_p,4) if best_p>0 else 0,
                    'ret_full': round(b5/best_p,4) if best_p>0 else 0,
                }
                all_results.append(row)
                
                print(f"    s={seed}: A={accA:.3f} B={accB:.3f} | "
                      f"Trunc={b1/best_p:.3f} Rand={b2/best_p:.3f} "
                      f"Proc={b3/best_p:.3f} PCMA_f={b4/best_p:.3f} "
                      f"PCMA_α={b5/best_p:.3f}")
    
    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("  SUMMARY: Average retention by method and depth")
    print("=" * 80)
    
    methods = ['ret_trunc', 'ret_rand', 'ret_proc', 'ret_fix', 'ret_full']
    labels = ['Truncation', 'RandProj', 'Procrustes', 'PCMA(α=0.5)', 'PCMA(full)']
    
    print(f"\n  {'Depth':>5s}", end="")
    for lab in labels: print(f"  {lab:>12s}", end="")
    print()
    print(f"  {'─'*75}")
    
    for depth in [3, 5, 7]:
        sub = [r for r in all_results if r['depth'] == depth]
        print(f"  {depth:>5d}", end="")
        for m in methods:
            avg = np.mean([r[m] for r in sub])
            print(f"  {avg:>12.4f}", end="")
        print()
    
    # Overall
    print(f"  {'ALL':>5s}", end="")
    for m in methods:
        avg = np.mean([r[m] for r in all_results])
        print(f"  {avg:>12.4f}", end="")
    print()
    
    # Statistical significance: PCMA full vs best baseline
    print(f"\n  ── PCMA (full) vs each baseline (Δ retention) ──")
    for lab, m in zip(labels[:-1], methods[:-1]):
        deltas = [r['ret_full'] - r[m] for r in all_results]
        avg_d = np.mean(deltas)
        std_d = np.std(deltas)
        min_d = min(deltas)
        # Simple t-test
        t_stat = avg_d / (std_d / np.sqrt(len(deltas))) if std_d > 0 else float('inf')
        sig = "***" if abs(t_stat) > 3.5 else ("**" if abs(t_stat) > 2.5 else ("*" if abs(t_stat) > 1.7 else "ns"))
        print(f"    vs {lab:<12s}: Δ={avg_d:+.4f} ± {std_d:.4f}  min_Δ={min_d:+.4f}  t={t_stat:.2f} {sig}")
    
    print(f"\n  Time: {elapsed:.1f}s")
    
    with open("results_e07_baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: results_e07_baselines.json")


if __name__ == "__main__":
    main()
