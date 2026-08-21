#!/usr/bin/env python3
"""
E10 [L3] — Full MNIST benchmark.
==================================
Full data: 50K train / 10K val / 10K test
3 arch pairs × 3 seeds = 9 runs
3-way split: CMA optimizes on val, evaluate on unseen test.
"""

import numpy as np
import torch
import torch.nn as nn
import time, sys, json, copy

SEEDS = [42, 123, 777]

# ─── Data ────────────────────────────────────────────────────────────

def load_full_mnist():
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
    te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
    
    X_all = torch.stack([tr[i][0] for i in range(len(tr))])
    y_all = torch.tensor([tr[i][1] for i in range(len(tr))])
    X_test = torch.stack([te[i][0] for i in range(len(te))])
    y_test = torch.tensor([te[i][1] for i in range(len(te))])
    
    idx = torch.randperm(60000, generator=torch.Generator().manual_seed(0))
    X_tr = X_all[idx[:50000]]; y_tr = y_all[idx[:50000]]
    X_val = X_all[idx[50000:]]; y_val = y_all[idx[50000:]]
    return X_tr, y_tr, X_val, y_val, X_test, y_test


# ─── Model ───────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(s, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch)-2: layers.append(nn.ReLU())
        s.net = nn.Sequential(*layers); s.arch = arch
    def forward(s, x): return s.net(x)


def train_model(arch, X, y, seed, epochs=8, bs=1024, lr=0.002):
    torch.manual_seed(seed); np.random.seed(seed)
    m = MLP(arch); opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(X); m.train()
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            b = perm[i:i+bs]
            l = nn.CrossEntropyLoss()(m(X[b]), y[b])
            opt.zero_grad(); l.backward(); opt.step()
    return m


def ev(model, X, y):
    model.eval()
    with torch.no_grad():
        p = []
        for i in range(0, len(X), 2000):
            p.append(model(X[i:i+2000]).argmax(1))
        return (torch.cat(p) == y).float().mean().item()


# ─── PCMA (inline for speed) ────────────────────────────────────────

def pcma_merge(mA, mB, X_cal, X_val, y_val, seed, maxiter=30, popsize=12):
    import cma
    archA, archB = mA.arch, mB.arch
    n_hidden = len(archA) - 2
    
    # SVD on calibration data
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
        dmin = min(archA[i+1], archB[i+1])
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
    
    x0 = np.concatenate([s0, [0.5]*n_lay])
    nd = len(x0)
    
    def fitness(x):
        try:
            m = build(x[:nd_s], x[nd_s:].tolist())
            return -ev(m, X_val, y_val)
        except: return 1.0
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd_s+[0.1]*n_lay, [3]*nd_s+[0.9]*n_lay],
        'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best = np.array(es.result.xbest)
    merged = build(best[:nd_s], best[nd_s:].tolist())
    proc = build(s0, [0.5]*n_lay)
    alphas = best[nd_s:].tolist()
    return merged, proc, alphas, nd, es.result.evaluations


def trunc_merge(mA, mB, archA):
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    merged = MLP(archA)
    with torch.no_grad():
        for li in range(len(archA)-1):
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            o, inp = wa.shape
            wt = wb[:o,:inp] if wb.shape[0]>=o and wb.shape[1]>=inp else wa
            bt = bb[:o] if len(bb)>=o else ba
            list(merged.parameters())[li*2].copy_(torch.tensor(0.5*wa+0.5*wt, dtype=torch.float32))
            list(merged.parameters())[li*2+1].copy_(torch.tensor(0.5*ba+0.5*bt, dtype=torch.float32))
    return merged


# ─── Main ────────────────────────────────────────────────────────────

CONFIGS = [
    ("3L_2x",  [784,128,64,10],    [784,256,128,10]),
    ("3L_3x",  [784,64,32,10],     [784,192,96,10]),
    ("4L_2x",  [784,128,64,32,10], [784,256,128,64,10]),
]


def main():
    t0 = time.time()
    print("=" * 75)
    print("  E10 [L3] Full MNIST — 50K/10K/10K split")
    print("  3 configs × 3 seeds = 9 runs")
    print("=" * 75)
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_full_mnist()
    X_cal = X_tr[:5000]  # calibration subset for SVD
    print(f"  Train={len(X_tr)} Val={len(X_val)} Test={len(X_te)} Cal={len(X_cal)}")
    
    results = []
    
    for name, archA, archB in CONFIGS:
        print(f"\n{'━' * 75}")
        print(f"  {name}: A={archA} vs B={archB}")
        print(f"{'━' * 75}")
        
        for seed in SEEDS:
            ts = time.time()
            mA = train_model(archA, X_tr, y_tr, seed)
            mB = train_model(archB, X_tr, y_tr, seed)
            t_train = time.time() - ts
            
            aA_te = ev(mA, X_te, y_te); aB_te = ev(mB, X_te, y_te)
            bp = max(aA_te, aB_te)
            
            # Truncation
            tm = trunc_merge(mA, mB, archA)
            trunc_te = ev(tm, X_te, y_te)
            
            # PCMA
            ts = time.time()
            merged, proc, alphas, nd, ne = pcma_merge(
                mA, mB, X_cal, X_val, y_val, seed, maxiter=30, popsize=12)
            t_cma = time.time() - ts
            
            pcma_val = ev(merged, X_val, y_val)
            pcma_te = ev(merged, X_te, y_te)
            proc_te = ev(proc, X_te, y_te)
            gap = pcma_val - pcma_te
            
            row = {
                'config': name, 'seed': seed,
                'accA': round(aA_te,4), 'accB': round(aB_te,4), 'best_parent': round(bp,4),
                'trunc_te': round(trunc_te,4), 'trunc_ret': round(trunc_te/bp,4),
                'proc_te': round(proc_te,4), 'proc_ret': round(proc_te/bp,4),
                'pcma_val': round(pcma_val,4), 'pcma_te': round(pcma_te,4),
                'pcma_ret': round(pcma_te/bp,4), 'gen_gap': round(gap,4),
                'alphas': [round(float(a),3) for a in alphas],
                'dims': nd, 'evals': ne,
                't_train': round(t_train,1), 't_cma': round(t_cma,1),
            }
            results.append(row)
            
            mk = "✅" if pcma_te/bp >= 0.85 else "🟡"
            print(f"  s={seed}: A={aA_te:.4f} B={aB_te:.4f}")
            print(f"    Trunc  ret={trunc_te/bp:.4f}  test={trunc_te:.4f}")
            print(f"    Proc   ret={proc_te/bp:.4f}  test={proc_te:.4f}")
            print(f"    PCMA   ret={pcma_te/bp:.4f}  test={pcma_te:.4f} val={pcma_val:.4f} gap={gap:+.4f} {mk}")
            print(f"    time: train={t_train:.1f}s cma={t_cma:.1f}s  dims={nd}")
    
    # Summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 75}")
    print("  L3 FINAL RESULTS — Full MNIST")
    print(f"{'=' * 75}")
    
    print(f"\n  {'Config':<8s} {'Trunc':>8s} {'Proc':>8s} {'PCMA':>8s} {'Gap':>8s}")
    print(f"  {'─' * 40}")
    for name in [c[0] for c in CONFIGS]:
        sub = [r for r in results if r['config']==name]
        at = np.mean([r['trunc_ret'] for r in sub])
        ap = np.mean([r['proc_ret'] for r in sub])
        am = np.mean([r['pcma_ret'] for r in sub])
        ag = np.mean([r['gen_gap'] for r in sub])
        print(f"  {name:<8s} {at:>8.4f} {ap:>8.4f} {am:>8.4f} {ag:>+8.4f}")
    
    overall = np.mean([r['pcma_ret'] for r in results])
    mn = min(r['pcma_ret'] for r in results)
    mx = max(r['pcma_ret'] for r in results)
    gap = np.mean([r['gen_gap'] for r in results])
    npass = sum(1 for r in results if r['pcma_ret'] >= 0.85)
    
    print(f"\n  PCMA overall: {overall:.4f} [{mn:.4f}, {mx:.4f}]")
    print(f"  Gen gap: {gap:+.4f}")
    print(f"  Pass (≥0.85): {npass}/{len(results)}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  L3 {'PASS ✅' if npass == len(results) else 'PARTIAL'}")
    
    with open("results_e10_L3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: results_e10_L3.json")


if __name__ == "__main__":
    main()
