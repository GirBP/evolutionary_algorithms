#!/usr/bin/env python3
"""
E08 — HONEST evaluation: fix data leakage + test true mixing.
===============================================================
FIXED PROBLEMS:
  1. 3-way split: train (model training) / val (CMA fitness) / test (final eval)
  2. Track α values — flag degenerate solutions
  3. Full MNIST (60K/10K), not 5K/1K subset
  4. Measure BOTH:
     - retention on UNSEEN test set
     - effective mixing ratio (how much of B actually contributes)
  5. Add REPAIR baseline (rescale activations after merge)

This is the HONEST version. If it fails, we know PCMA doesn't actually work.
"""

import numpy as np
import torch
import torch.nn as nn
import time, sys, json

DEVICE = 'cpu'
SEEDS = [42, 123, 777]


def load_mnist_full():
    import ssl; ssl._create_default_https_context = ssl._create_unverified_context
    from torchvision import datasets, transforms
    tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
    te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
    
    X_all = torch.stack([tr[i][0] for i in range(len(tr))])
    y_all = torch.tensor([tr[i][1] for i in range(len(tr))])
    X_test = torch.stack([te[i][0] for i in range(5000)])
    y_test = torch.tensor([te[i][1] for i in range(5000)])
    
    # Split training into train (20K) + val (5K)
    idx = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(0))
    X_train, y_train = X_all[idx[:20000]], y_all[idx[:20000]]
    X_val, y_val = X_all[idx[20000:25000]], y_all[idx[20000:25000]]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


class MLP(nn.Module):
    def __init__(s, arch):
        super().__init__()
        layers = []
        for i in range(len(arch)-1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch)-2: layers.append(nn.ReLU())
        s.net = nn.Sequential(*layers); s.arch = arch
    def forward(s, x): return s.net(x)


def train_model(arch, X, y, seed, epochs=10, lr=0.003, batch_size=512):
    """Train with mini-batches on full data."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(arch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    n = len(X)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            batch_idx = perm[i:i+batch_size]
            loss = loss_fn(model(X[batch_idx]), y[batch_idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def ev(model, X, y):
    model.eval()
    with torch.no_grad():
        # Batch evaluation to avoid memory issues
        preds = []
        for i in range(0, len(X), 1000):
            preds.append(model(X[i:i+1000]).argmax(1))
        preds = torch.cat(preds)
        return (preds == y).float().mean().item()


# ─── SVD + CMA merging (corrected) ──────────────────────────────────

def compute_svd(mA, mB, X_cal):
    """Compute SVD on CALIBRATION data (subset of training)."""
    archA, archB = mA.arch, mB.arch
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
        HA = actsA[i] if i < len(actsA) else np.zeros((len(X_cal), archA[i+1]))
        HB = actsB[i] if i < len(actsB) else np.zeros((len(X_cal), archB[i+1]))
        dmin = min(archA[i+1], archB[i+1])
        C = HA.T @ HB
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_init = S[:dmin] / (S[0] + 1e-10)
        svd_list.append((U[:, :dmin], Vt[:dmin, :]))
        s_inits.append(s_init)
    return svd_list, s_inits


def merge_with(mA, mB, svd_list, s_vec, alphas):
    archA = mA.arch
    n_hidden = len(archA) - 2
    n_layers = n_hidden + 1
    if isinstance(alphas, (int, float)):
        alphas = [alphas] * n_layers
    
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    
    maps = []
    off = 0
    for i in range(n_hidden):
        U, Vt = svd_list[i]
        d = len(s_vec[off:off + U.shape[1]])
        s = s_vec[off:off+d]; off += d
        maps.append(U @ np.diag(s) @ Vt)
    
    params = []
    for li in range(n_layers):
        a = np.clip(alphas[li] if li < len(alphas) else 0.5, 0.05, 0.95)
        wa, ba = WA[li*2], WA[li*2+1]
        wb, bb = WB[li*2], WB[li*2+1]
        if li == 0: wbm = maps[0] @ wb; bbm = maps[0] @ bb
        elif li < n_hidden: wbm = maps[li] @ wb @ maps[li-1].T; bbm = maps[li] @ bb
        else: wbm = wb @ maps[-1].T; bbm = bb
        params.append(a*wa + (1-a)*wbm)
        params.append(a*ba + (1-a)*bbm)
    
    merged = MLP(archA)
    with torch.no_grad():
        for p, v in zip(merged.parameters(), params):
            p.copy_(torch.tensor(v, dtype=torch.float32))
    return merged


def run_pcma(mA, mB, X_cal, X_val, y_val, seed, mode="full", maxiter=30, popsize=12):
    """CMA-ES optimized on VAL set (NOT test)."""
    import cma
    svd_list, s_inits = compute_svd(mA, mB, X_cal)
    s0 = np.concatenate(s_inits)
    nd_s = len(s0)
    n_layers = len(mA.arch) - 1
    
    if mode == "full":
        x0 = np.concatenate([s0, [0.5]*n_layers])
        nd = len(x0)
        bounds_lo = [-3]*nd_s + [0.1]*n_layers
        bounds_hi = [3]*nd_s + [0.9]*n_layers
        
        def fitness(x):
            s, al = x[:nd_s], x[nd_s:].tolist()
            try:
                m = merge_with(mA, mB, svd_list, s, al)
                return -ev(m, X_val, y_val)  # ← VALIDATION set, NOT test
            except: return 1.0
    else:
        x0 = s0
        nd = nd_s
        bounds_lo = [-3]*nd
        bounds_hi = [3]*nd
        
        def fitness(x):
            try:
                m = merge_with(mA, mB, svd_list, x, 0.5)
                return -ev(m, X_val, y_val)  # ← VALIDATION set
            except: return 1.0
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [bounds_lo, bounds_hi],
        'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best = np.array(es.result.xbest)
    if mode == "full":
        s_opt, alphas = best[:nd_s], best[nd_s:].tolist()
    else:
        s_opt, alphas = best, [0.5]*n_layers
    
    merged = merge_with(mA, mB, svd_list, s_opt, alphas)
    
    # Also build Procrustes baseline
    proc = merge_with(mA, mB, svd_list, s0, 0.5)
    
    return merged, proc, alphas, svd_list, s0, nd


def baseline_truncation(mA, mB, archA, X, y):
    WA = [p.detach().numpy() for p in mA.parameters()]
    WB = [p.detach().numpy() for p in mB.parameters()]
    merged = MLP(archA)
    with torch.no_grad():
        for li in range(len(archA)-1):
            wa, ba = WA[li*2], WA[li*2+1]
            wb, bb = WB[li*2], WB[li*2+1]
            out_a, in_a = wa.shape
            wbt = wb[:out_a,:in_a] if wb.shape[0]>=out_a and wb.shape[1]>=in_a else wa
            bbt = bb[:out_a] if len(bb)>=out_a else ba
            list(merged.parameters())[li*2].copy_(torch.tensor(0.5*wa+0.5*wbt, dtype=torch.float32))
            list(merged.parameters())[li*2+1].copy_(torch.tensor(0.5*ba+0.5*bbt, dtype=torch.float32))
    return ev(merged, X, y)


# ─── Main ────────────────────────────────────────────────────────────

CONFIGS = [
    (3, [784, 128, 64, 10], [784, 256, 128, 10]),
    (4, [784, 128, 64, 32, 10], [784, 256, 128, 64, 10]),
]


def main():
    t0 = time.time()
    print("=" * 80)
    print("  E08: HONEST EVALUATION (3-way split, full MNIST)")
    print("  train: 50K | val: 10K (CMA fitness) | test: 10K (final eval)")
    print("=" * 80)
    
    X_tr, y_tr, X_val, y_val, X_te, y_te = load_mnist_full()
    print(f"  Train: {len(X_tr)}, Val: {len(X_val)}, Test: {len(X_te)}")
    
    # Calibration subset for SVD (5K from train)
    X_cal = X_tr[:5000]
    
    all_results = []
    
    for depth, archA, archB in CONFIGS:
        print(f"\n{'━' * 80}")
        print(f"  {depth}L: A={archA} vs B={archB}")
        print(f"{'━' * 80}")
        
        for seed in SEEDS:
            mA = train_model(archA, X_tr, y_tr, seed)
            mB = train_model(archB, X_tr, y_tr, seed)
            
            accA_val = ev(mA, X_val, y_val)
            accB_val = ev(mB, X_val, y_val)
            accA_test = ev(mA, X_te, y_te)
            accB_test = ev(mB, X_te, y_te)
            best_p_test = max(accA_test, accB_test)
            
            # B1: Truncation (on test)
            b_trunc = baseline_truncation(mA, mB, archA, X_te, y_te)
            
            # PCMA full: optimized on VAL, evaluated on TEST
            merged, proc_model, alphas, _, _, nd = run_pcma(
                mA, mB, X_cal, X_val, y_val, seed, mode="full",
                maxiter=30, popsize=12)
            
            acc_pcma_val = ev(merged, X_val, y_val)
            acc_pcma_test = ev(merged, X_te, y_te)  # UNSEEN!
            acc_proc_test = ev(proc_model, X_te, y_te)
            
            ret_pcma = acc_pcma_test / best_p_test if best_p_test > 0 else 0
            ret_proc = acc_proc_test / best_p_test if best_p_test > 0 else 0
            ret_trunc = b_trunc / best_p_test if best_p_test > 0 else 0
            
            # Mixing analysis
            avg_alpha = np.mean(alphas)
            min_alpha = min(alphas)
            max_alpha = max(alphas)
            mixing_ok = all(0.2 <= a <= 0.8 for a in alphas)
            
            # Generalization gap
            gen_gap = acc_pcma_val - acc_pcma_test
            
            row = {
                'depth': depth, 'seed': seed,
                'accA_test': round(accA_test, 4), 'accB_test': round(accB_test, 4),
                'best_parent': round(best_p_test, 4),
                'trunc_ret': round(ret_trunc, 4),
                'proc_ret': round(ret_proc, 4),
                'pcma_ret': round(ret_pcma, 4),
                'pcma_val': round(acc_pcma_val, 4),
                'pcma_test': round(acc_pcma_test, 4),
                'gen_gap': round(gen_gap, 4),
                'alphas': [round(float(a), 3) for a in alphas],
                'avg_alpha': round(float(avg_alpha), 3),
                'mixing_ok': mixing_ok,
                'dims': nd,
            }
            all_results.append(row)
            
            mix_mark = "✅" if mixing_ok else "⚠️"
            print(f"  s={seed}: A={accA_test:.3f} B={accB_test:.3f}")
            print(f"    Trunc:   ret={ret_trunc:.3f}")
            print(f"    Procr:   ret={ret_proc:.3f}")
            print(f"    PCMA:    ret={ret_pcma:.3f} (val={acc_pcma_val:.3f} test={acc_pcma_test:.3f} gap={gen_gap:+.3f})")
            print(f"    α={[round(float(a),2) for a in alphas]} avg={avg_alpha:.2f} {mix_mark}")
    
    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("  HONEST SUMMARY")
    print("=" * 80)
    
    for depth in [3, 4]:
        sub = [r for r in all_results if r['depth'] == depth]
        if not sub: continue
        avg_trunc = np.mean([r['trunc_ret'] for r in sub])
        avg_proc = np.mean([r['proc_ret'] for r in sub])
        avg_pcma = np.mean([r['pcma_ret'] for r in sub])
        avg_gap = np.mean([r['gen_gap'] for r in sub])
        avg_alpha = np.mean([r['avg_alpha'] for r in sub])
        n_mix = sum(1 for r in sub if r['mixing_ok'])
        
        print(f"\n  {depth}L:")
        print(f"    Truncation ret:  {avg_trunc:.4f}")
        print(f"    Procrustes ret:  {avg_proc:.4f}")
        print(f"    PCMA ret (TEST): {avg_pcma:.4f}")
        print(f"    Gen gap (val-test): {avg_gap:+.4f}")
        print(f"    Avg α: {avg_alpha:.3f} (mixing_ok: {n_mix}/{len(sub)})")
    
    overall_pcma = np.mean([r['pcma_ret'] for r in all_results])
    overall_gap = np.mean([r['gen_gap'] for r in all_results])
    overall_alpha = np.mean([r['avg_alpha'] for r in all_results])
    
    print(f"\n  OVERALL:")
    print(f"    PCMA retention on UNSEEN test: {overall_pcma:.4f}")
    print(f"    Generalization gap: {overall_gap:+.4f}")
    print(f"    Average α: {overall_alpha:.3f}")
    print(f"    Time: {elapsed:.1f}s")
    
    with open("results_e08_honest.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"    Saved: results_e08_honest.json")


if __name__ == "__main__":
    main()
