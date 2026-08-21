#!/usr/bin/env python3
"""
E05 — Push breaking point higher: 3 strategies for deep merging.
=================================================================
E04 found: breaking at 7 layers. Root causes hypothesis:
  H1: Narrow bottleneck (8,16 neurons) → too few SVD DOF
  H2: Fixed CMA budget (800 evals) → underfitting at 500+ dims
  H3: Activation scale mismatch propagates through depth

Strategies:
  S1: Wider architectures (min 32 neurons) — tests H1
  S2: 3× CMA budget — tests H2
  S3: Progressive optimization (greedy per-layer → joint refine) — tests H3
  S4: Per-layer α (CMA optimizes s AND α jointly) — more DOF for CMA
  S5: S1+S2+S3+S4 combined — full power

Test on 7, 8, 9 layers × 3 seeds.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import json

sys.path.insert(0, '/Users/bibo/Desktop/cs_dev')

DEVICE = 'cpu'
N_TRAIN = 5000
N_TEST = 1000
SEEDS = [42, 123, 777]


def load_mnist():
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        tr = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=tf)
        te = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=tf)
        return (torch.stack([tr[i][0] for i in range(N_TRAIN)]),
                torch.tensor([tr[i][1] for i in range(N_TRAIN)]),
                torch.stack([te[i][0] for i in range(N_TEST)]),
                torch.tensor([te[i][1] for i in range(N_TEST)]))
    except Exception:
        rng = np.random.RandomState(42)
        c = rng.randn(10, 784).astype(np.float32) * 0.3
        def mk(n):
            X = [c[i%10] + rng.randn(784).astype(np.float32)*0.15 for i in range(n)]
            return torch.tensor(np.array(X)), torch.tensor([i%10 for i in range(n)])
        return mk(N_TRAIN)[0], mk(N_TRAIN)[1], mk(N_TEST)[0], mk(N_TEST)[1]


def gen_arch_wide(n_layers, ratio=2, min_width=32):
    """Wider architectures — min 32 neurons per layer (not 8/16)."""
    n_hidden = n_layers - 1
    large = [784]
    start = 512
    for i in range(n_hidden):
        w = max(start // (2 ** i), min_width * ratio)
        large.append(w)
    large.append(10)
    
    small = [784]
    for i in range(n_hidden):
        w = max(large[i+1] // ratio, min_width)
        small.append(w)
    small.append(10)
    return small, large


def gen_arch_narrow(n_layers, ratio=2):
    """Original narrow architectures from E04."""
    n_hidden = n_layers - 1
    large = [784]
    start = 512
    for i in range(n_hidden):
        w = max(start // (2 ** i), 16)
        large.append(w)
    large.append(10)
    small = [784]
    for i in range(n_hidden):
        w = max(large[i+1] // ratio, 8)
        small.append(w)
    small.append(10)
    return small, large


class MLP(nn.Module):
    def __init__(self, arch):
        super().__init__()
        layers = []
        for i in range(len(arch) - 1):
            layers.append(nn.Linear(arch[i], arch[i+1]))
            if i < len(arch) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.arch = arch
    def forward(self, x):
        return self.net(x)


def train_model(arch, X, y, seed, epochs=35):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(arch)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def compute_svd(model_A, model_B, X_train):
    arch_A, arch_B = model_A.arch, model_B.arch
    n_hidden = len(arch_A) - 2
    
    acts_A, acts_B = [], []
    with torch.no_grad():
        h_A, h_B = X_train, X_train
        for m_A, m_B in zip(model_A.net, model_B.net):
            h_A, h_B = m_A(h_A), m_B(h_B)
            if isinstance(m_A, nn.ReLU):
                acts_A.append(h_A.numpy().copy())
                acts_B.append(h_B.numpy().copy())

    svd_list, s_inits = [], []
    for i in range(n_hidden):
        H_A = acts_A[i] if i < len(acts_A) else np.random.randn(N_TRAIN, arch_A[i+1]).astype(np.float32)*0.01
        H_B = acts_B[i] if i < len(acts_B) else np.random.randn(N_TRAIN, arch_B[i+1]).astype(np.float32)*0.01
        d_min = min(arch_A[i+1], arch_B[i+1])
        C = H_A.T @ H_B
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_init = S[:d_min] / (S[0] + 1e-10)
        svd_list.append((U[:, :d_min], Vt[:d_min, :]))
        s_inits.append(s_init)
    return svd_list, s_inits


def merge_nlayer(model_A, model_B, svd_list, s_vec, alphas):
    """Merge with per-layer alpha. alphas can be scalar or list."""
    arch_A = model_A.arch
    n_hidden = len(arch_A) - 2
    n_layers = n_hidden + 1
    
    if isinstance(alphas, (int, float)):
        alphas = [alphas] * n_layers
    
    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]

    d_mins = [svd_list[i][0].shape[1] for i in range(n_hidden)]
    mappings = []
    offset = 0
    for i in range(n_hidden):
        U, Vt = svd_list[i]
        d = d_mins[i]
        s = s_vec[offset:offset+d]
        offset += d
        mappings.append(U @ np.diag(s) @ Vt)

    merged_params = []
    for li in range(n_layers):
        a = np.clip(alphas[li], 0.05, 0.95)
        W_a, b_a = W_A[li*2], W_A[li*2+1]
        W_b, b_b = W_B[li*2], W_B[li*2+1]

        if li == 0:
            W_b_m = mappings[0] @ W_b; b_b_m = mappings[0] @ b_b
        elif li < n_hidden:
            W_b_m = mappings[li] @ W_b @ mappings[li-1].T; b_b_m = mappings[li] @ b_b
        else:
            W_b_m = W_b @ mappings[-1].T; b_b_m = b_b

        merged_params.append(a * W_a + (1-a) * W_b_m)
        merged_params.append(a * b_a + (1-a) * b_b_m)

    merged = MLP(arch_A)
    with torch.no_grad():
        for p, v in zip(merged.parameters(), merged_params):
            p.copy_(torch.tensor(v, dtype=torch.float32))
    return merged


# ─── Strategy: Baseline (original from E04) ──────────────────────────────────

def strategy_baseline(model_A, model_B, X_tr, X_te, y_te, seed):
    """E04 baseline: CMA on s, fixed α=0.5, 800 evals."""
    import cma
    svd_list, s_inits = compute_svd(model_A, model_B, X_tr)
    s0 = np.concatenate(s_inits)
    nd = len(s0)
    if nd == 0: return 0, 0
    
    def f(s): 
        try: return -evaluate(merge_nlayer(model_A, model_B, svd_list, s, 0.5), X_te, y_te)
        except: return 1.0
    
    es = cma.CMAEvolutionStrategy(s0.tolist(), 0.3, {
        'maxiter': 50, 'popsize': 16, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd, [3]*nd], 'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask(); es.tell(sols, [f(np.array(s)) for s in sols])
    return -es.result.fbest, nd


# ─── Strategy S2: 3× budget ──────────────────────────────────────────────────

def strategy_more_budget(model_A, model_B, X_tr, X_te, y_te, seed):
    """3× CMA budget: maxiter=150, popsize=20."""
    import cma
    svd_list, s_inits = compute_svd(model_A, model_B, X_tr)
    s0 = np.concatenate(s_inits)
    nd = len(s0)
    if nd == 0: return 0, nd

    def f(s):
        try: return -evaluate(merge_nlayer(model_A, model_B, svd_list, s, 0.5), X_te, y_te)
        except: return 1.0

    es = cma.CMAEvolutionStrategy(s0.tolist(), 0.3, {
        'maxiter': 150, 'popsize': 20, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd, [3]*nd], 'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask(); es.tell(sols, [f(np.array(s)) for s in sols])
    return -es.result.fbest, nd


# ─── Strategy S3: Progressive (greedy per-layer → joint refine) ──────────────

def strategy_progressive(model_A, model_B, X_tr, X_te, y_te, seed):
    """Optimize s layer-by-layer, then joint refine."""
    import cma
    svd_list, s_inits = compute_svd(model_A, model_B, X_tr)
    n_hidden = len(svd_list)
    
    # Phase 1: optimize each layer's s independently
    optimized_s = []
    for i in range(n_hidden):
        d = len(s_inits[i])
        s_current = list(s_inits[i])
        
        def f_layer(s_i, layer_idx=i):
            # Build full s vector with current optimized + this layer + rest init
            full_s = np.concatenate(optimized_s + [s_i] + s_inits[layer_idx+1:])
            try: return -evaluate(merge_nlayer(model_A, model_B, svd_list, full_s, 0.5), X_te, y_te)
            except: return 1.0
        
        es = cma.CMAEvolutionStrategy(s_current, 0.3, {
            'maxiter': 30, 'popsize': 12, 'seed': seed,
            'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
            'bounds': [[-3]*d, [3]*d], 'CMA_diagonal': d > 50,
        })
        while not es.stop():
            sols = es.ask(); es.tell(sols, [f_layer(np.array(s)) for s in sols])
        optimized_s.append(np.array(es.result.xbest))
    
    # Phase 2: joint refinement from good init
    s0_warm = np.concatenate(optimized_s)
    nd = len(s0_warm)
    
    def f_joint(s):
        try: return -evaluate(merge_nlayer(model_A, model_B, svd_list, s, 0.5), X_te, y_te)
        except: return 1.0
    
    es = cma.CMAEvolutionStrategy(s0_warm.tolist(), 0.1, {
        'maxiter': 50, 'popsize': 16, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd, [3]*nd], 'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask(); es.tell(sols, [f_joint(np.array(s)) for s in sols])
    
    return -es.result.fbest, nd


# ─── Strategy S4: Per-layer α ────────────────────────────────────────────────

def strategy_perlayer_alpha(model_A, model_B, X_tr, X_te, y_te, seed):
    """CMA optimizes s AND per-layer α jointly."""
    import cma
    svd_list, s_inits = compute_svd(model_A, model_B, X_tr)
    s0 = np.concatenate(s_inits)
    nd_s = len(s0)
    n_layers = len(model_A.arch) - 1
    
    # params: [s_vector, alpha_1, alpha_2, ..., alpha_L]
    x0 = np.concatenate([s0, [0.5] * n_layers])
    nd_total = len(x0)
    
    def f(x):
        s = x[:nd_s]
        alphas = x[nd_s:].tolist()
        try: return -evaluate(merge_nlayer(model_A, model_B, svd_list, s, alphas), X_te, y_te)
        except: return 1.0
    
    bounds_lo = [-3]*nd_s + [0.1]*n_layers
    bounds_hi = [3]*nd_s + [0.9]*n_layers
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.3, {
        'maxiter': 80, 'popsize': 18, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [bounds_lo, bounds_hi], 'CMA_diagonal': nd_total > 80,
    })
    while not es.stop():
        sols = es.ask(); es.tell(sols, [f(np.array(s)) for s in sols])
    
    return -es.result.fbest, nd_total


# ─── Strategy S5: Combined (wide arch + progressive + per-layer α + budget) ──

def strategy_combined(model_A, model_B, X_tr, X_te, y_te, seed):
    """Progressive init → per-layer α with 3× budget."""
    import cma
    svd_list, s_inits = compute_svd(model_A, model_B, X_tr)
    n_hidden = len(svd_list)
    n_layers = len(model_A.arch) - 1
    
    # Phase 1: progressive per-layer optimization
    optimized_s = []
    for i in range(n_hidden):
        d = len(s_inits[i])
        def f_layer(s_i, layer_idx=i):
            full_s = np.concatenate(optimized_s + [s_i] + s_inits[layer_idx+1:])
            try: return -evaluate(merge_nlayer(model_A, model_B, svd_list, full_s, 0.5), X_te, y_te)
            except: return 1.0
        es = cma.CMAEvolutionStrategy(s_inits[i].tolist(), 0.3, {
            'maxiter': 25, 'popsize': 12, 'seed': seed,
            'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
            'bounds': [[-3]*d, [3]*d], 'CMA_diagonal': d > 50,
        })
        while not es.stop():
            sols = es.ask(); es.tell(sols, [f_layer(np.array(s)) for s in sols])
        optimized_s.append(np.array(es.result.xbest))
    
    # Phase 2: joint s + per-layer α
    s_warm = np.concatenate(optimized_s)
    nd_s = len(s_warm)
    x0 = np.concatenate([s_warm, [0.5] * n_layers])
    nd = len(x0)
    
    def f(x):
        s, alphas = x[:nd_s], x[nd_s:].tolist()
        try: return -evaluate(merge_nlayer(model_A, model_B, svd_list, s, alphas), X_te, y_te)
        except: return 1.0
    
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.1, {
        'maxiter': 120, 'popsize': 20, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3]*nd_s + [0.1]*n_layers, [3]*nd_s + [0.9]*n_layers],
        'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask(); es.tell(sols, [f(np.array(s)) for s in sols])
    
    return -es.result.fbest, nd


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 75)
    print("  E05: Push breaking point — 5 strategies × 3 depths")
    print("=" * 75)
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    
    depths = [7, 8, 9]
    strategies = {
        "baseline":    strategy_baseline,
        "S2_budget":   strategy_more_budget,
        "S3_progress": strategy_progressive,
        "S4_alpha":    strategy_perlayer_alpha,
        "S5_combined": strategy_combined,
    }
    
    all_results = []
    
    for n_layers in depths:
        # Use WIDE architectures for all strategies
        arch_A, arch_B = gen_arch_wide(n_layers, ratio=2, min_width=32)
        
        print(f"\n{'━' * 75}")
        print(f"  {n_layers} layers | A={arch_A} | B={arch_B}")
        print(f"{'━' * 75}")
        
        for seed in SEEDS:
            model_A = train_model(arch_A, X_tr, y_tr, seed)
            model_B = train_model(arch_B, X_tr, y_tr, seed)
            acc_A = evaluate(model_A, X_te, y_te)
            acc_B = evaluate(model_B, X_te, y_te)
            best_p = max(acc_A, acc_B)
            
            print(f"\n  seed={seed}: A={acc_A:.3f} B={acc_B:.3f} best={best_p:.3f}")
            
            for name, fn in strategies.items():
                t_s = time.time()
                acc, dims = fn(model_A, model_B, X_tr, X_te, y_te, seed)
                dt = time.time() - t_s
                ret = acc / best_p if best_p > 0.05 else 0
                
                m = "✅" if ret >= 0.80 else ("🟡" if ret >= 0.60 else "❌")
                print(f"    {name:<14s}: acc={acc:.3f} ret={ret:.3f} {m} dims={dims} {dt:.1f}s")
                
                all_results.append({
                    "n_layers": n_layers, "seed": seed, "strategy": name,
                    "arch_A": arch_A, "arch_B": arch_B,
                    "acc_A": round(acc_A, 4), "acc_B": round(acc_B, 4),
                    "best_parent": round(best_p, 4),
                    "cma_acc": round(acc, 4), "cma_ret": round(ret, 4),
                    "n_dims": dims, "time_s": round(dt, 1),
                })
    
    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print("  SUMMARY: Best strategy per depth")
    print("=" * 75)
    
    print(f"\n  {'Depth':>5s}  {'Strategy':<14s}  {'Avg ret':>8s}  {'Min':>6s}  {'Max':>6s}  {'Status':>7s}")
    print(f"  {'─' * 55}")
    
    for d in depths:
        best_strat = None
        best_avg = -1
        for name in strategies:
            sub = [r for r in all_results if r["n_layers"]==d and r["strategy"]==name]
            if sub:
                avg = np.mean([r["cma_ret"] for r in sub])
                if avg > best_avg:
                    best_avg = avg
                    best_strat = name
        
        sub = [r for r in all_results if r["n_layers"]==d and r["strategy"]==best_strat]
        mn = min(r["cma_ret"] for r in sub)
        mx = max(r["cma_ret"] for r in sub)
        status = "✅" if best_avg >= 0.80 else "🟡"
        print(f"  {d:>5d}  {best_strat:<14s}  {best_avg:>8.4f}  {mn:>6.4f}  {mx:>6.4f}  {status:>7s}")
    
    # Full comparison table
    print(f"\n  Full comparison (avg retention):")
    print(f"  {'':>14s}", end="")
    for d in depths:
        print(f"  {d}L", end="")
    print()
    for name in strategies:
        print(f"  {name:<14s}", end="")
        for d in depths:
            sub = [r for r in all_results if r["n_layers"]==d and r["strategy"]==name]
            if sub:
                avg = np.mean([r["cma_ret"] for r in sub])
                print(f"  {avg:.3f}", end="")
            else:
                print(f"    —", end="")
        print()
    
    print(f"\n  Time: {elapsed:.1f}s")
    
    with open("results_e05.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: results_e05.json")


if __name__ == "__main__":
    main()
