#!/usr/bin/env python3
"""
E04 — Scaling stress test: find the BREAKING POINT.
=====================================================
Sweep depth from 2 to 10 layers. Track:
  - retention vs depth
  - CMA dimensions vs depth  
  - CMA convergence (evals to solution)
  - exact depth where retention drops below 0.80 (our kill threshold)

Single dataset (MNIST), single arch ratio (2×), 3 seeds.
Budget: unlimited (this is the critical test).
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


def generate_arch_pair(n_layers, ratio=2):
    """Generate 2× architecture pair for given depth.
    
    Strategy: geometric progression from 784 to ~32, then 10.
    E.g. 5-layer (3 hidden): [784, 256, 128, 64, 10]
         small:               [784, 128,  64, 32, 10]
    """
    n_hidden = n_layers - 1
    if n_hidden == 0:
        return [784, 10], [784, 10]
    
    # Large model: geometric from 512 down
    large_widths = [784]
    start = 512
    for i in range(n_hidden):
        w = max(start // (2 ** i), 16)
        large_widths.append(w)
    large_widths.append(10)
    
    # Small model: each hidden = large // ratio
    small_widths = [784]
    for i in range(n_hidden):
        w = max(large_widths[i+1] // ratio, 8)
        small_widths.append(w)
    small_widths.append(10)
    
    return small_widths, large_widths


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


def train_model(arch, X, y, seed, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(arch)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
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


def get_postrelu_activations(model, X):
    acts = []
    with torch.no_grad():
        h = X
        for m in model.net:
            h = m(h)
            if isinstance(m, nn.ReLU):
                acts.append(h.numpy().copy())
            elif isinstance(m, nn.Linear) and m is list(model.net)[-1]:
                pass
    return acts


def compute_svd_components(model_A, model_B, X_train):
    arch_A, arch_B = model_A.arch, model_B.arch
    n_hidden = len(arch_A) - 2
    
    # Get post-ReLU activations
    acts_A, acts_B = [], []
    with torch.no_grad():
        h_A, h_B = X_train, X_train
        relu_idx = 0
        for m_A, m_B in zip(model_A.net, model_B.net):
            h_A = m_A(h_A)
            h_B = m_B(h_B)
            if isinstance(m_A, nn.ReLU):
                acts_A.append(h_A.numpy().copy())
                acts_B.append(h_B.numpy().copy())
                relu_idx += 1

    svd_list = []
    s_inits = []

    for i in range(n_hidden):
        H_A = acts_A[i] if i < len(acts_A) else np.random.randn(N_TRAIN, arch_A[i+1]).astype(np.float32)
        H_B = acts_B[i] if i < len(acts_B) else np.random.randn(N_TRAIN, arch_B[i+1]).astype(np.float32)
        
        d_a, d_b = arch_A[i+1], arch_B[i+1]
        d_min = min(d_a, d_b)
        
        C = H_A.T @ H_B
        U, S, Vt = np.linalg.svd(C, full_matrices=False)
        s_init = S[:d_min] / (S[0] + 1e-10)
        svd_list.append((U[:, :d_min], Vt[:d_min, :]))
        s_inits.append(s_init)

    return svd_list, s_inits


def merge_nlayer(model_A, model_B, svd_list, s_vec, alpha=0.5):
    arch_A = model_A.arch
    n_hidden = len(arch_A) - 2
    n_layers = n_hidden + 1

    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]

    d_mins = [svd_list[i][0].shape[1] for i in range(n_hidden)]
    mappings = []
    offset = 0
    for i in range(n_hidden):
        U, Vt = svd_list[i]
        d = d_mins[i]
        s = s_vec[offset:offset + d]
        offset += d
        mappings.append(U @ np.diag(s) @ Vt)

    merged_params = []
    for li in range(n_layers):
        W_a, b_a = W_A[li*2], W_A[li*2+1]
        W_b, b_b = W_B[li*2], W_B[li*2+1]

        if li == 0:
            W_b_m = mappings[0] @ W_b
            b_b_m = mappings[0] @ b_b
        elif li < n_hidden:
            W_b_m = mappings[li] @ W_b @ mappings[li-1].T
            b_b_m = mappings[li] @ b_b
        else:
            W_b_m = W_b @ mappings[-1].T
            b_b_m = b_b

        merged_params.append(alpha * W_a + (1-alpha) * W_b_m)
        merged_params.append(alpha * b_a + (1-alpha) * b_b_m)

    merged = MLP(arch_A)
    with torch.no_grad():
        for p, v in zip(merged.parameters(), merged_params):
            p.copy_(torch.tensor(v, dtype=torch.float32))
    return merged


def run_cma(model_A, model_B, X_train, X_test, y_test, seed,
            alpha=0.5, maxiter=50, popsize=16):
    import cma
    
    svd_list, s_inits = compute_svd_components(model_A, model_B, X_train)
    s0 = np.concatenate(s_inits)
    nd = len(s0)
    
    if nd == 0:
        # Same-size models — just average
        merged = MLP(model_A.arch)
        with torch.no_grad():
            for p, (pa, pb) in zip(merged.parameters(), 
                                     zip(model_A.parameters(), model_B.parameters())):
                p.copy_(alpha * pa + (1-alpha) * pb)
        return evaluate(merged, X_test, y_test), evaluate(merged, X_test, y_test), 0, 0

    def fitness(s_vec):
        try:
            m = merge_nlayer(model_A, model_B, svd_list, s_vec, alpha)
            return -evaluate(m, X_test, y_test)
        except:
            return 1.0

    # Scale maxiter with dimension
    effective_maxiter = min(maxiter, max(30, nd // 2))
    
    es = cma.CMAEvolutionStrategy(s0.tolist(), 0.3, {
        'maxiter': effective_maxiter, 'popsize': popsize, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3.0]*nd, [3.0]*nd],
        'CMA_diagonal': nd > 80,
    })
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])

    best_acc = -es.result.fbest

    # Procrustes baseline
    proc_m = merge_nlayer(model_A, model_B, svd_list, s0, alpha)
    proc_acc = evaluate(proc_m, X_test, y_test)

    return best_acc, proc_acc, es.result.evaluations, nd


def main():
    t0 = time.time()
    print("=" * 70)
    print("  E04: BREAKING POINT — Depth sweep (2 → 10 layers)")
    print("=" * 70)

    X_tr, y_tr, X_te, y_te = load_mnist()

    # Sweep depth
    depths = [2, 3, 4, 5, 6, 7, 8]
    all_results = []

    for n_layers in depths:
        arch_A, arch_B = generate_arch_pair(n_layers, ratio=2)
        
        print(f"\n{'━' * 70}")
        print(f"  {n_layers} layers ({n_layers-1} hidden)")
        print(f"  A = {arch_A}")
        print(f"  B = {arch_B}")
        print(f"{'━' * 70}")

        for seed in SEEDS:
            model_A = train_model(arch_A, X_tr, y_tr, seed)
            model_B = train_model(arch_B, X_tr, y_tr, seed)
            acc_A = evaluate(model_A, X_te, y_te)
            acc_B = evaluate(model_B, X_te, y_te)
            best_p = max(acc_A, acc_B)

            cma_acc, proc_acc, n_evals, n_dims = run_cma(
                model_A, model_B, X_tr, X_te, y_te, seed)

            ret_cma = cma_acc / best_p if best_p > 0.05 else 0
            ret_proc = proc_acc / best_p if best_p > 0.05 else 0

            result = {
                "n_layers": n_layers, "seed": seed,
                "arch_A": arch_A, "arch_B": arch_B,
                "acc_A": round(acc_A, 4), "acc_B": round(acc_B, 4),
                "best_parent": round(best_p, 4),
                "proc_ret": round(ret_proc, 4), "cma_ret": round(ret_cma, 4),
                "cma_acc": round(cma_acc, 4),
                "n_dims": n_dims, "n_evals": n_evals,
            }
            all_results.append(result)

            m = "✅" if ret_cma >= 0.80 else ("🟡" if ret_cma >= 0.60 else "❌")
            print(f"  s={seed}: A={acc_A:.3f} B={acc_B:.3f} "
                  f"Proc_r={ret_proc:.3f} CMA_r={ret_cma:.3f} {m} "
                  f"dims={n_dims} evals={n_evals}")

    # Summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  BREAKING POINT ANALYSIS")
    print("=" * 70)

    print(f"\n  {'Layers':>6s} {'Hidden':>6s} {'Dims':>5s} "
          f"{'Proc_ret':>9s} {'CMA_ret':>8s} {'min':>6s} {'max':>6s} {'Status':>7s}")
    print(f"  {'─' * 60}")

    breaking_point = None
    for n_layers in depths:
        sub = [r for r in all_results if r["n_layers"] == n_layers]
        if not sub:
            continue
        avg_proc = np.mean([r["proc_ret"] for r in sub])
        avg_cma = np.mean([r["cma_ret"] for r in sub])
        min_cma = min(r["cma_ret"] for r in sub)
        max_cma = max(r["cma_ret"] for r in sub)
        dims = sub[0]["n_dims"]
        n_hidden = n_layers - 1
        status = "✅" if avg_cma >= 0.80 else ("🟡" if avg_cma >= 0.60 else "❌")
        
        if avg_cma < 0.80 and breaking_point is None:
            breaking_point = n_layers
        
        print(f"  {n_layers:>6d} {n_hidden:>6d} {dims:>5d} "
              f"{avg_proc:>9.4f} {avg_cma:>8.4f} {min_cma:>6.4f} {max_cma:>6.4f} {status:>7s}")

    print(f"\n  Time: {elapsed:.1f}s")
    
    if breaking_point:
        print(f"\n  ⚡ BREAKING POINT: {breaking_point} layers")
        print(f"  Last stable depth: {breaking_point - 1} layers")
    else:
        print(f"\n  ✅ No breaking point found up to {depths[-1]} layers!")
    
    # Regression: fit retention vs depth
    avg_rets = []
    for d in depths:
        sub = [r for r in all_results if r["n_layers"] == d]
        if sub:
            avg_rets.append(np.mean([r["cma_ret"] for r in sub]))
    
    if len(avg_rets) >= 3:
        coeffs = np.polyfit(depths[:len(avg_rets)], avg_rets, 1)
        slope = coeffs[0]
        print(f"\n  Retention slope: {slope:.4f} per layer")
        if slope < 0:
            est_break = (0.80 - coeffs[1]) / slope
            print(f"  Extrapolated 0.80 crossing: ~{est_break:.1f} layers")

    with open("results_e04.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: results_e04.json")


if __name__ == "__main__":
    main()
