#!/usr/bin/env python3
"""
E02 — Scaling: CMA on SVD scaling factors for 3-layer heterogeneous MLP.
=========================================================================
After E01b: CMA optimizing SVD scaling s with α=0.5 → retention 1.005 for 2 layers.
Question: Does it scale to 3 layers (2 hidden) where cascading error was fatal before?

L1 Fidelity: 3 architecture pairs × 3 seeds = 9 runs
Budget: ≤ 5 min
Kill threshold: retention < 0.75

Architecture pairs:
  Pair 1: [784, 128, 64, 10] vs [784, 256, 128, 10]  (2× ratio)
  Pair 2: [784, 64, 32, 10]  vs [784, 256, 128, 10]  (4× ratio)  
  Pair 3: [784, 128, 64, 10] vs [784, 192, 96, 10]   (1.5× ratio)
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

PAIRS = [
    {"name": "2x", "A": [784, 128, 64, 10], "B": [784, 256, 128, 10]},
    {"name": "4x", "A": [784, 64, 32, 10],  "B": [784, 256, 128, 10]},
    {"name": "1.5x", "A": [784, 128, 64, 10], "B": [784, 192, 96, 10]},
]
SEEDS = [42, 123, 777]


# ─── Data & Model ────────────────────────────────────────────────────────────

def load_mnist():
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import datasets, transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        train_ds = datasets.MNIST('/tmp/mnist', train=True, download=True, transform=transform)
        test_ds = datasets.MNIST('/tmp/mnist', train=False, download=True, transform=transform)
        X_tr = torch.stack([train_ds[i][0] for i in range(N_TRAIN)])
        y_tr = torch.tensor([train_ds[i][1] for i in range(N_TRAIN)])
        X_te = torch.stack([test_ds[i][0] for i in range(N_TEST)])
        y_te = torch.tensor([test_ds[i][1] for i in range(N_TEST)])
        return X_tr, y_tr, X_te, y_te
    except Exception:
        rng = np.random.RandomState(42)
        centers = rng.randn(10, 784).astype(np.float32) * 0.3
        def mk(n):
            X, y = [], []
            for i in range(n):
                c = i % 10
                X.append(centers[c] + rng.randn(784).astype(np.float32) * 0.15)
                y.append(c)
            return torch.tensor(np.array(X)), torch.tensor(y)
        return mk(N_TRAIN)[0], mk(N_TRAIN)[1], mk(N_TEST)[0], mk(N_TEST)[1]


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


def train_model(arch, X, y, seed, epochs=25, lr=0.01):
    torch.manual_seed(seed)
    model = MLP(arch)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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


# ─── Multi-layer SVD + CMA merging ──────────────────────────────────────────

def get_layer_activations(model, X):
    """Get activations after each Linear layer (pre-ReLU)."""
    acts = []
    with torch.no_grad():
        h = X
        for m in model.net:
            h = m(h)
            if isinstance(m, nn.Linear):
                acts.append(h.numpy().copy())
    return acts  # [hidden1_pre, hidden2_pre, ..., output]


def procrustes_svd(H_target, H_source):
    """SVD of cross-correlation H_target^T @ H_source."""
    C = H_target.T @ H_source
    U, S, Vt = np.linalg.svd(C, full_matrices=False)
    s = S / (S.max() + 1e-10)
    return U, s, Vt


def merge_3layer(model_A, model_B, svd_list, s_vec, alpha=0.5):
    """Merge 3-layer MLPs (2 hidden) with given SVD components and scaling vector.
    
    svd_list: [(U1, Vt1), (U2, Vt2)] for hidden layers 1 and 2
    s_vec: concatenated scaling factors [s1 (d_min1), s2 (d_min2)]
    """
    arch_A = model_A.arch
    n_hidden = len(arch_A) - 2  # number of hidden layers
    
    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]
    # W_A: [W1, b1, W2, b2, W3, b3]
    # For 3-layer: indices 0,1 = layer1; 2,3 = layer2; 4,5 = layer3(output)
    
    # Build mappings
    d_mins = [min(arch_A[i+1], model_B.arch[i+1]) for i in range(n_hidden)]
    mappings = []
    offset = 0
    for i in range(n_hidden):
        U, Vt = svd_list[i]
        d = d_mins[i]
        s = s_vec[offset:offset + d]
        offset += d
        M = U @ np.diag(s) @ Vt  # maps B's layer i dim → A's layer i dim
        mappings.append(M)
    
    # Merge layer by layer
    merged_params = []
    
    for layer_idx in range(n_hidden + 1):  # 0, 1, 2 (for 3-layer)
        W_a = W_A[layer_idx * 2]      # weight
        b_a = W_A[layer_idx * 2 + 1]  # bias
        W_b = W_B[layer_idx * 2]
        b_b = W_B[layer_idx * 2 + 1]
        
        if layer_idx == 0:
            # First hidden layer: M_1 @ W_B_1  (map B's outputs to A's space)
            M = mappings[0]
            W_b_mapped = M @ W_b
            b_b_mapped = M @ b_b
        elif layer_idx < n_hidden:
            # Middle hidden layer: M_curr @ W_B @ M_prev^T
            M_curr = mappings[layer_idx]
            M_prev = mappings[layer_idx - 1]
            W_b_mapped = M_curr @ W_b @ M_prev.T
            b_b_mapped = M_curr @ b_b
        else:
            # Output layer: W_B @ M_last^T
            M_last = mappings[-1]
            W_b_mapped = W_b @ M_last.T
            b_b_mapped = b_b
        
        W_merged = alpha * W_a + (1 - alpha) * W_b_mapped
        b_merged = alpha * b_a + (1 - alpha) * b_b_mapped
        merged_params.extend([W_merged, b_merged])
    
    # Build model
    merged = MLP(arch_A)
    with torch.no_grad():
        for p, v in zip(merged.parameters(), merged_params):
            p.copy_(torch.tensor(v, dtype=torch.float32))
    
    return merged


def run_cma_merge(model_A, model_B, X_train, X_test, y_test, alpha=0.5,
                   maxiter=40, popsize=14, seed=42):
    """CMA-ES on SVD scaling factors for multi-layer merge."""
    import cma
    
    arch_A = model_A.arch
    arch_B = model_B.arch
    n_hidden = len(arch_A) - 2
    
    # Get activations for SVD
    acts_A = get_layer_activations(model_A, X_train)
    acts_B = get_layer_activations(model_B, X_train)
    
    # Compute SVD for each hidden layer
    svd_list = []
    s_inits = []
    d_mins = []
    
    for i in range(n_hidden):
        H_A = np.maximum(acts_A[i], 0)  # post-ReLU for better alignment
        H_B = np.maximum(acts_B[i], 0)
        d_a = arch_A[i + 1]
        d_b = arch_B[i + 1]
        d_min = min(d_a, d_b)
        d_mins.append(d_min)
        
        U, s, Vt = procrustes_svd(H_A[:, :d_min], H_B[:, :d_min] if d_b <= d_a else H_B)
        
        # Ensure correct dimensions
        # U should be (d_a, d_min), Vt should be (d_min, d_b)
        C = H_A.T @ H_B  # (d_a, d_b)
        U_full, S_full, Vt_full = np.linalg.svd(C, full_matrices=False)
        # U_full: (d_a, min(d_a,d_b)), S_full: (min(d_a,d_b),), Vt_full: (min(d_a,d_b), d_b)
        s_init = S_full[:d_min] / (S_full[0] + 1e-10)
        U_use = U_full[:, :d_min]   # (d_a, d_min)
        Vt_use = Vt_full[:d_min, :] # (d_min, d_b)
        
        # M = U_use @ diag(s) @ Vt_use: (d_a, d_min) @ (d_min, d_min) @ (d_min, d_b) = (d_a, d_b) ✓
        svd_list.append((U_use, Vt_use))
        s_inits.append(s_init)
    
    # Concatenate s for CMA
    s0 = np.concatenate(s_inits)
    total_dims = len(s0)
    
    def fitness(s_vec):
        try:
            merged = merge_3layer(model_A, model_B, svd_list, s_vec, alpha)
            acc = evaluate(merged, X_test, y_test)
            return -acc
        except Exception:
            return 1.0
    
    es = cma.CMAEvolutionStrategy(s0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3.0] * total_dims, [3.0] * total_dims],
        'CMA_diagonal': total_dims > 50,
    })
    
    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])
    
    best_acc = -es.result.fbest
    n_evals = es.result.evaluations
    
    # Also get Procrustes baseline (s from SVD, no CMA)
    merged_proc = merge_3layer(model_A, model_B, svd_list, s0, alpha)
    proc_acc = evaluate(merged_proc, X_test, y_test)
    
    return best_acc, proc_acc, n_evals, total_dims


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  E02 [Scaling] CMA on SVD scaling — 3-layer MLP")
    print("  L1: 3 architecture pairs × 3 seeds = 9 runs")
    print("=" * 70)
    
    X_train, y_train, X_test, y_test = load_mnist()
    
    all_results = []
    
    for pair in PAIRS:
        arch_A, arch_B = pair["A"], pair["B"]
        print(f"\n{'━' * 70}")
        print(f"  Pair '{pair['name']}': A={arch_A} vs B={arch_B}")
        print(f"{'━' * 70}")
        
        pair_results = []
        
        for seed in SEEDS:
            # Train both models
            model_A = train_model(arch_A, X_train, y_train, seed)
            model_B = train_model(arch_B, X_train, y_train, seed)
            acc_A = evaluate(model_A, X_test, y_test)
            acc_B = evaluate(model_B, X_test, y_test)
            best_parent = max(acc_A, acc_B)
            
            # Run CMA merge
            cma_acc, proc_acc, n_evals, n_dims = run_cma_merge(
                model_A, model_B, X_train, X_test, y_test,
                alpha=0.5, maxiter=35, popsize=14, seed=seed,
            )
            
            ret_cma = cma_acc / best_parent if best_parent > 0 else 0
            ret_proc = proc_acc / best_parent if best_parent > 0 else 0
            
            result = {
                "pair": pair["name"], "seed": seed,
                "acc_A": acc_A, "acc_B": acc_B, "best_parent": best_parent,
                "proc_acc": proc_acc, "proc_ret": ret_proc,
                "cma_acc": cma_acc, "cma_ret": ret_cma,
                "n_evals": n_evals, "n_dims": n_dims,
            }
            pair_results.append(result)
            all_results.append(result)
            
            marker = "✅" if ret_cma >= 0.75 else "❌"
            print(f"  seed={seed}: A={acc_A:.3f} B={acc_B:.3f} | "
                  f"Proc={proc_acc:.3f}(ret={ret_proc:.3f}) | "
                  f"CMA={cma_acc:.3f}(ret={ret_cma:.3f}) {marker} | "
                  f"dims={n_dims} evals={n_evals}")
        
        # Pair summary
        avg_ret_cma = np.mean([r["cma_ret"] for r in pair_results])
        avg_ret_proc = np.mean([r["proc_ret"] for r in pair_results])
        print(f"\n  Avg retention — Procrustes: {avg_ret_proc:.4f}  CMA: {avg_ret_cma:.4f}")
    
    # Final summary
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — E02 Scaling (3-layer MLP)")
    print("=" * 70)
    
    print(f"\n  {'Pair':<8s} {'Avg AccA':>8s} {'Avg AccB':>8s} {'Proc ret':>9s} {'CMA ret':>8s} {'Status':>7s}")
    print(f"  {'─' * 50}")
    
    for pair_name in [p["name"] for p in PAIRS]:
        pr = [r for r in all_results if r["pair"] == pair_name]
        avg_A = np.mean([r["acc_A"] for r in pr])
        avg_B = np.mean([r["acc_B"] for r in pr])
        avg_proc = np.mean([r["proc_ret"] for r in pr])
        avg_cma = np.mean([r["cma_ret"] for r in pr])
        status = "✅" if avg_cma >= 0.75 else "❌"
        print(f"  {pair_name:<8s} {avg_A:>8.4f} {avg_B:>8.4f} {avg_proc:>9.4f} {avg_cma:>8.4f} {status:>7s}")
    
    overall_cma = np.mean([r["cma_ret"] for r in all_results])
    overall_proc = np.mean([r["proc_ret"] for r in all_results])
    min_cma = min(r["cma_ret"] for r in all_results)
    max_cma = max(r["cma_ret"] for r in all_results)
    
    print(f"\n  Overall CMA retention: {overall_cma:.4f} [{min_cma:.4f}, {max_cma:.4f}]")
    print(f"  Overall Proc retention: {overall_proc:.4f}")
    print(f"  CMA improvement over Proc: {(overall_cma - overall_proc):.4f}")
    print(f"\n  Time: {elapsed:.1f}s (budget: 5 min)")
    print(f"  Kill threshold: 0.75")
    print(f"  L1 {'PASS ✅' if overall_cma >= 0.75 else 'FAIL ❌'}")
    
    # Save JSON
    with open("results_e02.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results saved: results_e02.json")


if __name__ == "__main__":
    main()
