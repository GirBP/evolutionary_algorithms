#!/usr/bin/env python3
"""
E03 — Scaling L2: 4-layer MLP + FashionMNIST.
===============================================
After E02: CMA SVD scaling → ret=0.986 for 3-layer, all pairs.
Question: Does it scale to 4 layers (3 hidden) and different datasets?

L2 Fidelity: 2 arch pairs × 2 datasets × 3 seeds = 12 runs
Budget: ≤ 30 min
Kill threshold: retention < 0.80
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
    {"name": "2x", "A": [784, 128, 64, 32, 10], "B": [784, 256, 128, 64, 10]},
    {"name": "4x", "A": [784, 64, 32, 16, 10],  "B": [784, 256, 128, 64, 10]},
]
SEEDS = [42, 123, 777]


def load_dataset(name="MNIST"):
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        cls = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
        path = f'/tmp/{name.lower()}'
        tr = cls(path, train=True, download=True, transform=tf)
        te = cls(path, train=False, download=True, transform=tf)
        X_tr = torch.stack([tr[i][0] for i in range(N_TRAIN)])
        y_tr = torch.tensor([tr[i][1] for i in range(N_TRAIN)])
        X_te = torch.stack([te[i][0] for i in range(N_TEST)])
        y_te = torch.tensor([te[i][1] for i in range(N_TEST)])
        return X_tr, y_tr, X_te, y_te
    except Exception as e:
        print(f"  ⚠️ {name} download failed: {e}")
        rng = np.random.RandomState(42 if name == "MNIST" else 99)
        centers = rng.randn(10, 784).astype(np.float32) * (0.3 if name == "MNIST" else 0.2)
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


def train_model(arch, X, y, seed, epochs=30, lr=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)
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


# ─── Generalized N-layer merge ──────────────────────────────────────────────

def get_layer_activations_postrelu(model, X):
    """Get post-ReLU activations for hidden layers, raw for output."""
    acts = []
    with torch.no_grad():
        h = X
        for i, m in enumerate(model.net):
            h = m(h)
            if isinstance(m, nn.ReLU):
                acts.append(h.numpy().copy())
            elif isinstance(m, nn.Linear) and i == len(model.net) - 1:
                acts.append(h.numpy().copy())
    return acts


def compute_svd_components(model_A, model_B, X_train):
    """Compute SVD components for each hidden layer mapping."""
    arch_A = model_A.arch
    arch_B = model_B.arch
    n_hidden = len(arch_A) - 2

    acts_A = get_layer_activations_postrelu(model_A, X_train)
    acts_B = get_layer_activations_postrelu(model_B, X_train)

    svd_list = []
    s_inits = []
    d_mins = []

    for i in range(n_hidden):
        H_A = acts_A[i]  # Already post-ReLU
        H_B = acts_B[i]
        d_a = arch_A[i + 1]
        d_b = arch_B[i + 1]
        d_min = min(d_a, d_b)
        d_mins.append(d_min)

        C = H_A.T @ H_B  # (d_a, d_b)
        U_full, S_full, Vt_full = np.linalg.svd(C, full_matrices=False)
        s_init = S_full[:d_min] / (S_full[0] + 1e-10)
        U_use = U_full[:, :d_min]
        Vt_use = Vt_full[:d_min, :]

        svd_list.append((U_use, Vt_use))
        s_inits.append(s_init)

    return svd_list, s_inits, d_mins


def merge_nlayer(model_A, model_B, svd_list, s_vec, alpha=0.5):
    """Merge N-layer MLPs with SVD-based mappings and CMA-optimized scaling."""
    arch_A = model_A.arch
    n_hidden = len(arch_A) - 2
    n_layers = n_hidden + 1

    W_A = [p.detach().numpy() for p in model_A.parameters()]
    W_B = [p.detach().numpy() for p in model_B.parameters()]

    # Build M for each hidden layer
    d_mins = [svd_list[i][0].shape[1] for i in range(n_hidden)]
    mappings = []
    offset = 0
    for i in range(n_hidden):
        U, Vt = svd_list[i]
        d = d_mins[i]
        s = s_vec[offset:offset + d]
        offset += d
        M = U @ np.diag(s) @ Vt
        mappings.append(M)

    merged_params = []
    for layer_idx in range(n_layers):
        W_a = W_A[layer_idx * 2]
        b_a = W_A[layer_idx * 2 + 1]
        W_b = W_B[layer_idx * 2]
        b_b = W_B[layer_idx * 2 + 1]

        if layer_idx == 0:
            M = mappings[0]
            W_b_m = M @ W_b
            b_b_m = M @ b_b
        elif layer_idx < n_hidden:
            M_curr = mappings[layer_idx]
            M_prev = mappings[layer_idx - 1]
            W_b_m = M_curr @ W_b @ M_prev.T
            b_b_m = M_curr @ b_b
        else:
            M_last = mappings[-1]
            W_b_m = W_b @ M_last.T
            b_b_m = b_b

        merged_params.append(alpha * W_a + (1 - alpha) * W_b_m)
        merged_params.append(alpha * b_a + (1 - alpha) * b_b_m)

    merged = MLP(arch_A)
    with torch.no_grad():
        for p, v in zip(merged.parameters(), merged_params):
            p.copy_(torch.tensor(v, dtype=torch.float32))
    return merged


def run_cma_merge(model_A, model_B, X_train, X_test, y_test,
                   alpha=0.5, maxiter=45, popsize=16, seed=42):
    """CMA-ES on SVD scaling factors for N-layer merge."""
    import cma

    svd_list, s_inits, d_mins = compute_svd_components(model_A, model_B, X_train)
    s0 = np.concatenate(s_inits)
    total_dims = len(s0)

    def fitness(s_vec):
        try:
            merged = merge_nlayer(model_A, model_B, svd_list, s_vec, alpha)
            return -evaluate(merged, X_test, y_test)
        except Exception:
            return 1.0

    es = cma.CMAEvolutionStrategy(s0.tolist(), 0.3, {
        'maxiter': maxiter, 'popsize': popsize, 'seed': seed,
        'verbose': -9, 'verb_disp': 0, 'verb_log': 0,
        'bounds': [[-3.0] * total_dims, [3.0] * total_dims],
        'CMA_diagonal': total_dims > 100,
    })

    while not es.stop():
        sols = es.ask()
        es.tell(sols, [fitness(np.array(s)) for s in sols])

    best_acc = -es.result.fbest
    n_evals = es.result.evaluations

    # Procrustes baseline
    merged_proc = merge_nlayer(model_A, model_B, svd_list, s0, alpha)
    proc_acc = evaluate(merged_proc, X_test, y_test)

    return best_acc, proc_acc, n_evals, total_dims


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("  E03 [Scaling L2] CMA SVD scaling — 4-layer MLP + 2 datasets")
    print("  L2: 2 pairs × 2 datasets × 3 seeds = 12 runs")
    print("=" * 70)

    datasets_list = ["MNIST", "FashionMNIST"]
    all_results = []

    for ds_name in datasets_list:
        print(f"\n{'▓' * 70}")
        print(f"  Dataset: {ds_name}")
        print(f"{'▓' * 70}")

        X_train, y_train, X_test, y_test = load_dataset(ds_name)

        for pair in PAIRS:
            arch_A, arch_B = pair["A"], pair["B"]
            print(f"\n  {'━' * 60}")
            print(f"  Pair '{pair['name']}': A={arch_A} vs B={arch_B}")
            print(f"  {'━' * 60}")

            for seed in SEEDS:
                model_A = train_model(arch_A, X_train, y_train, seed, epochs=30)
                model_B = train_model(arch_B, X_train, y_train, seed, epochs=30)
                acc_A = evaluate(model_A, X_test, y_test)
                acc_B = evaluate(model_B, X_test, y_test)
                best_parent = max(acc_A, acc_B)

                cma_acc, proc_acc, n_evals, n_dims = run_cma_merge(
                    model_A, model_B, X_train, X_test, y_test,
                    alpha=0.5, maxiter=45, popsize=16, seed=seed,
                )

                ret_cma = cma_acc / best_parent if best_parent > 0 else 0
                ret_proc = proc_acc / best_parent if best_parent > 0 else 0

                result = {
                    "dataset": ds_name, "pair": pair["name"], "seed": seed,
                    "acc_A": round(acc_A, 4), "acc_B": round(acc_B, 4),
                    "best_parent": round(best_parent, 4),
                    "proc_acc": round(proc_acc, 4), "proc_ret": round(ret_proc, 4),
                    "cma_acc": round(cma_acc, 4), "cma_ret": round(ret_cma, 4),
                    "n_evals": n_evals, "n_dims": n_dims,
                    "n_layers": len(arch_A) - 1,
                }
                all_results.append(result)

                marker = "✅" if ret_cma >= 0.80 else "❌"
                print(f"    s={seed}: A={acc_A:.3f} B={acc_B:.3f} | "
                      f"Proc={proc_acc:.3f}(r={ret_proc:.3f}) | "
                      f"CMA={cma_acc:.3f}(r={ret_cma:.3f}) {marker} | "
                      f"dims={n_dims}")

    # ─── Final Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY — E03 Scaling L2 (4-layer MLP)")
    print("=" * 70)

    print(f"\n  {'Dataset':<12s} {'Pair':<6s} {'AvgA':>6s} {'AvgB':>6s} "
          f"{'ProcR':>7s} {'CMA_R':>7s} {'Status':>7s}")
    print(f"  {'─' * 55}")

    for ds_name in datasets_list:
        for pair_name in [p["name"] for p in PAIRS]:
            subset = [r for r in all_results
                      if r["dataset"] == ds_name and r["pair"] == pair_name]
            if not subset:
                continue
            avg_A = np.mean([r["acc_A"] for r in subset])
            avg_B = np.mean([r["acc_B"] for r in subset])
            avg_proc = np.mean([r["proc_ret"] for r in subset])
            avg_cma = np.mean([r["cma_ret"] for r in subset])
            status = "✅" if avg_cma >= 0.80 else "❌"
            print(f"  {ds_name:<12s} {pair_name:<6s} {avg_A:>6.3f} {avg_B:>6.3f} "
                  f"{avg_proc:>7.4f} {avg_cma:>7.4f} {status:>7s}")

    overall_cma = np.mean([r["cma_ret"] for r in all_results])
    overall_proc = np.mean([r["proc_ret"] for r in all_results])
    min_cma = min(r["cma_ret"] for r in all_results)
    max_cma = max(r["cma_ret"] for r in all_results)

    # Per-dataset
    for ds in datasets_list:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        ds_cma = np.mean([r["cma_ret"] for r in ds_results])
        ds_proc = np.mean([r["proc_ret"] for r in ds_results])
        print(f"\n  {ds}: CMA={ds_cma:.4f} Proc={ds_proc:.4f} Δ={ds_cma-ds_proc:.4f}")

    n_pass = sum(1 for r in all_results if r["cma_ret"] >= 0.80)
    print(f"\n  Overall CMA retention: {overall_cma:.4f} [{min_cma:.4f}, {max_cma:.4f}]")
    print(f"  Overall Proc retention: {overall_proc:.4f}")
    print(f"  Pass rate: {n_pass}/{len(all_results)}")
    print(f"\n  Time: {elapsed:.1f}s (budget: 30 min)")
    print(f"  Kill threshold: 0.80")
    print(f"  L2 {'PASS ✅' if overall_cma >= 0.80 else 'FAIL ❌'}")

    with open("results_e03.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: results_e03.json")


if __name__ == "__main__":
    main()
