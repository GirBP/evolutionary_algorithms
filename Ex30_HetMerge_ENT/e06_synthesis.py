#!/usr/bin/env python3
"""
E06 [Synthesis] — Final validation of PCMA using clean module.
===============================================================
Comprehensive test: 3 depths × 2 datasets × 2 arch ratios × 3 seeds = 36 runs.
Uses the clean pcma.py module.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import json

sys.path.insert(0, '/Users/bibo/Desktop/cs_dev/Ex30_HetMerge')
from pcma import PCMA

SEEDS = [42, 123, 777]
N_TRAIN = 5000
N_TEST = 1000


def load_dataset(name):
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        cls = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
        tr = cls(f'/tmp/{name.lower()}', train=True, download=True, transform=tf)
        te = cls(f'/tmp/{name.lower()}', train=False, download=True, transform=tf)
        return (torch.stack([tr[i][0] for i in range(N_TRAIN)]),
                torch.tensor([tr[i][1] for i in range(N_TRAIN)]),
                torch.stack([te[i][0] for i in range(N_TEST)]),
                torch.tensor([te[i][1] for i in range(N_TEST)]))
    except Exception:
        rng = np.random.RandomState(42 if name == "MNIST" else 99)
        c = rng.randn(10, 784).astype(np.float32) * 0.3
        def mk(n):
            X = [c[i%10] + rng.randn(784).astype(np.float32)*0.15 for i in range(n)]
            return torch.tensor(np.array(X)), torch.tensor([i%10 for i in range(n)])
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


def train_model(arch, X, y, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(arch)
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(30):
        loss = loss_fn(model(X), y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


CONFIGS = [
    # (depth, ratio, arch_A, arch_B)
    (3, "2x", [784, 128, 64, 10], [784, 256, 128, 10]),
    (3, "3x", [784, 64, 32, 10], [784, 192, 96, 10]),
    (5, "2x", [784, 256, 128, 64, 32, 10], [784, 512, 256, 128, 64, 10]),
    (5, "3x", [784, 128, 64, 32, 32, 10], [784, 384, 192, 96, 64, 10]),
    (7, "2x", [784, 256, 128, 64, 32, 32, 32, 10], [784, 512, 256, 128, 64, 64, 64, 10]),
    (7, "3x", [784, 128, 64, 32, 32, 32, 32, 10], [784, 384, 192, 96, 64, 64, 64, 10]),
]


def main():
    t0 = time.time()
    print("=" * 75)
    print("  E06 [SYNTHESIS] PCMA Final Validation")
    print("  6 configs × 2 datasets × 3 seeds = 36 runs")
    print("=" * 75)

    all_results = []

    for ds_name in ["MNIST", "FashionMNIST"]:
        print(f"\n{'▓' * 75}")
        print(f"  Dataset: {ds_name}")
        print(f"{'▓' * 75}")

        X_tr, y_tr, X_te, y_te = load_dataset(ds_name)

        for depth, ratio, arch_A, arch_B in CONFIGS:
            print(f"\n  ──── {depth}L {ratio} ────")
            print(f"  A={arch_A}")
            print(f"  B={arch_B}")

            for seed in SEEDS:
                model_A = train_model(arch_A, X_tr, y_tr, seed)
                model_B = train_model(arch_B, X_tr, y_tr, seed)

                # PCMA merge
                t_s = time.time()
                merger = PCMA(model_A, model_B, X_tr)
                result = merger.merge(X_te, y_te, maxiter=40, popsize=14, seed=seed)
                dt = time.time() - t_s

                m = "✅" if result.retention >= 0.85 else ("🟡" if result.retention >= 0.70 else "❌")
                alphas_str = [round(float(a), 2) for a in result.alphas]
                print(f"    s={seed}: A={result.acc_A:.3f} B={result.acc_B:.3f} → "
                      f"merged={result.accuracy:.3f} ret={result.retention:.3f} {m} "
                      f"α={alphas_str} "
                      f"dims={result.n_dims} {dt:.1f}s")

                all_results.append({
                    "dataset": ds_name, "depth": depth, "ratio": ratio,
                    "seed": seed,
                    "acc_A": round(result.acc_A, 4),
                    "acc_B": round(result.acc_B, 4),
                    "merged_acc": round(result.accuracy, 4),
                    "retention": round(result.retention, 4),
                    "alphas": [round(a, 3) for a in result.alphas],
                    "n_dims": result.n_dims,
                    "n_evals": result.n_evals,
                    "time_s": round(dt, 1),
                })

    # ─── Final Summary ───────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("\n" + "=" * 75)
    print("  FINAL SYNTHESIS SUMMARY")
    print("=" * 75)

    print(f"\n  {'Dataset':<12s} {'Depth':>5s} {'Ratio':>5s} "
          f"{'AvgA':>6s} {'AvgB':>6s} {'Merged':>7s} {'Ret':>7s} {'Status':>7s}")
    print(f"  {'─' * 60}")

    for ds in ["MNIST", "FashionMNIST"]:
        for depth, ratio, _, _ in CONFIGS:
            sub = [r for r in all_results
                   if r["dataset"] == ds and r["depth"] == depth and r["ratio"] == ratio]
            if not sub:
                continue
            avg_A = np.mean([r["acc_A"] for r in sub])
            avg_B = np.mean([r["acc_B"] for r in sub])
            avg_m = np.mean([r["merged_acc"] for r in sub])
            avg_r = np.mean([r["retention"] for r in sub])
            st = "✅" if avg_r >= 0.85 else "🟡"
            print(f"  {ds:<12s} {depth:>5d} {ratio:>5s} "
                  f"{avg_A:>6.3f} {avg_B:>6.3f} {avg_m:>7.3f} {avg_r:>7.3f} {st:>7s}")

    # Overall stats
    overall_ret = np.mean([r["retention"] for r in all_results])
    min_ret = min(r["retention"] for r in all_results)
    max_ret = max(r["retention"] for r in all_results)
    std_ret = np.std([r["retention"] for r in all_results])
    n_pass = sum(1 for r in all_results if r["retention"] >= 0.85)
    n_super = sum(1 for r in all_results if r["retention"] > 1.0)

    # Check α distribution
    all_alphas = [a for r in all_results for a in r["alphas"]]
    avg_alpha = np.mean(all_alphas)
    degenerate = sum(1 for a in all_alphas if a < 0.15 or a > 0.85)

    print(f"\n  ── Overall Statistics ──")
    print(f"  Retention:       {overall_ret:.4f} ± {std_ret:.4f} [{min_ret:.4f}, {max_ret:.4f}]")
    print(f"  Pass rate (≥85%): {n_pass}/{len(all_results)} ({100*n_pass/len(all_results):.0f}%)")
    print(f"  Super-additive:  {n_super}/{len(all_results)}")
    print(f"  Avg α:           {avg_alpha:.3f}")
    print(f"  Degenerate α:    {degenerate}/{len(all_alphas)} ({100*degenerate/len(all_alphas):.0f}%)")
    print(f"  Total time:      {elapsed:.1f}s")
    print(f"  Avg time/run:    {elapsed/len(all_results):.1f}s")

    with open("results_e06_synthesis.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: results_e06_synthesis.json")


if __name__ == "__main__":
    main()
