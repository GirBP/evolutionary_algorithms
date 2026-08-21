# Ex01-03_CMA_Boundary: перевірка відтворюваності W Кендала (§2.2 дисертації).
# Читає лише збережені дані з Ex01/data/ та Ex03/data/ (нічого не змінює), рахує
# тест Фрідмана та ефект-розмір W Кендала за тією самою формулою, що й у
# оригінальних ex01_visualize.py / ex03_visualize.py: W = chi2 / (N * (k - 1)).
# Не залежить від пакета common/ чи PyTorch — лише numpy/pandas/scipy.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
EX01_DATA = ROOT / "Ex01" / "data" / "ex01_data_n20.json"
EX03_DATA = ROOT / "Ex03" / "data" / "ex03_data_n5.json"

# Дисертаційні значення (work_paper/dissertation_full.md, §2.2, рядки 776, 838)
DISSERTATION_W_LOW_DIM = 0.943   # Ex01: Rastrigin d=10, CMA-ES vs градієнтні (MR)
DISSERTATION_W_HIGH_DIM = 0.700  # Ex03: зведений тест по 3 датасетах (ваги НМ)


def kendall_w_from_friedman(chi2: float, n_blocks: int, k_methods: int) -> float:
    """W = chi2 / (N * (k - 1)); N — кількість блоків, k — кількість методів."""
    denom = n_blocks * (k_methods - 1)
    return chi2 / denom if denom > 0 else float("nan")


def ex01_kendall_w() -> dict[str, Any]:
    """Ex01: Rastrigin (d=10), блок = Trial, метрика — Final Loss (менше = краще)."""
    data = json.loads(EX01_DATA.read_text(encoding="utf-8"))
    df = pd.DataFrame(data["final"])
    methods = sorted(df["Method"].unique())
    pivot = df.pivot_table(index="Trial", columns="Method", values="Final Loss")
    pivot = pivot[methods].dropna()
    chi2, p = stats.friedmanchisquare(*[pivot[m].values for m in methods])
    n_blocks, k = len(pivot), len(methods)
    w = kendall_w_from_friedman(chi2, n_blocks, k)
    return {
        "experiment": "Ex01 (Rastrigin, d=10)",
        "dim": data["metadata"]["DIM"],
        "methods": methods,
        "n_blocks": n_blocks,
        "k_methods": k,
        "chi2": chi2,
        "p": p,
        "kendall_w": w,
    }


def ex03_kendall_w_aggregate() -> dict[str, Any]:
    """Ex03: усі 3 датасети разом, блок = (Dataset, Run), метрика — F1-Score."""
    data = json.loads(EX03_DATA.read_text(encoding="utf-8"))
    df = pd.DataFrame(data["final"])
    methods = sorted(df["Method"].unique())
    pivot = df.pivot_table(index=["Dataset", "Run"], columns="Method", values="F1-Score")
    pivot = pivot[methods].dropna(how="all")
    chi2, p = stats.friedmanchisquare(*[pivot[m].values for m in methods])
    n_blocks, k = len(pivot), len(methods)
    w = kendall_w_from_friedman(chi2, n_blocks, k)
    return {
        "experiment": "Ex03 (ваги НМ, 3 датасети, зведено)",
        "datasets": sorted(df["Dataset"].unique().tolist()),
        "methods": methods,
        "n_blocks": n_blocks,
        "k_methods": k,
        "chi2": chi2,
        "p": p,
        "kendall_w": w,
    }


# Розмірність простору ваг MLP для кожного датасету Ex03 (див. Ex03/code/ex03.py:
# build_model/MLPBinary/MLPMulti; n_features визначено датасетом, hidden — архітектурою).
EX03_WEIGHT_DIM = {
    # MLPBinary(in, hidden=32): Linear(in,32) + Linear(32,1)
    "moons": 2 * 32 + 32 + 32 * 1 + 1,                # n_features=2  -> 129
    "classification20": 20 * 32 + 32 + 32 * 1 + 1,    # n_features=20 -> 705
    # MLPMulti(in, n_classes=10, hidden=64): Linear(in,64) + Linear(64,10)
    "digits": 64 * 64 + 64 + 64 * 10 + 10,            # n_features=64 -> 4810
}


def ex03_kendall_w_per_dataset() -> list[dict[str, Any]]:
    """Ex03: W Кендала окремо для кожного датасету (блок = Run, N=5), для
    графіка W vs розмірність простору ваг (реальні проміжні точки з тих самих
    даних, що й зведений показник)."""
    data = json.loads(EX03_DATA.read_text(encoding="utf-8"))
    df = pd.DataFrame(data["final"])
    methods = sorted(df["Method"].unique())
    out = []
    for ds in sorted(df["Dataset"].unique()):
        sub = df[df["Dataset"] == ds]
        pivot = sub.pivot_table(index="Run", columns="Method", values="F1-Score")
        pivot = pivot[methods].dropna()
        chi2, p = stats.friedmanchisquare(*[pivot[m].values for m in methods])
        n_blocks, k = len(pivot), len(methods)
        w = kendall_w_from_friedman(chi2, n_blocks, k)
        out.append({
            "dataset": ds,
            "weight_dim": EX03_WEIGHT_DIM[ds],
            "n_blocks": n_blocks,
            "k_methods": k,
            "chi2": chi2,
            "p": p,
            "kendall_w": w,
        })
    return out


def main() -> None:
    print("Ex01-03_CMA_Boundary — перевірка відтворюваності W Кендала (§2.2)\n")

    r1 = ex01_kendall_w()
    diff1 = r1["kendall_w"] - DISSERTATION_W_LOW_DIM
    print(f"[Ex01] {r1['experiment']}: методи={r1['methods']}")
    print(f"       chi2={r1['chi2']:.4f}, p={r1['p']:.6f}, N={r1['n_blocks']}, k={r1['k_methods']}")
    print(f"       W Кендала (перераховано) = {r1['kendall_w']:.4f}")
    print(f"       W Кендала (дисертація)   = {DISSERTATION_W_LOW_DIM:.4f}")
    print(f"       різниця = {diff1:+.4f}  {'OK' if abs(diff1) < 5e-4 else 'РОЗБІЖНІСТЬ'}\n")

    r3 = ex03_kendall_w_aggregate()
    diff3 = r3["kendall_w"] - DISSERTATION_W_HIGH_DIM
    print(f"[Ex03] {r3['experiment']}: датасети={r3['datasets']}, методи={r3['methods']}")
    print(f"       chi2={r3['chi2']:.4f}, p={r3['p']:.6f}, N={r3['n_blocks']}, k={r3['k_methods']}")
    print(f"       W Кендала (перераховано) = {r3['kendall_w']:.4f}")
    print(f"       W Кендала (дисертація)   = {DISSERTATION_W_HIGH_DIM:.4f}")
    print(f"       різниця = {diff3:+.4f}  {'OK' if abs(diff3) < 5e-4 else 'РОЗБІЖНІСТЬ'}\n")

    print("[Ex03] W Кендала по окремих датасетах (реальні дані, для графіка):")
    for row in ex03_kendall_w_per_dataset():
        print(f"       {row['dataset']:<18} dim(ваги)={row['weight_dim']:>5}  "
              f"N={row['n_blocks']}  W={row['kendall_w']:.4f}  p={row['p']:.4f}")


if __name__ == "__main__":
    main()
