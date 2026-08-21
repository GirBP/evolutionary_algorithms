#!/usr/bin/env python3
# regen_figures.py — Відтворення рисунків 2.10 (CD-діаграма Фрідмана-Немені)
# та 2.11 (F1 vs розрідженість) розділу 2.3 дисертації, 5-методний зріз:
# TESA-26 + Magnitude + WANDA (SOTA) + SparseGPT (SOTA) + RIA (SOTA).
#
# Читає вихідні CSV з results/*_table.csv цієї папки (лише читання)
# і записує PNG у ../figs/. Обчислення не змінені відносно скрипта-джерела.
from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---- Paths -------------------------------------------------------------------
SRC_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "results"
OUT_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: Final[list[str]] = ["blobs", "circles", "cnn", "moons", "resnet", "spirals"]
METHODS: Final[list[str]] = [
    "TESA-26",
    "Magnitude",
    "WANDA (SOTA)",
    "SparseGPT (SOTA)",
    "RIA (SOTA)",
]
SPARSITY_COLS: Final[list[str]] = [
    "50%", "70%", "80%", "85%", "90%", "91%", "92%",
    "93%", "94%", "95%", "96%", "97%",
]
HIGH_SPARSITY: Final[list[str]] = ["90%", "91%", "92%", "93%", "94%", "95%", "96%", "97%"]

# Colors: TESA-26 highlighted, baselines in muted tones
COLORS: Final[dict[str, str]] = {
    "TESA-26": "#C0392B",         # bold red
    "Magnitude": "#7F8C8D",       # grey
    "WANDA (SOTA)": "#2980B9",    # blue
    "SparseGPT (SOTA)": "#27AE60", # green
    "RIA (SOTA)": "#8E44AD",      # purple
}


def load_all() -> pd.DataFrame:
    """Load all 6 CSV tables, melt to long form."""
    frames: list[pd.DataFrame] = []
    for ds in DATASETS:
        fp = SRC_DIR / f"{ds}_table.csv"
        df = pd.read_csv(fp, sep=";", encoding="utf-8-sig")
        # The leading "#" column is the rank index — drop it
        if "#" in df.columns:
            df = df.drop(columns=["#"])
        df = df[df["Метод"].isin(METHODS)].copy()
        # Keep only sparsity columns + method
        keep = ["Метод"] + [c for c in SPARSITY_COLS if c in df.columns]
        df = df[keep]
        long = df.melt(id_vars=["Метод"], var_name="sparsity", value_name="F1")
        long["dataset"] = ds
        # F1 may already be float but if string with decimal — convert
        long["F1"] = pd.to_numeric(long["F1"], errors="coerce")
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    return out


# ---- Friedman + Nemenyi CD ---------------------------------------------------
# Critical values q_{alpha} for Nemenyi (two-tailed-equivalent),
# from Demsar (2006), Table 5. k = number of methods, alpha=0.05.
NEMENYI_Q_005: Final[dict[int, float]] = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
    7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
}


def compute_ranks(df_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, int]:
    """Pivot to (dataset×sparsity) × method matrix and average ranks (higher F1 = rank 1)."""
    pivot = df_long.pivot_table(
        index=["dataset", "sparsity"],
        columns="Метод",
        values="F1",
        aggfunc="mean",
    )
    pivot = pivot[METHODS]  # enforce method order
    pivot = pivot.dropna(axis=0, how="any")  # need all 5 methods present
    # Rank: higher F1 -> better -> rank 1 -> use negative for ascending
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    mean_ranks = ranks.mean(axis=0)
    n_pairs = len(pivot)
    return pivot, mean_ranks, n_pairs


def friedman_stat(pivot: pd.DataFrame) -> tuple[float, float]:
    """Return Friedman chi-square stat and p-value across methods."""
    arrays = [pivot[m].to_numpy() for m in METHODS]
    chi2, p = stats.friedmanchisquare(*arrays)
    return float(chi2), float(p)


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    q = NEMENYI_Q_005[k]
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


# ---- Figure 2.10 : CD diagram (Demsar style) ---------------------------------

def draw_cd_diagram(mean_ranks: pd.Series, cd: float, n_pairs: int, out_path: Path) -> None:
    """Demsar-style horizontal CD diagram with method lines and CD bar.

    Width ≈ 14 cm at 800 dpi.
    """
    k = len(mean_ranks)
    rmin = 1.0
    rmax = float(k)
    # Sort by mean rank ascending (best on the left)
    sr = mean_ranks.sort_values(ascending=True)

    fig_w_in = 14 / 2.54  # 14 cm
    fig_h_in = 9 / 2.54   # 9 cm
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in), dpi=800)
    ax.set_xlim(rmin - 0.3, rmax + 0.3)
    ax.set_ylim(0, 6)
    ax.invert_yaxis()
    ax.axis("off")

    # Main horizontal axis at the top
    y_axis = 1.0
    ax.hlines(y_axis, rmin, rmax, colors="black", linewidth=1.2)
    # Major ticks at integer ranks
    for r in range(int(rmin), int(rmax) + 1):
        ax.vlines(r, y_axis - 0.07, y_axis, colors="black", linewidth=1.0)
        ax.text(r, y_axis - 0.18, str(r), ha="center", va="bottom", fontsize=9)
    # Minor ticks 0.5
    for r_h in np.arange(rmin + 0.5, rmax, 1.0):
        ax.vlines(r_h, y_axis - 0.04, y_axis, colors="black", linewidth=0.7)

    # Methods: half on the left, half on the right
    n = len(sr)
    half = (n + 1) // 2
    left_methods = sr.iloc[:half]
    right_methods = sr.iloc[half:][::-1]  # reverse so the worst goes outermost

    label_y_step = 0.55
    base_y = 1.7

    def draw_label(rank: float, name: str, side: str, idx: int) -> None:
        y_label = base_y + idx * label_y_step
        # vertical+ horizontal connector
        ax.vlines(rank, y_axis, y_label, colors="black", linewidth=0.8)
        if side == "left":
            ax.hlines(y_label, rank, rmin - 0.25, colors="black", linewidth=0.8)
            ax.text(
                rmin - 0.30, y_label, name,
                ha="right", va="center",
                fontsize=9,
                fontweight="bold" if name == "TESA-26" else "normal",
                color=COLORS.get(name, "black"),
            )
        else:
            ax.hlines(y_label, rmax + 0.25, rank, colors="black", linewidth=0.8)
            ax.text(
                rmax + 0.30, y_label, name,
                ha="left", va="center",
                fontsize=9,
                fontweight="bold" if name == "TESA-26" else "normal",
                color=COLORS.get(name, "black"),
            )

    for i, (name, rank) in enumerate(left_methods.items()):
        draw_label(float(rank), str(name), "left", i)
    for i, (name, rank) in enumerate(right_methods.items()):
        draw_label(float(rank), str(name), "right", i)

    # CD bar above axis
    cd_y = 0.45
    cd_x0 = rmin
    cd_x1 = rmin + cd
    ax.hlines(cd_y, cd_x0, cd_x1, colors="black", linewidth=2.0)
    ax.vlines(cd_x0, cd_y - 0.06, cd_y + 0.06, colors="black", linewidth=1.5)
    ax.vlines(cd_x1, cd_y - 0.06, cd_y + 0.06, colors="black", linewidth=1.5)
    ax.text(
        (cd_x0 + cd_x1) / 2, cd_y - 0.18,
        f"CD = {cd:.2f}",
        ha="center", va="top", fontsize=9,
    )

    # Group lines: methods whose mean ranks differ by < CD are not significantly different
    ranks_sorted = sr.values
    names_sorted = sr.index.tolist()
    cliques: list[tuple[int, int]] = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ranks_sorted[j + 1] - ranks_sorted[i] < cd:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    # Keep only maximal cliques
    maximal: list[tuple[int, int]] = []
    for a, b in cliques:
        if not any((a >= x and b <= y and (a, b) != (x, y)) for x, y in cliques):
            maximal.append((a, b))

    group_y = 1.25
    for k_idx, (a, b) in enumerate(maximal):
        y = group_y + k_idx * 0.13
        ax.hlines(
            y,
            ranks_sorted[a] - 0.04,
            ranks_sorted[b] + 0.04,
            colors="black",
            linewidth=2.5,
        )

    ax.text(
        rmin - 0.3, 5.7,
        f"N = {n_pairs} (датасет × sparsity), k = {k}, α = 0.05",
        ha="left", va="bottom", fontsize=8, style="italic",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=800, bbox_inches="tight")
    plt.close(fig)


# ---- Figure 2.11 : F1 vs Sparsity --------------------------------------------

def main() -> None:
    df = load_all()
    pivot, mean_ranks, n_pairs = compute_ranks(df)
    chi2, p_val = friedman_stat(pivot)
    cd = nemenyi_cd(k=len(METHODS), n=n_pairs, alpha=0.05)

    log: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log.append(line)

    emit("=== Coverage ===")
    emit(f"datasets present : {sorted(df['dataset'].unique())}")
    emit(f"methods present  : {sorted(df['Метод'].unique())}")
    emit(f"pairs (datasets×sparsity) with all 5 methods: {n_pairs}")

    emit("\n=== Mean ranks (lower = better) ===")
    for m, r in mean_ranks.sort_values().items():
        emit(f"  {m:20s}  {r:5.2f}")

    emit(f"\nFriedman chi2 = {chi2:.3f}, p = {p_val:.4g}")
    emit(f"Nemenyi CD (k=5, N={n_pairs}, α=0.05) = {cd:.3f}")

    # Mean F1 on 95-98% (we have 95,96,97 in source)
    high = ["95%", "96%", "97%"]
    sub = df[df["sparsity"].isin(high)]
    f1_high = sub.groupby("Метод")["F1"].mean()
    emit("\n=== Mean F1 at 95-97% sparsity ===")
    for m in METHODS:
        if m in f1_high.index:
            emit(f"  {m:20s}  {f1_high[m]:.3f}")

    out_2_10 = OUT_DIR / "new_fig_2_10_ex08_cd_5methods.png"
    draw_cd_diagram(mean_ranks, cd, n_pairs, out_2_10)

    emit(f"\n[OK] figs/{out_2_10.name}")

    stats_path = OUT_DIR / "friedman_stats.txt"
    stats_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"[OK] figs/{stats_path.name}")


if __name__ == "__main__":
    main()
