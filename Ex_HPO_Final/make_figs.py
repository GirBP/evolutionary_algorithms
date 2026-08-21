#!/usr/bin/env python3
"""
Фігури для дисертації (§ 3.6): середні ранги Фрідмана, AUCC, ранги за групами задач.

Читає ЛИШЕ готові канонічні CSV з results/GLOBAL_ANALYSIS/ (жодних power
даних не перераховується і не підганяється) — числа збігаються з Summary.md.

Джерела:
    results/GLOBAL_ANALYSIS/aggregated_results.csv  — колонка 'rank' (метод × задача)
    results/GLOBAL_ANALYSIS/aucc_results.csv        — колонка 'aucc' (метод × задача)

Використання:
    python3 make_figs.py

Вихід: figs/01_friedman_ranks.png, figs/02_aucc.png, figs/03_rank_by_tier.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

ROOT = Path(__file__).resolve().parent
GLOBAL_ANALYSIS = ROOT / "results" / "GLOBAL_ANALYSIS"
FIGS_DIR = ROOT / "figs"

# Флагманські методи цієї зони (§ 3.3–3.5 дисертації), що виділяються на рисунках.
FLAGSHIP_METHODS = {"SACMA-DAC", "SACMA-MAB", "WL-CMA"}
# Четвертий авторський метод (§ 3.2), присутній серед 11, але без окремого виділення.
OTHER_AUTHOR_METHOD = "Sigma-CMA"

COLOR_FLAGSHIP = "#1a5276"
COLOR_AUTHOR = "#5499c7"
COLOR_BASELINE = "#95a5a6"

TIER_ORDER = ["L0", "L2", "L2_MLP_PD1", "L3_NAS_SUPER", "L4", "L5_FCNET"]
TIER_SHORT_LABELS = {
    "L0": "Гр.1\nМЛП (D=7)",
    "L2": "Гр.2\nСурогат.\nкласиф. (D=7)",
    "L2_MLP_PD1": "Гр.3\nСтохаст.\n(D=4)",
    "L3_NAS_SUPER": "Гр.4\nНАС (D=5)",
    "L4": "Гр.5\nВис. розм.\n(D=9-17)",
    "L5_FCNET": "Гр.6\nГлибокі\nМЛП (D=6)",
}

FRIEDMAN_CHI2_GLOBAL = 176.80  # results/GLOBAL_ANALYSIS/Summary.md, розділ 3.1 (Global, k=11, N=43)
FRIEDMAN_P_GLOBAL = "1.08e-32"


def bar_color(method: str) -> str:
    if method in FLAGSHIP_METHODS:
        return COLOR_FLAGSHIP
    if method == OTHER_AUTHOR_METHOD:
        return COLOR_AUTHOR
    return COLOR_BASELINE


def fig_friedman_ranks(agg: pd.DataFrame) -> None:
    """(a) Барчарт середніх рангів Фрідмана 11 методів, з χ² в анотації."""
    mean_rank = agg.groupby("method")["rank"].mean().sort_values()
    colors = [bar_color(m) for m in mean_rank.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(mean_rank.index, mean_rank.values, color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Середній ранг (тест Фрідмана, менше = краще)")
    ax.set_title("Глобальний рейтинг 11 методів HPO (43 задачі)")

    for bar, val in zip(bars, mean_rank.values):
        ax.text(val + 0.08, bar.get_y() + bar.get_height() / 2, f"{val:.2f}",
                va="center", fontsize=9)

    ax.annotate(
        f"Тест Фрідмана (omnibus): χ² = {FRIEDMAN_CHI2_GLOBAL:.2f}, "
        f"p = {FRIEDMAN_P_GLOBAL}, k=11, N=43",
        xy=(0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=9, style="italic",
    )

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FLAGSHIP, label="SACMA-DAC / SACMA-MAB / WL-CMA"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_AUTHOR, label="Sigma-CMA (авторський)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASELINE, label="Базові методи"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", bbox_to_anchor=(0.98, 0.97), fontsize=8.5)

    out = FIGS_DIR / "01_friedman_ranks.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  збережено: {out.relative_to(ROOT)}")


def fig_aucc(aucc: pd.DataFrame) -> None:
    """(b) Барчарт середнього AUCC 11 методів (крос-методна нормалізація)."""
    mean_aucc = aucc.groupby("method")["aucc"].mean().sort_values(ascending=False)
    colors = [bar_color(m) for m in mean_aucc.index]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(mean_aucc.index, mean_aucc.values, color=colors, edgecolor="white")
    ax.set_ylabel("Середній AUCC (більше = швидша збіжність)")
    ax.set_title("Швидкість збіжності 11 методів HPO (AUCC, крос-методна нормалізація)")
    ax.set_ylim(mean_aucc.min() - 0.02, mean_aucc.max() + 0.015)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    for bar, val in zip(bars, mean_aucc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.001, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8.5, rotation=0)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_FLAGSHIP, label="SACMA-DAC / SACMA-MAB / WL-CMA"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_AUTHOR, label="Sigma-CMA (авторський)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_BASELINE, label="Базові методи"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8.5)

    out = FIGS_DIR / "02_aucc.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  збережено: {out.relative_to(ROOT)}")


def fig_rank_by_tier(agg: pd.DataFrame) -> None:
    """(в) Теплова карта середніх рангів 11 методів по 6 групах задач (Summary.md, розділ 4)."""
    pivot = (
        agg.groupby(["method", "tier"])["rank"]
        .mean()
        .unstack("tier")
        .reindex(columns=TIER_ORDER)
    )
    # Методи впорядковано за глобальним середнім рангом (як у Summary.md, розділ 2).
    pivot = pivot.loc[agg.groupby("method")["rank"].mean().sort_values().index]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=11)

    ax.set_xticks(range(len(TIER_ORDER)))
    ax.set_xticklabels([TIER_SHORT_LABELS[t] for t in TIER_ORDER], fontsize=8.5)
    ax.set_yticks(range(len(pivot.index)))
    y_labels = [
        f"★ {m}" if m in FLAGSHIP_METHODS else (f"○ {m}" if m == OTHER_AUTHOR_METHOD else m)
        for m in pivot.index
    ]
    ax.set_yticklabels(y_labels)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                continue
            text_color = "white" if (v <= 2.5 or v >= 8.5) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5, color=text_color)

    ax.set_title("Середній ранг методу за 6 групами задач (менше = краще)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Середній ранг")

    out = FIGS_DIR / "03_rank_by_tier.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  збережено: {out.relative_to(ROOT)}")


def main() -> None:
    FIGS_DIR.mkdir(exist_ok=True)

    agg = pd.read_csv(GLOBAL_ANALYSIS / "aggregated_results.csv")
    aucc = pd.read_csv(GLOBAL_ANALYSIS / "aucc_results.csv")

    print("Генерація фігур із results/GLOBAL_ANALYSIS/ ...")
    fig_friedman_ranks(agg)
    fig_aucc(aucc)
    fig_rank_by_tier(agg)
    print("Готово.")


if __name__ == "__main__":
    main()
