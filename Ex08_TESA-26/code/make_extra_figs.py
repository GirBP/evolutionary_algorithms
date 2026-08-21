#!/usr/bin/env python3
# make_extra_figs.py — Додаткові ілюстративні рисунки для Ex08 (не з дисертації):
# (а) F1 проти розрідженості на всіх 6 наборах даних (сітка 2×3, TESA-26 виділено);
# (б) барчарт середнього RCU по методах.
#
# Читає ЛИШЕ наявні results/*_table.csv цієї папки (без змін чисел) і записує
# PNG у ../figs/. Порядок наборів даних та список методів визначаються складом
# самих CSV-файлів.
from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import matplotlib.pyplot as plt
import pandas as pd

# ---- Paths -------------------------------------------------------------------
SRC_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "results"
OUT_DIR: Final[Path] = Path(__file__).resolve().parents[1] / "figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS: Final[list[str]] = ["blobs", "circles", "cnn", "moons", "resnet", "spirals"]
DATASET_TITLES: Final[dict[str, str]] = {
    "blobs": "Blobs (MLP)",
    "circles": "Circles (MLP)",
    "cnn": "FashionMNIST / CompactCNN",
    "moons": "Moons (MLP)",
    "resnet": "FashionMNIST / CompactResNet",
    "spirals": "Spirals (MLP)",
}

HIGHLIGHT_METHOD: Final[str] = "TESA-26"
HIGHLIGHT_COLOR: Final[str] = "#C0392B"  # bold red
OTHER_COLOR: Final[str] = "#95A5A6"      # muted grey

SPARSITY_COL_RE: Final[re.Pattern[str]] = re.compile(r"^\d+%$")
NON_SPARSITY_COLS: Final[set[str]] = {"#", "Метод", "AUSC", "Dev%", "AUSCa", "RCU"}


def load_table(dataset: str) -> pd.DataFrame:
    """Read one results/<dataset>_table.csv as-is (semicolon-separated)."""
    fp = SRC_DIR / f"{dataset}_table.csv"
    df = pd.read_csv(fp, sep=";", encoding="utf-8-sig")
    if "#" in df.columns:
        df = df.drop(columns=["#"])
    return df


def sparsity_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in NON_SPARSITY_COLS and SPARSITY_COL_RE.match(str(c))]
    return sorted(cols, key=lambda c: int(c.rstrip("%")))


# ---- (a) F1 vs sparsity, 2x3 grid, TESA-26 highlighted ------------------------

def draw_f1_grid(out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16 / 2.54 * 1.9, 9 / 2.54 * 1.9), dpi=300)

    for ax, dataset in zip(axes.flat, DATASETS):
        df = load_table(dataset)
        sp_cols = sparsity_columns(df)
        sp_x = [int(c.rstrip("%")) for c in sp_cols]

        for _, row in df.iterrows():
            method = str(row["Метод"])
            y = pd.to_numeric(row[sp_cols], errors="coerce").to_numpy(dtype=float)
            is_tesa = method == HIGHLIGHT_METHOD
            ax.plot(
                sp_x, y,
                color=HIGHLIGHT_COLOR if is_tesa else OTHER_COLOR,
                linewidth=2.2 if is_tesa else 0.9,
                alpha=1.0 if is_tesa else 0.5,
                zorder=3 if is_tesa else 1,
                marker="o" if is_tesa else None,
                markersize=3.5,
            )

        ax.set_title(DATASET_TITLES.get(dataset, dataset), fontsize=10)
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Розрідженість, %", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("F1-міра", fontsize=9)

    # Common legend: TESA-26 (highlighted) vs решта методів (grey)
    handles = [
        plt.Line2D([0], [0], color=HIGHLIGHT_COLOR, linewidth=2.2, marker="o", markersize=4, label="TESA-26"),
        plt.Line2D([0], [0], color=OTHER_COLOR, linewidth=0.9, alpha=0.6, label="решта методів (Ex08)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Ex08 — F1-міра проти розрідженості на всіх 6 наборах даних", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---- (b) Bar chart of mean RCU per method -------------------------------------

def draw_rcu_bar(out_path: Path) -> None:
    frames: list[pd.DataFrame] = []
    for dataset in DATASETS:
        df = load_table(dataset)
        if "RCU" not in df.columns:
            continue
        sub = df[["Метод", "RCU"]].copy()
        sub["RCU"] = pd.to_numeric(sub["RCU"], errors="coerce")
        frames.append(sub)
    all_rcu = pd.concat(frames, ignore_index=True)

    # Канонічний перелік 10 методів (як у results/tables/ex08_friedman_nemenyi.md)
    CANON = ["TESA-26", "SET-v2", "FES-NSDE", "ACDE", "E-HTA", "Magnitude",
             "EvoSynFlow", "SparseGPT", "WANDA", "RIA"]
    all_rcu["Метод"] = all_rcu["Метод"].str.replace(r"\s*\((SOTA|ours|авт\.?)\)", "", regex=True).str.strip()
    all_rcu = all_rcu[all_rcu["Метод"].str.replace("Evo-SynFlow", "EvoSynFlow").isin(CANON) |
                      all_rcu["Метод"].isin(["Evo-SynFlow (Ex07)", "SET-v2 (TESA init)", "E-HTA (Hessian)"])]
    all_rcu["Метод"] = (all_rcu["Метод"]
                        .str.replace("Evo-SynFlow (Ex07)", "EvoSynFlow", regex=False)
                        .str.replace("SET-v2 (TESA init)", "SET-v2", regex=False)
                        .str.replace("E-HTA (Hessian)", "E-HTA", regex=False)
                        .str.replace("Evo-SynFlow", "EvoSynFlow", regex=False))
    mean_rcu = (
        all_rcu.groupby("Метод")["RCU"]
        .mean()
        .dropna()
        .sort_values()
    )

    colors = [HIGHLIGHT_COLOR if m == HIGHLIGHT_METHOD else OTHER_COLOR for m in mean_rcu.index]

    fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54), dpi=300)
    ax.barh(mean_rcu.index, mean_rcu.to_numpy(), color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("RCU (усереднено по наборах даних, логарифмічна шкала)", fontsize=10)
    ax.set_title("Ex08 — обчислювальна вартість (RCU) по методах", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_grid = OUT_DIR / "ex08_extra_f1_vs_sparsity_grid.png"
    out_rcu = OUT_DIR / "ex08_extra_rcu_by_method.png"

    draw_f1_grid(out_grid)
    print(f"[OK] {out_grid}")

    draw_rcu_bar(out_rcu)
    print(f"[OK] {out_rcu}")


if __name__ == "__main__":
    main()
