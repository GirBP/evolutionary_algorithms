#!/usr/bin/env python3
"""
HPO Benchmark — Повний статистичний аналіз (6-рівневий протокол).

Стандарти: Demšar (2006), Benavoli et al. (2017), Pineau et al. (2021).

Використання:
    python3 analyze_full.py              # Повний аналіз
    python3 analyze_full.py --tier L0    # Тільки один тієр
    python3 analyze_full.py --no-plots   # Тільки таблиці, без графіків

Вихід: results/GLOBAL_ANALYSIS/
"""
from __future__ import annotations

import os
import sys
import json
import glob
import warnings
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════

RESULTS_ROOT = Path(__file__).parent / "results"
OUTPUT_DIR = RESULTS_ROOT / "GLOBAL_ANALYSIS"

# Display names
METHOD_NAMES = {
    "sacma_v3":       "SACMA-DAC",
    "sacma_base":     "SACMA-base",
    "sacma_lazy":     "SACMA-lazy",
    "sacma_mab":      "SACMA-MAB",
    "antivanila":     "Sigma-CMA",
    "whales_cma":     "WL-CMA",
    "iw_moea":        "IW-MOEA",
    "ordinv_cma":     "OrdInv-CMA",
    "cmaes_pure":     "CMA-ES",
    "random_search":  "Random",
    "tpe":            "TPE",
    "bo_gp":          "GP-BO",
    "smac_method":    "SMAC",
    "dehb_method":    "DEHB",
    "shade":          "SHADE",
    "lshade":         "L-SHADE",
}

AUTHOR_METHODS = {
    "SACMA-DAC", "SACMA-MAB",
    "Sigma-CMA", "WL-CMA",
}

BASELINE_METHODS = {
    "CMA-ES", "Random", "TPE", "GP-BO", "SMAC", "DEHB", "L-SHADE",
}

EXCLUDED_METHODS = {
    "SHADE", "SACMA-lazy", "SACMA-base", "OrdInv-CMA", "IW-MOEA"
}

# Method families for grouped analysis
METHOD_FAMILIES = {
    "CMA-surrogate": ["SACMA-DAC", "SACMA-MAB", "Sigma-CMA"],
    "CMA-hybrid":    ["WL-CMA"],
    "DE-based":      ["L-SHADE"],
    "Model-based":   ["GP-BO", "TPE", "SMAC", "DEHB"],
    "Baseline":      ["CMA-ES", "Random"],
}

TIER_ORDER = ["L0", "L2", "L2_MLP_PD1", "L3_NAS_SUPER", "L4", "L5_FCNET"]

TIER_DISPLAY_NAMES = {
    "L0": "Група 1: Базові багатошарові персептрони (D=7)",
    "L2": "Група 2: Суррогатні моделі класифікації (D=7)",
    "L2_MLP_PD1": "Група 3: Задачі з високим рівнем стохастичності (D=4)",
    "L3_NAS_SUPER": "Група 4: Нейроархітектурний пошук (D=5)",
    "L4": "Група 5: Задачі високої розмірності (D=9-17)",
    "L5_FCNET": "Група 6: Глибокі повнозв'язні архітектури (D=6)"
}


# ═══════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_all_results(tiers=None):
    """Load all JSON results into a DataFrame."""
    rows = []
    search_tiers = tiers or TIER_ORDER

    for tier in search_tiers:
        tier_dir = RESULTS_ROOT / tier
        if not tier_dir.exists():
            continue
        for fp in sorted(tier_dir.glob("*.json")):
            try:
                with open(fp) as f:
                    d = json.load(f)
            except Exception:
                continue

            method_raw = d.get("method", fp.stem.split("__")[0])
            display = METHOD_NAMES.get(method_raw, method_raw)

            if display in EXCLUDED_METHODS:
                continue

            curve = d.get("curve", [])
            rcu_hpo = d.get("rcu_hpo", 0)
            rcu_total = d.get("rcu_total", 0)

            rows.append({
                "tier": tier,
                "dataset": d.get("dataset", "?"),
                "model": d.get("model", "?"),
                "method_raw": method_raw,
                "method": display,
                "seed": d.get("seed", 0),
                "loss": d.get("loss", float("inf")),
                "rcu_hpo": rcu_hpo if rcu_hpo else 0,
                "rcu_total": rcu_total if rcu_total else 0,
                "budget": len(curve),
                "curve": curve,
                "is_author": display in AUTHOR_METHODS,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[ERROR] No result JSONs found!")
        sys.exit(1)

    # Task identifier
    df["task"] = df["tier"] + "/" + df["dataset"] + "/" + df["model"]

    # Filter out inf losses (failed runs)
    n_total = len(df)
    df = df[df["loss"] < 1e10].copy()
    n_valid = len(df)
    if n_total - n_valid > 0:
        print(f"  Filtered {n_total - n_valid} failed runs (inf loss)")

    return df


# ═══════════════════════════════════════════════════════════
# 2. PER-TASK AGGREGATION
# ═══════════════════════════════════════════════════════════

def compute_aggregation(df):
    """Compute median loss, IQR, ranks, and normalized regret per task."""
    agg = df.groupby(["task", "method"]).agg(
        median_loss=("loss", "median"),
        mean_loss=("loss", "mean"),
        std_loss=("loss", "std"),
        iqr_loss=("loss", lambda x: x.quantile(0.75) - x.quantile(0.25)),
        median_rcu_hpo=("rcu_hpo", "median"),
        median_rcu_total=("rcu_total", "median"),
        n_seeds=("seed", "count"),
    ).reset_index()

    # Rank within each task (by median_loss, lower is better)
    agg["rank"] = agg.groupby("task")["median_loss"].rank(method="average")

    # Normalized regret: (L_m - L_best) / (L_random - L_best)
    task_stats = agg.groupby("task").agg(
        L_min=("median_loss", "min"),
        L_max=("median_loss", "max"),
    )
    # Get random search loss per task
    random_losses = agg[agg["method"] == "Random"].set_index("task")["median_loss"]
    task_stats["L_random"] = random_losses
    # Fallback: use max if random is missing or is the best
    task_stats["L_random"] = task_stats["L_random"].fillna(task_stats["L_max"])

    agg = agg.merge(task_stats, on="task", how="left")
    denom = (agg["L_random"] - agg["L_min"]).replace(0, 1e-10)
    agg["norm_regret"] = (agg["median_loss"] - agg["L_min"]) / denom
    agg["norm_regret"] = agg["norm_regret"].clip(0, None)

    # Tier column
    agg["tier"] = agg["task"].str.split("/").str[0]

    # Is author?
    agg["is_author"] = agg["method"].isin(AUTHOR_METHODS)

    return agg


# ═══════════════════════════════════════════════════════════
# 3. AUCC (Convergence Speed) — cross-method normalization
#    Ref: Pushak & Hoos (2022), "AutoML: A Survey"
# ═══════════════════════════════════════════════════════════

def compute_aucc(df):
    """Compute Area Under Convergence Curve per (task, method).

    Normalization: cross-method, per-task.
    L_best(t) = best final loss across ALL methods on task t.
    L_worst(t) = worst first-step loss across ALL methods on task t.
    AUCC = integral of (L_worst - curve) / (L_worst - L_best) dt.
    Higher = faster convergence to better solution.
    """
    results = []

    # First pass: compute per-task global bounds
    task_bounds = {}
    for task, t_grp in df.groupby("task"):
        all_final_losses = t_grp.groupby("method")["loss"].median()
        all_curves = []
        for _, row in t_grp.iterrows():
            if row["curve"] and len(row["curve"]) > 0:
                all_curves.append(row["curve"][0])  # first step
        L_best = all_final_losses.min()
        L_worst = max(all_curves) if all_curves else all_final_losses.max()
        # Ensure L_worst > L_best
        if L_worst <= L_best:
            L_worst = all_final_losses.max()
        task_bounds[task] = (L_best, L_worst)

    # Second pass: compute AUCC with cross-method bounds
    for (task, method), grp in df.groupby(["task", "method"]):
        L_best, L_worst = task_bounds.get(task, (0, 1))
        rng = L_worst - L_best
        if rng < 1e-12:
            rng = 1e-12

        aucc_vals = []
        for _, row in grp.iterrows():
            curve = row["curve"]
            if not curve or len(curve) < 2:
                continue
            # Normalize: higher = better
            normalized = [max(0, (L_worst - c) / rng) for c in curve]
            # Clip negative values (method worse than L_worst)
            normalized = [min(n, 1.0) for n in normalized]
            aucc_vals.append(np.trapz(normalized) / (len(normalized) - 1))

        if aucc_vals:
            results.append({
                "task": task,
                "method": method,
                "aucc": np.median(aucc_vals),
            })
    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════
# 4. STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════

def friedman_test(rank_matrix):
    """Run Friedman omnibus test. Input: tasks × methods DataFrame of ranks."""
    k = rank_matrix.shape[1]
    N = rank_matrix.shape[0]

    stat, p = stats.friedmanchisquare(*[rank_matrix.iloc[:, i] for i in range(k)])
    avg_ranks = rank_matrix.mean().sort_values()

    return {
        "statistic": stat,
        "p_value": p,
        "k": k,
        "N": N,
        "significant": p < 0.05,
        "avg_ranks": avg_ranks,
    }


def nemenyi_cd(k, N, alpha=0.05):
    """Compute Nemenyi Critical Difference.

    CD = q_α * sqrt(k(k+1)/(6N))
    q_α from Studentized Range distribution (divided by sqrt(2)).
    """
    # Use scipy studentized_range if available
    try:
        from scipy.stats import studentized_range
        q_raw = studentized_range.ppf(1 - alpha, k, np.inf)
        q = q_raw / np.sqrt(2)
    except Exception:
        # Fallback: hardcoded critical values for α=0.05 (Demšar 2006, Table 5)
        q_table = {
            2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
            7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
            12: 3.268, 13: 3.314, 14: 3.354, 15: 3.391, 16: 3.426,
            17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544,
        }
        q = q_table.get(k, 3.5)

    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    return cd


def compute_nemenyi_groups(avg_ranks, cd):
    """Find groups of methods that are NOT significantly different (rank diff < CD)."""
    methods_sorted = avg_ranks.sort_values()

    # Compute cliques (maximal groups within CD)
    cliques = []
    names = list(methods_sorted.index)
    vals = list(methods_sorted.values)
    for i in range(len(names)):
        clique = [names[i]]
        for j in range(i + 1, len(names)):
            if vals[j] - vals[i] < cd:
                clique.append(names[j])
            else:
                break
        if len(clique) > 1:
            cliques.append(clique)

    # Remove subsets
    unique_cliques = []
    for c in cliques:
        is_subset = False
        for uc in unique_cliques:
            if set(c).issubset(set(uc)):
                is_subset = True
                break
        if not is_subset:
            unique_cliques = [uc for uc in unique_cliques if not set(uc).issubset(set(c))]
            unique_cliques.append(c)

    return unique_cliques


def bayesian_signed_rank(ranks_a, ranks_b, rope=1.0, n_samples=50000):
    """Bayesian Signed-Rank test (Benavoli et al., 2017).

    Compares two methods via their ranks across tasks.
    ROPE = Region of Practical Equivalence (1 rank unit = practically same).

    Returns: P(A better), P(equivalent), P(B better)
    """
    diff = ranks_a - ranks_b  # negative = A has lower (better) rank
    n = len(diff)

    if n == 0:
        return 0.33, 0.34, 0.33

    # Wilcoxon signed-rank statistic with Bayesian interpretation
    # Use bootstrap sampling for posterior estimation
    rng = np.random.default_rng(42)
    counts = np.zeros(3)  # [A_better, equivalent, B_better]

    for _ in range(n_samples):
        # Dirichlet-bootstrapped sample
        weights = rng.dirichlet(np.ones(n))
        weighted_mean = np.sum(weights * diff)

        if weighted_mean < -rope:  # A has significantly lower rank
            counts[0] += 1
        elif weighted_mean > rope:  # B has significantly lower rank
            counts[2] += 1
        else:  # Within ROPE
            counts[1] += 1

    probs = counts / n_samples
    return probs[0], probs[1], probs[2]


def compute_bayesian_pairwise(agg, reference_method, competitors):
    """Run Bayesian signed-rank for reference vs each competitor."""
    results = []
    tasks = agg["task"].unique()

    ref_ranks = agg[agg["method"] == reference_method].set_index("task")["rank"]

    for comp in competitors:
        comp_ranks = agg[agg["method"] == comp].set_index("task")["rank"]
        common = ref_ranks.index.intersection(comp_ranks.index)
        if len(common) < 3:
            continue

        p_better, p_equiv, p_worse = bayesian_signed_rank(
            ref_ranks.loc[common].values,
            comp_ranks.loc[common].values,
            rope=1.0  # 1 rank unit ROPE (Benavoli et al., 2017)
        )

        results.append({
            "reference": reference_method,
            "competitor": comp,
            "P(ref_better)": p_better,
            "P(equivalent)": p_equiv,
            "P(comp_better)": p_worse,
            "verdict": ("REF WINS" if p_better > 0.5 else
                        "EQUIVALENT" if p_equiv > 0.5 else
                        "COMP WINS"),
            "n_tasks": len(common),
        })

    return pd.DataFrame(results)


def compute_top6_per_tier(agg):
    """Per-tier Top-6 focused analysis."""
    tier_top6 = {}
    for tier in TIER_ORDER:
        tier_data = agg[agg["tier"] == tier]
        if tier_data.empty:
            continue
        tier_ranks = tier_data.groupby("method")["rank"].mean().sort_values()
        top6_names = tier_ranks.head(6).index.tolist()

        # Detailed stats for top-6
        top6_detail = tier_data[tier_data["method"].isin(top6_names)].groupby("method").agg(
            avg_rank=("rank", "mean"),
            avg_loss=("median_loss", "mean"),
            avg_rcu=("median_rcu_total", "mean"),
            avg_iqr=("iqr_loss", "mean"),
            avg_norm_regret=("norm_regret", "mean"),
            n_wins=("rank", lambda x: (x == 1).sum()),
            n_top3=("rank", lambda x: (x <= 3).sum()),
            n_tasks=("rank", "count"),
        ).loc[top6_names]  # maintain ranking order

        tier_top6[tier] = top6_detail

    return tier_top6


# ═══════════════════════════════════════════════════════════
# 5. VISUALIZATIONS (UKRAINIAN LOCALIZATION)
# ═══════════════════════════════════════════════════════════

def plot_cd_diagram(avg_ranks, cd, title, output_path, highlight_author=True):
    """Draw a Critical Difference diagram (Demšar style)."""
    methods_sorted = avg_ranks.sort_values()
    k = len(methods_sorted)
    names = list(methods_sorted.index)
    ranks = list(methods_sorted.values)

    # Find cliques
    cliques = compute_nemenyi_groups(avg_ranks, cd)

    fig, ax = plt.subplots(1, 1, figsize=(max(10, k * 0.6), max(4, k * 0.25 + 2)))

    # Axis setup
    low = 1
    high = k
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(0, 1)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_yticks([])

    # Top ruler
    ax.set_xticks(range(1, k + 1))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", direction="out", length=5)

    # Split methods: left half (top ranks) and right half (bottom ranks)
    half = k // 2
    left_methods = [(names[i], ranks[i]) for i in range(half)]
    right_methods = [(names[i], ranks[i]) for i in range(half, k)]

    # Draw clique bars (horizontal bold lines connecting non-significant pairs)
    # We define their y-levels first to prevent overlap with text lines
    clique_y_start = 0.85
    clique_height = 0.04
    lowest_clique_y = clique_y_start - (max(len(cliques), 1) * clique_height)

    # Draw method names and lines
    y_max_text = lowest_clique_y - 0.02
    y_base = 0.05
    y_step = max(0.05, (y_max_text - y_base) / max(1, half, len(right_methods)))

    x_left_text = low - 0.2
    x_right_text = high + 0.2

    for idx, (name, rank) in enumerate(left_methods):
        y = y_base + (len(left_methods) - 1 - idx) * y_step
        color = "#1a5276" if name in AUTHOR_METHODS else "#616161"
        weight = "bold" if name in AUTHOR_METHODS else "normal"

        # vertical clip line from top axis to y
        ax.plot([rank, rank], [0.89, y], color=color, lw=1.2, clip_on=False)
        # horizontal elbow line from vertical drop to text start
        ax.plot([rank, x_left_text], [y, y], color=color, lw=1.2, clip_on=False)
        # text
        ax.text(x_left_text - 0.1, y, f"{name} ({rank:.2f})",
                ha="right", va="center", fontsize=11, color=color, fontweight=weight)

    for idx, (name, rank) in enumerate(right_methods):
        y = y_base + (len(right_methods) - 1 - idx) * y_step
        color = "#1a5276" if name in AUTHOR_METHODS else "#616161"
        weight = "bold" if name in AUTHOR_METHODS else "normal"

        # vertical clip line from top axis to y
        ax.plot([rank, rank], [0.89, y], color=color, lw=1.2, clip_on=False)
        # horizontal elbow line from vertical drop to text start
        ax.plot([rank, x_right_text], [y, y], color=color, lw=1.2, clip_on=False)
        # text
        ax.text(x_right_text + 0.1, y, f"({rank:.2f}) {name}",
                ha="left", va="center", fontsize=11, color=color, fontweight=weight)

    # Draw CD bar at top (moved below ticks at 1.0 to avoid overlap)
    cd_x = low + 0.5
    ax.plot([cd_x, cd_x + cd], [0.93, 0.93], color="red", lw=2.5, clip_on=False)
    ax.text(cd_x + cd / 2, 0.94, f"CD = {cd:.2f}", ha="center", va="bottom",
            fontsize=10, color="red", fontweight="bold")

    # Draw clique bars
    for ci, clique in enumerate(cliques):
        clique_ranks = [avg_ranks[m] for m in clique]
        r_min, r_max = min(clique_ranks), max(clique_ranks)
        y = clique_y_start - ci * clique_height
        ax.plot([r_min, r_max], [y, y], color="#333333", lw=3.0, solid_capstyle="round")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   CD Diagram → {output_path.name}")


def plot_rank_heatmap(agg, out_path):
    """Aggregated heatmap: method ranks averaged per Tier (11×6 with annotations).
    Rows sorted by global average rank (best on top)."""
    import numpy as np

    # --- Tier short names for column headers ---
    tier_short = {
        "L0":           "Гр.1\nBP-мережі\n(D=7)",
        "L2":           "Гр.2\nСурогатні\n(D=7)",
        "L2_MLP_PD1":   "Гр.3\nСтохаст.\n(D=4)",
        "L3_NAS_SUPER":  "Гр.4\nNAS\n(D=5)",
        "L4":           "Гр.5\nВис. розм.\n(D=9-17)",
        "L5_FCNET":     "Гр.6\nFC-Net\n(D=6)",
    }

    # --- Build 11×6 tier-aggregated pivot ---
    tier_pivot = agg.groupby(["method", "tier"])["rank"].mean().reset_index()
    tier_pivot = tier_pivot.pivot(index="method", columns="tier", values="rank")
    # Keep only known tiers in fixed order
    cols_ordered = [t for t in TIER_ORDER if t in tier_pivot.columns]
    tier_pivot = tier_pivot[cols_ordered]
    tier_pivot.columns = [tier_short.get(c, c) for c in tier_pivot.columns]
    tier_pivot = tier_pivot.fillna(tier_pivot.max().max())

    # --- Sort rows by global average rank: best (lowest) on top ---
    tier_pivot["_global_rank"] = tier_pivot.mean(axis=1)
    tier_pivot = tier_pivot.sort_values("_global_rank", ascending=True)
    tier_pivot = tier_pivot.drop(columns=["_global_rank"])

    # --- Row labels with author marker ---
    new_index = [f" {m}" if m in AUTHOR_METHODS else m for m in tier_pivot.index]
    tier_pivot.index = new_index

    # --- Plot with sns.heatmap (no dendrogram, explicit rank order) ---
    fig, ax = plt.subplots(figsize=(12, 8))

    sns.heatmap(
        tier_pivot,
        cmap="coolwarm_r",
        vmin=1, vmax=tier_pivot.values.max(),
        annot=True,
        fmt=".1f",
        annot_kws={"size": 12, "weight": "bold"},
        linewidths=1.0,
        linecolor="white",
        ax=ax,
        cbar_kws={"shrink": 0.6, "label": "Середній ранг"},
    )

    # Colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.invert_yaxis()  # Put 1 (best) at the top, 11 (worst) at the bottom
    cbar.ax.set_ylabel("← Гірше          Краще →", fontsize=10, rotation=90, labelpad=12)

    ax.set_xlabel("Група задач", fontsize=12, labelpad=8)
    ax.set_ylabel("")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=0, ha="center")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=11, rotation=0)
    ax.set_title(
        "Агрегована теплова карта середніх рангів (Менше = Краще)",
        fontsize=14, fontweight="bold", pad=14,
    )

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   Rank Heatmap (sorted by global rank) → {out_path.name}")


def plot_pareto_front(agg, output_path):
    """Pareto front: Normalized Regret vs RCU (per method, averaged over tasks)."""
    method_agg = agg.groupby("method").agg(
        avg_norm_regret=("norm_regret", "mean"),
        avg_rcu_total=("median_rcu_total", "mean"),
        avg_rank=("rank", "mean"),
    ).reset_index()

    def is_pareto(costs):
        import numpy as np
        is_eff = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            if is_eff[i]:
                is_eff[is_eff] = np.any(costs[is_eff] < c, axis=1) | np.all(costs[is_eff] == c, axis=1)
                is_eff[i] = True
        return is_eff

    costs = method_agg[["avg_norm_regret", "avg_rcu_total"]].values
    pareto_mask = is_pareto(costs)

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, row in method_agg.iterrows():
        color = "#1a5276" if row["method"] in AUTHOR_METHODS else "#b03a2e" if row["method"] == "Random" else "#616161"
        marker = "D" if row["method"] in AUTHOR_METHODS else "o"
        edge = "gold" if pareto_mask[idx] else color
        import matplotlib.patches as mpatches
        lw = 2.5 if pareto_mask[idx] else 1.0
        size = 100 if pareto_mask[idx] else 60

        ax.scatter(row["avg_rcu_total"], row["avg_norm_regret"],
                   c=color, marker=marker, s=size, edgecolors=edge, linewidths=lw, zorder=3)
        ax.annotate(row["method"], (row["avg_rcu_total"], row["avg_norm_regret"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(5, 5), textcoords="offset points",
                    color=color, fontweight="bold" if pareto_mask[idx] else "normal")

    pareto_pts = method_agg[pareto_mask].sort_values("avg_rcu_total")
    if len(pareto_pts) > 1:
        ax.plot(pareto_pts["avg_rcu_total"], pareto_pts["avg_norm_regret"],
                "k--", alpha=0.4, lw=1.5, label="Парето-фронт")

    ax.set_xlabel("Середня обчислювальна вартість RCU (менше = дешевше)", fontsize=12)
    ax.set_ylabel("Середня нормована похибка пошуку (менше = краще)", fontsize=12)
    ax.set_title("Парето-фронт: Якість × Вартість (усі задачі)", fontsize=15, fontweight="bold")

    import matplotlib.patches as mpatches
    author_patch = mpatches.Patch(color="#1a5276", label="Запропоновані")
    baseline_patch = mpatches.Patch(color="#616161", label="Базові")
    pareto_patch = mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Парето-оптимальні")
    ax.legend(handles=[author_patch, baseline_patch, pareto_patch], loc="upper right")

    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   Pareto Front → {output_path.name}")


def plot_pareto_per_tier(agg, output_path):
    """Pareto front per tier (2x3 grid)."""
    import numpy as np
    import matplotlib.patches as mpatches

    def is_pareto(costs):
        is_eff = np.ones(len(costs), dtype=bool)
        for i, c in enumerate(costs):
            if is_eff[i]:
                is_eff[is_eff] = np.any(costs[is_eff] < c, axis=1) | np.all(costs[is_eff] == c, axis=1)
                is_eff[i] = True
        return is_eff

    tiers = getattr(agg["tier"], "cat", agg["tier"]).categories if hasattr(agg["tier"], "cat") else sorted(agg["tier"].unique())
    # Ensure L0, L2, L2_MLP_PD1, L3_NAS_SUPER, L4, L5_FCNET order
    tiers = [t for t in ["L0", "L2", "L2_MLP_PD1", "L3_NAS_SUPER", "L4", "L5_FCNET"] if t in agg["tier"].unique()]

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 10))
    axes = axes.flatten()

    for tidx, tier in enumerate(tiers):
        ax = axes[tidx]
        tier_agg = agg[agg["tier"] == tier]

        method_agg = tier_agg.groupby("method").agg(
            avg_norm_regret=("norm_regret", "mean"),
            avg_rcu_total=("median_rcu_total", "mean")
        ).reset_index()

        costs = method_agg[["avg_norm_regret", "avg_rcu_total"]].values
        pareto_mask = is_pareto(costs)

        for idx, row in method_agg.iterrows():
            color = "#1a5276" if row["method"] in AUTHOR_METHODS else "#b03a2e" if row["method"] == "Random" else "#616161"
            marker = "D" if row["method"] in AUTHOR_METHODS else "o"
            edge = "gold" if pareto_mask[idx] else color
            lw = 2.5 if pareto_mask[idx] else 1.0
            size = 100 if pareto_mask[idx] else 60

            ax.scatter(row["avg_rcu_total"], row["avg_norm_regret"],
                       c=color, marker=marker, s=size, edgecolors=edge, linewidths=lw, zorder=3)
            ax.annotate(row["method"], (row["avg_rcu_total"], row["avg_norm_regret"]),
                        fontsize=8, ha="left", va="bottom",
                        xytext=(5, 5), textcoords="offset points",
                        color=color, fontweight="bold" if pareto_mask[idx] else "normal")

        pareto_pts = method_agg[pareto_mask].sort_values("avg_rcu_total")
        if len(pareto_pts) > 1:
            ax.plot(pareto_pts["avg_rcu_total"], pareto_pts["avg_norm_regret"],
                    "k--", alpha=0.4, lw=1.5)

        tier_name = TIER_DISPLAY_NAMES.get(tier, tier)
        ax.set_title(tier_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("RCU", fontsize=9)
        ax.set_ylabel("Нормована похибка", fontsize=9)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    for i in range(len(tiers), len(axes)):
        axes[i].set_visible(False)

    author_patch = mpatches.Patch(color="#1a5276", label="Запропоновані")
    baseline_patch = mpatches.Patch(color="#616161", label="Базові")
    pareto_patch = mpatches.Patch(facecolor="white", edgecolor="gold", linewidth=2, label="Парето-оптимальні")
    fig.legend(handles=[author_patch, baseline_patch, pareto_patch], loc="upper center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   Pareto (Per Tier) → {output_path.name}")



def plot_convergence_top5(df, agg, tier, output_path):
    """Convergence curves: Top-5 methods on 4 representative tasks (2×2 grid, shared legend)."""
    import numpy as np

    tier_agg = agg[agg["tier"] == tier]
    if tier_agg.empty:
        return

    # --- Select top-5 methods by mean rank in this tier ---
    top5 = tier_agg.groupby("method")["rank"].mean().nsmallest(5).index.tolist()
    tier_data = df[df["task"].str.startswith(tier + "/")]
    tasks_in_tier = list(tier_data["task"].unique())

    # --- Pick 4 representative tasks: prefer one per model type ---
    MODEL_PRIORITY = ["mlp", "gb", "rf", "svm", "hgb", "transformer", "resnet"]
    selected_tasks = []
    for model in MODEL_PRIORITY:
        candidates = [t for t in sorted(tasks_in_tier) if t.endswith("/" + model)]
        if candidates and len(selected_tasks) < 4:
            selected_tasks.append(candidates[0])
    # If fewer than 4 found via model types, pad with remaining tasks
    for t in sorted(tasks_in_tier):
        if len(selected_tasks) >= 4:
            break
        if t not in selected_tasks:
            selected_tasks.append(t)
    selected_tasks = selected_tasks[:4]

    # --- Colors: author methods darker blue, others gray scale ---
    PALETTE = {
        top5[0]: "#1a5276",
        top5[1]: "#2e86c1",
        top5[2]: "#27ae60",
        top5[3]: "#d35400",
        top5[4]: "#7d3c98",
    }

    n_shown = len(selected_tasks)
    rows, cols = 2, 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), squeeze=False)

    line_handles = []
    line_labels = []

    for tidx, task in enumerate(selected_tasks):
        ax = axes[tidx // cols][tidx % cols]
        task_data = tier_data[tier_data["task"] == task]
        task_medians = []

        for midx, method in enumerate(top5):
            method_curves = task_data[task_data["method"] == method]["curve"].values
            if len(method_curves) == 0:
                continue

            max_len = max(len(c) for c in method_curves)
            padded = []
            for c in method_curves:
                c_padded = list(c) + [c[-1]] * (max_len - len(c)) if len(c) < max_len else list(c)
                padded.append(c_padded)
            padded = np.array(padded)
            median_curve = np.median(padded, axis=0)
            task_medians.append(median_curve)
            q25 = np.percentile(padded, 25, axis=0)
            q75 = np.percentile(padded, 75, axis=0)

            color = PALETTE.get(method, "#888888")
            lw = 2.4 if method in AUTHOR_METHODS else 1.6
            ls = "-" if method in AUTHOR_METHODS else "--"
            x = range(1, len(median_curve) + 1)
            line, = ax.plot(x, median_curve, label=method, color=color, lw=lw, linestyle=ls)
            ax.fill_between(x, q25, q75, alpha=0.10, color=color)

            # Collect legend handles only from first subplot to avoid duplicates
            if tidx == 0:
                line_handles.append(line)
                line_labels.append(method)

        # Clean title: remove "L0/" prefix and replace "/" with " / "
        parts = task.split("/")
        short_task = " / ".join(parts[1:]) if len(parts) > 1 else task
        ax.set_title(short_task, fontsize=11, fontweight="bold")
        ax.set_xlabel("Епохи пошуку (Evaluations)", fontsize=10)
        ax.set_ylabel("Похибка (Loss)", fontsize=10)

        # Dynamic Y-axis zoom to prevent early outliers from squashing the lines
        if task_medians:
            try:
                # Ignore the first 3 epochs (index 0,1,2) to find the competitive range
                tail_max = max(np.max(c[3:]) for c in task_medians if len(c) > 3)
                tail_min = min(np.min(c) for c in task_medians)
                margin = (tail_max - tail_min) * 0.15
                if margin <= 0: margin = tail_max * 0.1 + 1e-4
                ax.set_ylim(bottom=max(0, tail_min - margin), top=tail_max + margin)
            except ValueError:
                pass

        ax.grid(True, alpha=0.3)

    # Hide any unused axes (if n_shown < 4)
    for tidx in range(n_shown, rows * cols):
        axes[tidx // cols][tidx % cols].set_visible(False)

    # --- Shared legend above all subplots ---
    fig.legend(
        line_handles, line_labels,
        loc="upper center",
        ncol=len(line_handles),
        fontsize=10,
        frameon=True,
        bbox_to_anchor=(0.5, 1.01),
    )

    tier_name = TIER_DISPLAY_NAMES.get(tier, tier)
    fig.suptitle(
        f"Криві збіжності: Топ-5 методів на:\n{tier_name} (медіана ± IQR)",
        fontweight="bold", fontsize=13, y=1.06,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"   Convergence ({tier}) → {output_path.name}")


def plot_win_rate_matrix(agg, output_path):
    """16×16 pairwise win percentage matrix."""
    import numpy as np
    methods = sorted(agg["method"].unique())
    tasks = agg["task"].unique()
    n = len(methods)

    win_matrix = np.zeros((n, n))
    for task in tasks:
        task_data = agg[agg["task"] == task].set_index("method")
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                if i == j: continue
                if mi in task_data.index and mj in task_data.index:
                    if task_data.loc[mi, "median_loss"] < task_data.loc[mj, "median_loss"]:
                        win_matrix[i, j] += 1
                    elif task_data.loc[mi, "median_loss"] == task_data.loc[mj, "median_loss"]:
                        win_matrix[i, j] += 0.5

    total_tasks = len(tasks)
    win_pct = (win_matrix / total_tasks * 100)

    overall_win = win_pct.mean(axis=1)
    sort_idx = np.argsort(overall_win)[::-1]
    win_pct = win_pct[sort_idx][:, sort_idx]
    sorted_methods = np.array(methods)[sort_idx]

    row_labels = [f" {m}" if m in AUTHOR_METHODS else m for m in sorted_methods]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(win_pct, xticklabels=sorted_methods, yticklabels=row_labels,
                cmap="RdYlGn", center=50, annot=True, fmt=".0f",
                linewidths=0.5, cbar_kws={"label": "Ймовірність перемоги (%)"}, ax=ax)

    ax.set_title("Матриця попарних перемог (Рядок перемагає Колонку)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Метод конкурент (Колонка)", fontsize=12)
    ax.set_ylabel("Основний метод (Рядок)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   Win-Rate Matrix → {output_path.name}")


def plot_method_family_boxplot(agg, output_path):
    """Boxplot comparing method families."""
    rows = []
    for family, members in METHOD_FAMILIES.items():
        fdata = agg[agg["method"].isin(members)]
        for _, r in fdata.iterrows():
            rows.append({"family": family, "method": r["method"],
                         "norm_regret": r["norm_regret"], "rank": r["rank"]})
    fdf = pd.DataFrame(rows)
    if fdf.empty: return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    order = list(METHOD_FAMILIES.keys())
    palette = {"CMA-surrogate": "#2980b9", "CMA-hybrid": "#1abc9c",
               "DE-based": "#e67e22", "Model-based": "#9b59b6", "Baseline": "#95a5a6"}

    sns.boxplot(data=fdf, x="family", y="rank", order=order, palette=palette, ax=ax1, showfliers=False)
    sns.stripplot(data=fdf, x="family", y="rank", order=order, color="black", alpha=0.2, ax=ax1)
    ax1.set_title("Розподіл продуктивності за класом алгоритму", fontsize=14, fontweight="bold")
    ax1.set_xlabel("")
    ax1.set_ylabel("Ранг (менше = краще)")
    ax1.tick_params(axis="x", rotation=20)

    sns.boxplot(data=fdf, x="family", y="norm_regret", order=order, palette=palette, ax=ax2, showfliers=False)
    sns.stripplot(data=fdf, x="family", y="norm_regret", order=order, color="black", alpha=0.2, ax=ax2)
    ax2.set_title("Нормована похибка за класом алгоритму", fontsize=14, fontweight="bold")
    ax2.set_xlabel("")
    ax2.set_ylabel("Нормована похибка пошуку")
    ax2.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"   Family Boxplot → {output_path.name}")

# 6. INTERACTION ANALYSIS
# ═══════════════════════════════════════════════════════════

def compute_interaction_table(agg):
    """For each method: where it wins, where it loses."""
    methods = sorted(agg["method"].unique())
    results = []

    for m in methods:
        m_data = agg[agg["method"] == m]
        total_tasks = len(m_data)
        wins = m_data[m_data["rank"] == 1]["task"].tolist()
        top3 = m_data[m_data["rank"] <= 3]["task"].tolist()
        bottom3 = m_data[m_data["rank"] >= (m_data["rank"].max() - 2)]["task"].tolist()
        worst = m_data[m_data["rank"] == m_data.groupby("task")["rank"].transform("max")]["task"].tolist()

        results.append({
            "method": m,
            "is_author": m in AUTHOR_METHODS,
            "avg_rank": m_data["rank"].mean(),
            "median_rank": m_data["rank"].median(),
            "win_count": len(wins),
            "win_pct": len(wins) / total_tasks * 100 if total_tasks else 0,
            "top3_count": len(top3),
            "top3_pct": len(top3) / total_tasks * 100 if total_tasks else 0,
            "total_tasks": total_tasks,
            "best_on": wins[:5],  # cap for readability
            "worst_on": worst[:5],
            "avg_norm_regret": m_data["norm_regret"].mean(),
            "avg_rcu_total": m_data["median_rcu_total"].mean(),
        })

    return pd.DataFrame(results).sort_values("avg_rank")


# ═══════════════════════════════════════════════════════════
# 7. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════

def generate_summary(df, agg, friedman_results, cd_values, interaction, aucc_df, output_dir,
                     bayesian_df=None, tier_top6=None, best_author=None):
    """Generate comprehensive Markdown summary."""
    lines = []
    lines.append("# Бенчмарк HPO — Глобальний статистичний аналіз\n")
    lines.append(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")

    # Data overview
    n_methods = df["method"].nunique()
    n_tasks = df["task"].nunique()
    n_seeds = df.groupby(["task", "method"])["seed"].count().median()
    n_records = len(df)
    lines.append("## 1. Огляд даних\n")
    lines.append(f"| Параметр | Значення |")
    lines.append(f"|---|---|")
    lines.append(f"| Методи | {n_methods} ({len(AUTHOR_METHODS)} запропоновано + {len(BASELINE_METHODS)} базових) |")
    lines.append(f"| Задачі | {n_tasks} |")
    lines.append(f"| Запусків на комірку | {n_seeds:.0f} (медіана) |")
    lines.append(f"| Усього записів | {n_records} |")
    lines.append(f"| Класи задач | {', '.join(sorted(df['tier'].unique()))} |")
    lines.append("")

    # Overall ranking
    lines.append("## 2. Глобальний рейтинг (усі класи задач)\n")
    global_ranks = agg.groupby("method").agg(
        avg_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        avg_norm_regret=("norm_regret", "mean"),
        avg_rcu=("median_rcu_total", "mean"),
    ).sort_values("avg_rank")

    lines.append("| # | Метод | Тип | Сер. ранг | Медіана рангу | Сер. похибка | Сер. RCU |")
    lines.append("|---|-------|-----|-----------|---------------|--------------|----------|")
    for i, (method, row) in enumerate(global_ranks.iterrows(), 1):
        mtype = "**Запропоновано**" if method in AUTHOR_METHODS else "Базовий"
        medal = "" if i == 1 else "" if i == 2 else "" if i == 3 else f"{i}"
        lines.append(f"| {medal} | {method} | {mtype} | "
                     f"{row['avg_rank']:.2f} | {row['median_rank']:.1f} | "
                     f"{row['avg_norm_regret']:.4f} | {row['avg_rcu']:.0f} |")
    lines.append("")

    # Friedman tests
    lines.append("## 3. Статистичні тести\n")
    lines.append("### 3.1 Тест Фрідмана (Omnibus)\n")
    for label, res in friedman_results.items():
        sig = " ЗНАЧУЩИЙ" if res["significant"] else " НЕ значущий"
        lines.append(f"**{label}**: χ² = {res['statistic']:.2f}, "
                     f"p = {res['p_value']:.2e}, k={res['k']}, N={res['N']} → {sig}\n")

    lines.append("### 3.2 Нємені Post-hoc (критична різниця)\n")
    for label, cd in cd_values.items():
        lines.append(f"**{label}**: CD = {cd:.2f}")
    lines.append("")
    lines.append("*Див. діаграми CD у вихідній директорії.*\n")

    # Bayesian Signed-Rank
    if bayesian_df is not None and not bayesian_df.empty and best_author:
        lines.append("### 3.3 Баєсівський знаковий ранговий тест (Benavoli et al., 2017)\n")
        lines.append(f"Референтний метод: **{best_author}** | ROPE = 1 ранг | N = 50 000 семплів апостеріорного розподілу\n")
        lines.append("| Конкурент | P(ref краще) | P(еквівалентні) | P(конк. краще) | Вердикт |")
        lines.append("|-----------|-------------|-----------------|----------------|---------|")
        for _, row in bayesian_df.iterrows():
            icon = "" if row["verdict"] == "REF WINS" else "" if row["verdict"] == "EQUIVALENT" else ""
            lines.append(f"| {row['competitor']} | {row['P(ref_better)']:.3f} | "
                         f"{row['P(equivalent)']:.3f} | {row['P(comp_better)']:.3f} | {icon} {row['verdict']} |")
        lines.append("")

    # Per-tier ranking
    lines.append("## 4. Рейтинг за групами експериментів\n")
    for tier in TIER_ORDER:
        tier_data = agg[agg["tier"] == tier]
        if tier_data.empty:
            continue
        tier_ranks = tier_data.groupby("method")["rank"].mean().sort_values()
        n_tasks_tier = tier_data["task"].nunique()
        tier_name = TIER_DISPLAY_NAMES.get(tier, tier)
        lines.append(f"### {tier_name} ({n_tasks_tier} задач)\n")
        lines.append(f"| # | Метод | Сер. ранг |")
        lines.append(f"|---|-------|-----------|")
        for i, (method, rank) in enumerate(tier_ranks.items(), 1):
            bold = "**" if method in AUTHOR_METHODS else ""
            lines.append(f"| {i} | {bold}{method}{bold} | {rank:.2f} |")
        lines.append("")

    # Top-6 per-tier detailed
    if tier_top6:
        lines.append("## 4б. Деталізований аналіз Топ-6\n")
        for tier, t6 in tier_top6.items():
            n_tasks_tier = agg[agg["tier"] == tier]["task"].nunique()
            tier_name = TIER_DISPLAY_NAMES.get(tier, tier)
            lines.append(f"### {tier_name} — Топ-6 ({n_tasks_tier} задач)\n")
            lines.append("| Метод | Тип | Сер. ранг | Сер. Loss | Сер. RCU | Перемоги | Топ-3 | Стабільність (IQR) |")
            lines.append("|-------|-----|-----------|-----------|----------|----------|-------|--------------------|")
            for method, row in t6.iterrows():
                mtype = "Запропоновано" if method in AUTHOR_METHODS else "Базовий"
                bold = "**" if method in AUTHOR_METHODS else ""
                lines.append(f"| {bold}{method}{bold} | {mtype} | {row['avg_rank']:.2f} | "
                             f"{row['avg_loss']:.4f} | {row['avg_rcu']:.0f} | "
                             f"{int(row['n_wins'])}/{int(row['n_tasks'])} | "
                             f"{int(row['n_top3'])}/{int(row['n_tasks'])} | {row['avg_iqr']:.4f} |")
            lines.append("")

    # Interaction table
    lines.append("## 5. Аналіз домінування методів\n")
    lines.append("| Метод | Тип | Сер. ранг | Перемоги% | Toп-3% | Найкраще на | Найгірше на |")
    lines.append("|-------|-----|-----------|-----------|--------|-------------|-------------|")
    for _, row in interaction.iterrows():
        mtype = "Запропоновано" if row["is_author"] else "Базовий"
        best = ", ".join([t.split("/")[-1] for t in row["best_on"]]) if row["best_on"] else "—"
        worst = ", ".join([t.split("/")[-1] for t in row["worst_on"]]) if row["worst_on"] else "—"
        lines.append(f"| {row['method']} | {mtype} | {row['avg_rank']:.2f} | "
                     f"{row['win_pct']:.1f}% | {row['top3_pct']:.1f}% | {best} | {worst} |")
    lines.append("")

    # Author method highlights
    lines.append("## 6. Результати запропонованих методів\n")
    top5_global = global_ranks.head(5).index.tolist()
    author_in_top5 = [m for m in top5_global if m in AUTHOR_METHODS]
    baseline_in_top5 = [m for m in top5_global if m in BASELINE_METHODS]

    lines.append(f"**Топ-5 глобально**: {', '.join(top5_global)}\n")
    lines.append(f"- Запропоновані у Топ-5: **{len(author_in_top5)}** ({', '.join(author_in_top5) or 'немає'})")
    lines.append(f"- Базові у Топ-5: **{len(baseline_in_top5)}** ({', '.join(baseline_in_top5) or 'немає'})")
    lines.append("")

    # Per-tier author highlights
    for tier in TIER_ORDER:
        tier_data = agg[agg["tier"] == tier]
        if tier_data.empty:
            continue
        tier_ranks = tier_data.groupby("method")["rank"].mean().sort_values()
        top3_tier = tier_ranks.head(3).index.tolist()
        author_top3 = [m for m in top3_tier if m in AUTHOR_METHODS]
        tier_name = TIER_DISPLAY_NAMES.get(tier, tier)
        if author_top3:
            lines.append(f"- **{tier}**: Запропоновані у Топ-3: {', '.join(author_top3)}")
    lines.append("")

    # AUCC summary
    if not aucc_df.empty:
        lines.append("## 7. Швидкість збіжності (AUCC — крос-метод нормалізація)\n")
        aucc_summary = aucc_df.groupby("method")["aucc"].mean().sort_values(ascending=False)
        lines.append("| # | Метод | Сер. AUCC (більше = швидше) |")
        lines.append("|---|-------|----------------------------|")
        for i, (method, aucc) in enumerate(aucc_summary.items(), 1):
            bold = "**" if method in AUTHOR_METHODS else ""
            lines.append(f"| {i} | {bold}{method}{bold} | {aucc:.4f} |")
        lines.append("")

    # NFLAttestation
    lines.append("## 8. Підтвердження теореми No Free Lunch\n")
    all_winners = agg[agg["rank"] == 1].groupby("method")["task"].count()
    total_tasks_with_data = agg["task"].nunique()
    lines.append(f"- Усього задач: {total_tasks_with_data}")
    lines.append(f"- Унікальних переможців: {len(all_winners)} із {n_methods} методів")
    if len(all_winners) > 0:
        top_winner = all_winners.idxmax()
        top_win_pct = all_winners.max() / total_tasks_with_data * 100
        lines.append(f"- Найчастіший переможець: **{top_winner}** ({all_winners.max()}/{total_tasks_with_data} = {top_win_pct:.1f}%)")
    lines.append("")
    lines.append("> Жоден метод не домінує на всіх задачах — що відповідає ")
    lines.append("> теоремі No Free Lunch (Wolpert & Macready, 1997).\n")

    # Write
    summary_path = output_dir / "Summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"   Summary → {summary_path.name}")
    return summary_path


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HPO Benchmark — Full Statistical Analysis")
    parser.add_argument("--tier", type=str, default=None, help="Analyze single tier only")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    tiers = [args.tier.upper()] if args.tier else None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  HPO BENCHMARK — FULL STATISTICAL ANALYSIS")
    print("  Protocol: Demšar (2006), Benavoli et al. (2017)")
    print("=" * 70)

    # 1. Load data
    print("\n[1/7] Loading results...")
    df = load_all_results(tiers)
    n_methods = df["method"].nunique()
    n_tasks = df["task"].nunique()
    print(f"  Loaded: {len(df)} records | {n_methods} methods | {n_tasks} tasks")
    for tier in sorted(df["tier"].unique()):
        t_data = df[df["tier"] == tier]
        print(f"    {tier}: {t_data['task'].nunique()} tasks × {t_data['method'].nunique()} methods "
              f"× {t_data.groupby(['task','method'])['seed'].count().median():.0f} seeds")

    # 2. Aggregation
    print("\n[2/7] Computing per-task aggregation...")
    agg = compute_aggregation(df)
    print(f"  Aggregated: {len(agg)} (task × method) cells")

    # 3. AUCC
    print("\n[3/7] Computing convergence speed (AUCC)...")
    aucc_df = compute_aucc(df)
    if not aucc_df.empty:
        print(f"  AUCC computed for {len(aucc_df)} (task × method) pairs")

    # 4. Statistical tests
    print("\n[4/7] Running statistical tests...")

    # Build rank matrices
    friedman_results = {}
    cd_values = {}

    # Global
    rank_pivot_global = agg.pivot_table(index="task", columns="method", values="rank")
    rank_pivot_global = rank_pivot_global.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if rank_pivot_global.shape[0] >= 3 and rank_pivot_global.shape[1] >= 3:
        fr = friedman_test(rank_pivot_global)
        friedman_results["Global (all tiers)"] = fr
        cd = nemenyi_cd(fr["k"], fr["N"])
        cd_values["Global"] = cd
        print(f"  Global Friedman: χ²={fr['statistic']:.2f}, p={fr['p_value']:.2e} "
              f"({' significant' if fr['significant'] else ' not significant'})")
        print(f"  Nemenyi CD = {cd:.2f} (k={fr['k']}, N={fr['N']})")

    # Per-tier
    active_tiers = sorted(df["tier"].unique())
    for tier in active_tiers:
        tier_agg = agg[agg["tier"] == tier]
        rp = tier_agg.pivot_table(index="task", columns="method", values="rank")
        rp = rp.dropna(axis=1, how="all").dropna(axis=0, how="any")
        if rp.shape[0] >= 3 and rp.shape[1] >= 3:
            fr = friedman_test(rp)
            friedman_results[f"Tier {tier}"] = fr
            cd = nemenyi_cd(fr["k"], fr["N"])
            cd_values[tier] = cd
            print(f"  {tier}: χ²={fr['statistic']:.2f}, p={fr['p_value']:.2e}, CD={cd:.2f}")

    # 5. Interaction analysis
    print("\n[5/7] Computing interaction analysis...")
    interaction = compute_interaction_table(agg)
    print(f"  Top-5 by avg rank: {', '.join(interaction.head(5)['method'].tolist())}")

    # 6. Visualizations
    if not args.no_plots:
        print("\n[6/7] Generating visualizations...")

        # CD diagrams
        if "Global (all tiers)" in friedman_results and friedman_results["Global (all tiers)"]["significant"]:
            plot_cd_diagram(
                friedman_results["Global (all tiers)"]["avg_ranks"],
                cd_values["Global"],
                "Діаграма критичних різниць (CD) — Усі класи задач",
                OUTPUT_DIR / "01_CD_Diagram_Global.png"
            )

        for tier in active_tiers:
            key = f"Tier {tier}"
            tier_name = TIER_DISPLAY_NAMES.get(tier, tier)
            if key in friedman_results and friedman_results[key]["significant"]:
                plot_cd_diagram(
                    friedman_results[key]["avg_ranks"],
                    cd_values[tier],
                    f"Діаграма критичних різниць (CD) —\n{tier_name}",
                    OUTPUT_DIR / f"02_CD_Diagram_{tier}.png"
                )

        # Rank heatmap
        plot_rank_heatmap(agg, OUTPUT_DIR / "03_Rank_Heatmap.png")

        # Pareto front
        plot_pareto_front(agg, OUTPUT_DIR / "04_Pareto_Front.png")
        plot_pareto_per_tier(agg, OUTPUT_DIR / "08_Pareto_Per_Tier.png")

        # Convergence curves per tier
        for tier in active_tiers:
            plot_convergence_top5(df, agg, tier, OUTPUT_DIR / f"05_Convergence_{tier}.png")

        # Win-rate matrix
        plot_win_rate_matrix(agg, OUTPUT_DIR / "06_Win_Rate_Matrix.png")

        # Method family boxplot
        plot_method_family_boxplot(agg, OUTPUT_DIR / "07_Family_Boxplot.png")

    else:
        print("\n[6/9] Skipping plots (--no-plots)")

    # 7. Bayesian Signed-Rank Tests (Benavoli et al., 2017)
    print("\n[7/9] Bayesian Signed-Rank Tests (ROPE = 1 rank)...")

    # Determine best author method globally
    global_ranks_sr = agg.groupby("method")["rank"].mean().sort_values()
    best_author = [m for m in global_ranks_sr.index if m in AUTHOR_METHODS][0]
    all_other_methods = [m for m in global_ranks_sr.index if m != best_author]

    bayesian_df = compute_bayesian_pairwise(agg, best_author, all_other_methods)
    if not bayesian_df.empty:
        print(f"\n  Bayesian: {best_author} vs all others (ROPE=1 rank, N=50000 samples):")
        for _, row in bayesian_df.iterrows():
            icon = "" if row["verdict"] == "REF WINS" else "" if row["verdict"] == "EQUIVALENT" else ""
            print(f"    {icon} vs {row['competitor']:15s}: "
                  f"P(better)={row['P(ref_better)']:.3f}, "
                  f"P(equiv)={row['P(equivalent)']:.3f}, "
                  f"P(worse)={row['P(comp_better)']:.3f} → {row['verdict']}")
        bayesian_df.to_csv(OUTPUT_DIR / "bayesian_signed_rank.csv", index=False)
        print(f"   Bayesian results → bayesian_signed_rank.csv")

    # 8. Top-6 per-tier analysis
    print("\n[8/9] Top-6 per-tier focused analysis...")
    tier_top6 = compute_top6_per_tier(agg)
    for tier, t6 in tier_top6.items():
        print(f"  {tier} Top-6: {', '.join(t6.index.tolist())}")
        t6.to_csv(OUTPUT_DIR / f"top6_{tier}.csv")

    # 9. Summary report
    print("\n[9/9] Generating summary report...")
    summary_path = generate_summary(df, agg, friedman_results, cd_values,
                                    interaction, aucc_df, OUTPUT_DIR,
                                    bayesian_df=bayesian_df if not bayesian_df.empty else None,
                                    tier_top6=tier_top6,
                                    best_author=best_author)

    # Save raw data for further analysis
    agg.to_csv(OUTPUT_DIR / "aggregated_results.csv", index=False)
    interaction.to_csv(OUTPUT_DIR / "interaction_analysis.csv", index=False)
    if not aucc_df.empty:
        aucc_df.to_csv(OUTPUT_DIR / "aucc_results.csv", index=False)

    print("\n" + "=" * 70)
    print(f"  ANALYSIS COMPLETE")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  Summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
