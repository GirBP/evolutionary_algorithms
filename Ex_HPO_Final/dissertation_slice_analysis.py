#!/usr/bin/env python3
"""
Зрізи таблиць 3.1 та 3.2 дисертації (§ 3.6) — реконструкція методології.

Ці два зрізи НЕ збігаються з глобальним 11-методним аналізом у
results/GLOBAL_ANALYSIS/ (make_figs.py, figs/01-03): вони відтворюють
підмножини методів і крос-методну нормалізацію, застосовані окремо для
таблиць 3.1 та 3.2 дисертації.

Таблиця 3.1 (8-методний зріз): SACMA-DAC проти 7 базових методів
    (TPE, GP-BO, SMAC, L-SHADE, CMA-ES, DEHB, Random). SACMA-MAB, WL-CMA
    та Sigma-CMA виключені з цього зрізу. Ранг кожного методу на кожній
    задачі рахується ЛИШЕ в межах цих 8 методів (rankdata по median_loss),
    середній ранг — по 43 задачах. Джерело даних:
    results/GLOBAL_ANALYSIS/aggregated_results.csv (median_loss).

Таблиця 3.2 (AUCC, 9-методний зріз): SACMA-DAC + SACMA-MAB + 7 базових
    методів. AUCC (Area Under Convergence Curve) рахується з крос-методною
    нормалізацією кривих збіжності, але межі нормалізації (L_best/L_worst
    на задачу) обчислюються ЛИШЕ по цих 9 методів — інакше, ніж
    results/GLOBAL_ANALYSIS/aucc_results.csv, де межі беруться по всіх 11.
    Тому таблиця 3.2 читає сирі JSON з results/L0, L2, L2_MLP_PD1,
    L3_NAS_SUPER, L4, L5_FCNET (ті самі 6 груп × 43 задачі, що й таблиця
    3.1; L2_WCT і L_ABLATION виключені — допоміжні тіри, не входять у 43).

Методологія AUCC адаптована з /Users/bibo/Desktop/cs_dev/Ex_index/proposals/scripts/analyze_full_filtered.py
(compute_aucc, READ-ONLY першоджерело) — той самий рецепт нормалізації,
застосований до 9-методної підмножини.

Вихід:
    figs/06_diss_table31_ranks_8m.png
    figs/07_diss_table32_aucc_9m.png
    figs/diss_slice_stats.txt

Використання:
    python3 dissertation_slice_analysis.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import studentized_range

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
RESULTS_ROOT = ROOT / "results"
FIGS_DIR = ROOT / "figs"

TIER_ORDER = ["L0", "L2", "L2_MLP_PD1", "L3_NAS_SUPER", "L4", "L5_FCNET"]  # 43 задачі; L2_WCT, L_ABLATION виключені

# Метод-файли -> відображувані назви (як у results/GLOBAL_ANALYSIS/aggregated_results.csv
# та у сирих JSON results/<tier>/*.json, поле "method").
METHOD_NAMES = {
    "sacma_v3": "SACMA-DAC", "sacma_base": "SACMA-base", "sacma_lazy": "SACMA-lazy",
    "sacma_mab": "SACMA-MAB", "antivanila": "Sigma-CMA", "whales_cma": "WL-CMA",
    "iw_moea": "IW-MOEA", "ordinv_cma": "OrdInv-CMA", "cmaes_pure": "CMA-ES",
    "random_search": "Random", "tpe": "TPE", "bo_gp": "GP-BO",
    "smac_method": "SMAC", "dehb_method": "DEHB", "shade": "SHADE", "lshade": "L-SHADE",
}

BASE7 = ["TPE", "GP-BO", "SMAC", "L-SHADE", "CMA-ES", "DEHB", "Random"]
METHODS_8 = ["SACMA-DAC"] + BASE7          # табл. 3.1
METHODS_9 = ["SACMA-DAC", "SACMA-MAB"] + BASE7  # табл. 3.2

# Порядок виводу табл. 3.2 у дисертації (не відсортовано за значенням AUCC).
TABLE32_ORDER = ["SACMA-DAC", "SACMA-MAB", "TPE", "GP-BO", "SMAC", "L-SHADE", "CMA-ES", "DEHB", "Random"]

FLAGSHIP_METHODS = {"SACMA-DAC", "SACMA-MAB"}
COLOR_FLAGSHIP = "#1a5276"
COLOR_BASELINE = "#95a5a6"


def bar_color(method: str) -> str:
    return COLOR_FLAGSHIP if method in FLAGSHIP_METHODS else COLOR_BASELINE


# ═══════════════════════════════════════════════════════════
# Табл. 3.1 — 8-методний зріз рангів (SACMA-DAC + 7 бейзлайнів)
# ═══════════════════════════════════════════════════════════

def compute_table31(agg_csv: Path) -> dict:
    df = pd.read_csv(agg_csv)
    df = df[df["method"].isin(METHODS_8)].copy()

    tasks = sorted(df["task"].unique())
    by_task = {t: g.set_index("method")["median_loss"] for t, g in df.groupby("task")}

    missing = [t for t in tasks if not set(METHODS_8).issubset(by_task[t].index)]
    if missing:
        raise RuntimeError(f"Табл. 3.1: {len(missing)} задач без усіх 8 методів, напр. {missing[:3]}")

    mat = np.array([[by_task[t][m] for m in METHODS_8] for t in tasks])
    ranks = np.apply_along_axis(lambda x: stats.rankdata(x), 1, mat)
    mean_ranks = dict(zip(METHODS_8, ranks.mean(axis=0)))

    chi2, p = stats.friedmanchisquare(*[mat[:, i] for i in range(len(METHODS_8))])
    # компенсація: friedmanchisquare рахує на сирих значеннях, але для χ² статистики
    # тесту Фрідмана результат ідентичний обчисленню на матриці рангів.
    chi2_r, p_r = stats.friedmanchisquare(*[ranks[:, i] for i in range(len(METHODS_8))])

    k, N = len(METHODS_8), len(tasks)
    q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))

    return {
        "n_tasks": N,
        "mean_ranks": mean_ranks,
        "chi2": chi2_r,
        "p": p_r,
        "cd": cd,
        "k": k,
    }


def load_raw_results_9m() -> pd.DataFrame:
    rows = []
    for tier in TIER_ORDER:
        tier_dir = RESULTS_ROOT / tier
        for fp in sorted(tier_dir.glob("*.json")):
            try:
                d = json.load(open(fp))
            except Exception:
                continue
            method_raw = d.get("method", fp.stem.split("__")[0])
            display = METHOD_NAMES.get(method_raw, method_raw)
            if display not in METHODS_9:
                continue
            rows.append({
                "tier": tier,
                "dataset": d.get("dataset", "?"),
                "model": d.get("model", "?"),
                "method": display,
                "seed": d.get("seed", 0),
                "loss": d.get("loss", float("inf")),
                "curve": d.get("curve", []),
            })
    df = pd.DataFrame(rows)
    df["task"] = df["tier"] + "/" + df["dataset"] + "/" + df["model"]
    df = df[df["loss"] < 1e10].copy()
    return df


def compute_aucc_9m(df: pd.DataFrame) -> pd.DataFrame:
    """Крос-методна нормалізація AUCC, межі рахуються ЛИШЕ по 9 методах табл. 3.2.

    Той самий рецепт, що compute_aucc() у analyze_full_filtered.py (READ-ONLY
    першоджерело), застосований до df, вже відфільтрованого до METHODS_9.
    """
    results = []
    task_bounds = {}
    for task, t_grp in df.groupby("task"):
        all_final_losses = t_grp.groupby("method")["loss"].median()
        all_curves = []
        for _, row in t_grp.iterrows():
            if row["curve"] and len(row["curve"]) > 0:
                all_curves.append(row["curve"][0])
        L_best = all_final_losses.min()
        L_worst = max(all_curves) if all_curves else all_final_losses.max()
        if L_worst <= L_best:
            L_worst = all_final_losses.max()
        task_bounds[task] = (L_best, L_worst)

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
            normalized = [max(0, (L_worst - c) / rng) for c in curve]
            normalized = [min(n, 1.0) for n in normalized]
            aucc_vals.append(np.trapezoid(normalized) / (len(normalized) - 1))

        if aucc_vals:
            results.append({"task": task, "method": method, "aucc": np.median(aucc_vals)})
    return pd.DataFrame(results)


def main() -> None:
    FIGS_DIR.mkdir(exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 70)
    log("Зріз табл. 3.1: SACMA-DAC + 7 базових методів, ранги (8-методний)")
    log("=" * 70)
    res31 = compute_table31(GLOBAL_ANALYSIS / "aggregated_results.csv")
    mean_ranks_sorted = pd.Series(res31["mean_ranks"]).sort_values()
    for m, v in mean_ranks_sorted.items():
        log(f"  {m:12s} {v:.3f}")
    log(f"chi2 = {res31['chi2']:.3f}")
    log(f"p = {res31['p']:.3e}")
    log(f"CD(k={res31['k']}, N={res31['n_tasks']}) = {res31['cd']:.3f}")


    log()
    log("=" * 70)
    log("Зріз табл. 3.2: AUCC 9 методів (SACMA-DAC/MAB + 7 базових)")
    log("=" * 70)
    df9 = load_raw_results_9m()
    log(f"завантажено рядків (метод×задача×seed): {len(df9)}, задач: {df9['task'].nunique()}, методів: {sorted(df9['method'].unique())}")
    aucc_df = compute_aucc_9m(df9)
    mean_aucc = aucc_df.groupby("method")["aucc"].mean()

    log()
    log("AUCC за спаданням (обчислено, нормалізація по 9 методах):")
    for m, v in mean_aucc.sort_values(ascending=False).items():
        log(f"  {m:12s} {v:.4f}")


    out_stats = FIGS_DIR / "diss_slice_stats.txt"
    out_stats.write_text("\n".join(lines) + "\n")
    print(f"\nзбережено: {out_stats.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
