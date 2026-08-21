# Ex03: Візуалізація результатів експерименту
# Читає збережені дані з JSON, будує графіки та таблиці, зберігає в results/.

import sys
import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Спільний common/ — на корені публічного репозиторію (на рівень вище за
# Ex01-03_CMA_Boundary/; тому тут на один .parent більше, ніж в оригіналі Ex03/)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d
from scipy import stats
from scipy.stats import studentized_range
import torch

from common import (
    create_figure,
    ensure_dir,
    save_figure,
    save_table_latex,
    save_table_markdown,
    setup_experiment,
    load_experiment_data,
    set_dstu_style,
)
# ДСТУ 3008:2015: графіки та таблиці — Times New Roman, 12 pt
set_dstu_style()

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR / "code"))
from ex03 import build_model, get_data, CONFIG, DEVICE

RESULTS_DIR = setup_experiment(EXPERIMENT_DIR)
TABLES_DIR = RESULTS_DIR / "tables"
FIGS_DIR = RESULTS_DIR / "figs"
RAW_DIR = RESULTS_DIR / "raw"
ensure_dir(TABLES_DIR)
ensure_dir(FIGS_DIR)
ensure_dir(RAW_DIR)
DATA_DIR = EXPERIMENT_DIR / "data"

parser = argparse.ArgumentParser(description="Ex03: Візуалізація результатів")
parser.add_argument("--data", "-d", type=str, default=None,
                    help="Шлях до JSON з даними. За замовчуванням: останній ex03_data_*.json")
args = parser.parse_args()

if args.data:
    data_file = Path(args.data)
    if not data_file.exists():
        print(f"Файл не знайдено: {data_file}")
        sys.exit(1)
else:
    data_files = list(DATA_DIR.glob("ex03_data_*.json"))
    if not data_files:
        print("Немає даних. Запустіть: python ex03_run_test.py (швидкий) або ex03_run_experiment.py (експериментальний) або ex03_run_experiment.py N")
        sys.exit(1)
    data_file = max(data_files, key=lambda p: p.stat().st_mtime)
    print(f"Використовується: {data_file}")

data = load_experiment_data(data_file)
df_conv = data["convergence"]
df_final = data["final"]
metadata = data["metadata"]

n_runs = metadata["n_runs"]
seed_base = metadata["seed_base"]
# Використовуємо common_time_grid з metadata як fallback, але перебудовуємо з реальних даних конвергенції (тепер в RCU)
if "Time" in df_conv.columns and not df_conv.empty:
    t_max = df_conv["Time"].max()
    common_time_grid = np.linspace(0, t_max, 300) if t_max > 0 else np.array(metadata["common_time_grid"])
else:
    common_time_grid = np.array(metadata["common_time_grid"])

snapshots_run0 = metadata.get("snapshots_run0", {})
datasets = metadata.get("datasets", ["moons"])
methods = metadata.get("methods", ["Adam", "CMA-ES"])

# Протокол запуску (ПРАВИЛА_ЕКСПЕРИМЕНТІВ п. 4.9): щоб при оформленні дисертації не вичитувати код
_run_info_lines = [
    "Ex03 — протокол запуску (конфігурація, з якої побудовані графіки та таблиці)",
    "",
    f"Файл даних: {data_file.name}",
    f"Режим: {metadata.get('config_mode', '—')}",
    f"Кількість прогонів на метод (n_runs): {n_runs}",
    f"Бюджет RCU на run: {metadata.get('rcu_budget', '—')}",
    f"Набори даних: {datasets}",
    f"Методи: {methods}",
    f"seed_base: {seed_base}",
]
(RAW_DIR / "ex03_run_info.txt").write_text("\n".join(_run_info_lines), encoding="utf-8")

# Палітра для всіх методів
_method_order = ["Adam", "CMA-ES", "L-SHADE", "CLPSO", "RTS", "RS"]
_pal = sns.color_palette("tab10", len(_method_order))
THEMES = {m: {"main_color": _pal[i], "cmap": "viridis", "pt_c1": "#333", "pt_c0_edge": _pal[i]} for i, m in enumerate(_method_order)}
for m in methods:
    if m not in THEMES:
        THEMES[m] = {"main_color": "#888", "cmap": "viridis", "pt_c1": "#333", "pt_c0_edge": "#888"}

plt.style.use("seaborn-v0_8-whitegrid")
# Додаткові параметри (фоновий стиль уже задано через set_dstu_style: 12 pt, Times New Roman)
plt.rcParams.update({
    "figure.figsize": (9, 7),
    "lines.linewidth": 1.5,
})


def _json_to_state_dict(sd_lists):
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in sd_lists.items()}


# --- 1. Криві збіжності (Accuracy vs Time): окремий файл на кожен набір даних ---
for dataset_name in datasets:
    fig1, ax, rect = create_figure("wide", legend_outside=True)
    sub_conv = df_conv[df_conv["Dataset"] == dataset_name] if "Dataset" in df_conv.columns else df_conv
    for method in methods:
        sub = sub_conv[sub_conv["Method"] == method]
        if sub.empty:
            continue
        interp_accs = []
        for run_id in sub["Run"].unique():
            r = sub[sub["Run"] == run_id].sort_values("Time")
            t, a = r["Time"].values, r["Accuracy"].values
            if len(t) < 2:
                interp_accs.append(np.full_like(common_time_grid, a[0] if len(a) else 0.5))
            else:
                f = interp1d(t, a, kind="previous", bounds_error=False, fill_value=(0.5, a[-1]))
                interp_accs.append(f(common_time_grid))
        if not interp_accs:
            continue
        arr = np.array(interp_accs)
        mu, sigma = arr.mean(axis=0), arr.std(axis=0)
        c = THEMES.get(method, {}).get("main_color", "#888")
        ax.plot(common_time_grid, mu, label=method, color=c, lw=1.5)
        ax.fill_between(common_time_grid, mu - sigma, mu + sigma, color=c, alpha=0.2)
    ax.set_title(f"Набір даних «{dataset_name}»", fontweight="bold")
    ax.set_xlabel("RCU")
    ax.set_ylabel("Точність")
    
    cur_ylim = ax.get_ylim()
    ax.set_ylim(bottom=max(0.0, cur_ylim[0] - 0.02), top=min(1.05, cur_ylim[1] + 0.02))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    # Анотація з anchor та бюджетом у секундах
    anchor_ms = metadata.get("anchor_avg_ms", 0)
    rcu_budget = metadata.get("rcu_budget", 0)
    budget_sec = anchor_ms / 1000.0 * rcu_budget if anchor_ms else 0
    info_text = f"1 anchor ≈ {anchor_ms:.1f} мс\nБюджет: {rcu_budget} RCU ≈ {budget_sec:.1f} с"
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=10,
            va="bottom", ha="right", color="#555",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.8))
    if rect:
        plt.tight_layout(rect=rect)
    else:
        plt.tight_layout()
    safe_name = dataset_name.replace(" ", "_")
    save_figure(fig1, FIGS_DIR / f"ex03_01_convergence_{safe_name}.png")
    plt.close(fig1)

# --- 2. Границі рішень (moons: Adam, CMA-ES, RTS) ---
boundary_methods = [m for m in ["Adam", "CMA-ES", "RTS"] if snapshots_run0 and m in snapshots_run0]
if len(boundary_methods) >= 2:
    (_, _), (_, _), (_, _), data_np, n_feat, n_cl = get_data("moons", seed_base)
    X_test_np, y_test_np = data_np
    x_min, x_max = X_test_np[:, 0].min() - 0.5, X_test_np[:, 0].max() + 0.5
    y_min, y_max = X_test_np[:, 1].min() - 0.5, X_test_np[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)
    X_0 = X_test_np[y_test_np == 0]
    X_1 = X_test_np[y_test_np == 1]
    stages = ["Start", "Mid", "Final"]
    stages_ukr = ["Початок", "Середина", "Фінал"]

    n_rows = len(boundary_methods)
    fig2 = plt.figure(figsize=(9, 2.5 * n_rows), constrained_layout=True)
    gs = fig2.add_gridspec(n_rows + 1, 3, height_ratios=[0.4] + [1] * n_rows)

    def plot_boundary(ax, state_dict, title_ukr, algo_name, show_ylabel=False):
        theme = THEMES.get(algo_name, {})
        model = build_model("moons", 2, 2).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        with torch.no_grad():
            Z = model(grid_tensor).reshape(xx.shape).cpu().numpy()
        cmap = theme.get("cmap", "viridis")
        pt_c0 = theme.get("pt_c0_edge", "#666")
        pt_c1 = theme.get("pt_c1", "#333")
        ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.15, levels=20, vmin=0, vmax=1)
        ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=1.0)
        ax.scatter(X_0[:, 0], X_0[:, 1], c="white", edgecolors=pt_c0, linewidth=0.8, s=15)
        ax.scatter(X_1[:, 0], X_1[:, 1], c=pt_c1, edgecolors="white", linewidth=0.8, s=15)
        ax.set_title(title_ukr, fontsize=11)
        if show_ylabel:
            ax.set_ylabel(algo_name, fontweight="bold", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    for row, algo_name in enumerate(boundary_methods):
        for i, stage in enumerate(stages):
            key = "End" if stage == "Final" else stage
            ax = fig2.add_subplot(gs[row + 1, i])
            plot_boundary(ax, _json_to_state_dict(snapshots_run0[algo_name][key]), stages_ukr[i], algo_name, show_ylabel=(i == 0))

    fig2.suptitle("Границі рішень (Moons): " + ", ".join(boundary_methods), fontsize=11, fontweight="bold")
    save_figure(fig2, FIGS_DIR / "ex03_02_decision_boundaries.png")
    plt.close(fig2)

# df_plot для таблиць і heatmap
df_plot = df_final.drop(columns=["Run"], errors="ignore")
colors = {m: THEMES.get(m, {}).get("main_color", "#888") for m in methods}

# --- 3a. Heatmap F1-Score (Dataset × Method) ---
if "Dataset" in df_plot.columns and "Method" in df_plot.columns and "F1-Score" in df_plot.columns and len(datasets) > 0 and len(methods) > 0:
    heat_pivot = df_plot.groupby(["Dataset", "Method"])["F1-Score"].mean().unstack(fill_value=0)
    heat_pivot = heat_pivot.reindex(index=datasets, columns=[m for m in methods if m in heat_pivot.columns], fill_value=0)
    fig_heat, ax_heat = plt.subplots(figsize=(max(6, len(methods) * 1.2), max(3, len(datasets) * 0.8)))
    sns.heatmap(heat_pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax_heat, cbar_kws={"label": "F1-міра (середнє)"})
    ax_heat.collections[0].set_alpha(0.9)
    ax_heat.set_title("F1-міра (середнє) — набір даних × метод")
    ax_heat.set_xlabel("Метод")
    ax_heat.set_ylabel("Набір даних")
    save_figure(fig_heat, FIGS_DIR / "ex03_04_heatmap_f1.png")
    plt.close(fig_heat)

# --- 3b. Сумарний графік: Якість vs вартість — по методу середнє (кулька) ± відхилення ---
if "F1-Score" in df_final.columns and ("Time_RCU" in df_final.columns or "Time (s)" in df_final.columns or "NFE" in df_final.columns):
    fig_qc, ax_qc, rect_qc = create_figure("wide", legend_outside=True)
    x_col = "Time_RCU" if "Time_RCU" in df_final.columns else ("Time (s)" if "Time (s)" in df_final.columns else "NFE")
    x_label = "RCU" if x_col == "Time_RCU" else ("Час (с)" if x_col == "Time (s)" else "NFE")
    df_qc = df_final.dropna(subset=["F1-Score", x_col])
    if not df_qc.empty:
        t_max = df_qc[x_col].max()
        if t_max > 0:
            time_q = df_qc[x_col].quantile(0.33)
            f1_q = df_qc["F1-Score"].quantile(0.67)
            ax_qc.fill([0, time_q, time_q, 0], [f1_q, f1_q, 1.0, 1.0], color="green", alpha=0.15, label="Краща область")
        agg = df_qc.groupby("Method").agg({"F1-Score": ["mean", "std"], x_col: ["mean", "std"]})
        for method in agg.index:
            c = THEMES.get(method, {}).get("main_color", "gray")
            mx = agg.loc[method, (x_col, "mean")]
            my = agg.loc[method, ("F1-Score", "mean")]
            sx = float(agg.loc[method, (x_col, "std")]) if pd.notna(agg.loc[method, (x_col, "std")]) else 0.0
            sy = float(agg.loc[method, ("F1-Score", "std")]) if pd.notna(agg.loc[method, ("F1-Score", "std")]) else 0.0
            ax_qc.errorbar(mx, my, xerr=sx, yerr=sy, fmt="o", color=c, capsize=4, capthick=1.5, markersize=10, label=method)
    ax_qc.set_xlabel(x_label, fontweight="bold")
    ax_qc.set_ylabel("F1-міра (якість)", fontweight="bold")
    ax_qc.set_title("Якість vs обчислювальна вартість\n(середнє ± ст.в. по методу; краще — верхній лівий кут)", fontweight="bold")
    ax_qc.legend(title="Метод", loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True)
    cur_ylim = ax_qc.get_ylim()
    ax_qc.set_ylim(bottom=max(0.0, cur_ylim[0] - 0.05), top=min(1.05, cur_ylim[1] + 0.05))
    ax_qc.grid(True, alpha=0.3)
    ax_qc.text(0.02, 0.98, f"← Менший {x_label}\n(Краще)", transform=ax_qc.transAxes, fontsize=10, verticalalignment="top", horizontalalignment="left", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax_qc.text(0.98, 0.02, "Менші втрати / Вища F1-міра\n(Краще) ↓", transform=ax_qc.transAxes, fontsize=10, verticalalignment="bottom", horizontalalignment="right", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    if rect_qc:
        plt.tight_layout(rect=rect_qc)
    else:
        plt.tight_layout()
    save_figure(fig_qc, FIGS_DIR / "ex03_05_quality_vs_cost.png")
    plt.close(fig_qc)

# --- 3d. Ранжування Фрідмана та критична різниця Немені (усі датасети та runs) ---
if "Dataset" in df_final.columns and "Run" in df_final.columns and "Method" in df_final.columns and "F1-Score" in df_final.columns:
    pivot_f1 = df_final.pivot_table(index=["Dataset", "Run"], columns="Method", values="F1-Score")
    method_list_f = [c for c in _method_order if c in pivot_f1.columns]
    if len(method_list_f) < 2:
        method_list_f = list(pivot_f1.columns)
    if len(method_list_f) >= 2 and len(pivot_f1) >= 2:
        pivot_f1 = pivot_f1[method_list_f].dropna(how="all")
        if len(pivot_f1) >= 2:
            try:
                friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot_f1[m].values for m in method_list_f])
            except Exception:
                friedman_stat, friedman_p = np.nan, np.nan
            ranks = pivot_f1.rank(axis=1, ascending=False, method="average")
            mean_rank = ranks.mean(axis=0).reindex(method_list_f)
            k, N = len(method_list_f), len(pivot_f1)
            q_05 = studentized_range.ppf(0.95, k, np.inf)
            CD = q_05 * np.sqrt(k * (k + 1) / (6 * N))
            kendall_w = friedman_stat / (N * (k - 1)) if (N * (k - 1)) > 0 and np.isfinite(friedman_stat) else np.nan
            df_rank = mean_rank.reset_index()
            df_rank.columns = ["Method", "Mean rank"]
            df_rank = df_rank.sort_values("Mean rank")
            fig_fr, ax_fr, _ = create_figure("friedman")
            colors_fr = [THEMES.get(m, {}).get("main_color", "gray") for m in df_rank["Method"]]
            bars = ax_fr.barh(df_rank["Method"], df_rank["Mean rank"], color=colors_fr)
            for bar, rv in zip(bars, df_rank["Mean rank"]):
                ax_fr.text(rv + 0.05, bar.get_y() + bar.get_height() / 2, f"{rv:.2f}", va="center", ha="left", fontsize=12, fontweight="bold")
            best_rank = df_rank["Mean rank"].iloc[0]
            ax_fr.axvline(best_rank + CD, color="red", linestyle="--", linewidth=1.5, label=f"Еквівалентні (CD = {CD:.3f})")
            handles, _ = ax_fr.get_legend_handles_labels()
            handles.append(Line2D([0], [0], color="none", label=f"W = {kendall_w:.3f}" if np.isfinite(kendall_w) else "W = —"))
            handles.append(Line2D([0], [0], color="none", label=f"N = {N} повторів"))
            p_str = f"{friedman_p:.4f}" if np.isfinite(friedman_p) else "—"
            handles.append(Line2D([0], [0], color="none", label=f"p = {p_str}"))
            ax_fr.legend(handles=handles, loc="lower right", frameon=True, fontsize=12)
            ax_fr.set_xlim(0, df_rank["Mean rank"].max() + 0.9)
            ax_fr.set_ylabel("Метод", fontsize=12)
            ax_fr.set_xlabel("Середній ранг (1 = найкращий)", fontsize=12)
            ax_fr.set_title("Ранжування Фрідмана та критична різниця Немені (α=0.05)\n(усі набори даних та запуски)", fontsize=12, fontweight="bold")
            ax_fr.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            save_figure(fig_fr, FIGS_DIR / "ex03_06_friedman_nemenyi.png")
            plt.close(fig_fr)

# --- 4. Підсумкова таблиця (mean ± std): 2 знаки після коми ---
def _fmt_table(val, decimals=2):
    """Два знаки після коми; для значень після коми двох розрядів достатньо."""
    if pd.isna(val):
        return "—"
    return f"{float(val):.{decimals}f}"

columns = ["NFE", "Accuracy", "F1-Score", "ROC-AUC", "Log-Loss", "Time_RCU"]
cols_present = [c for c in columns if c in df_plot.columns]
if "Dataset" in df_plot.columns and len(datasets) > 1:
    summary = df_plot.groupby(["Dataset", "Method"])[cols_present].agg(["mean", "std"])
    final_table = pd.DataFrame()
    for col in cols_present:
        if col == "NFE":
            # NFE у тисячах
            final_table[col] = summary[col].apply(
                lambda r: f"{r['mean']/1000:.2f} ± {r['std']/1000:.2f}" if pd.notna(r.get("std")) and r.get("std", 0) > 0 else _fmt_table(r["mean"]/1000),
                axis=1
            )
        else:
            final_table[col] = summary[col].apply(
                lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}" if pd.notna(r.get("std")) and r.get("std", 0) > 0 else _fmt_table(r["mean"]),
                axis=1
            )
    final_table = final_table.reset_index()
    # Ранжування Фрідмана по кожному набору даних: у кожному Run ранг 1 = найкращий F1, середній ранг по Run
    if "Run" in df_final.columns and "F1-Score" in df_final.columns:
        rank_col = []
        for _, row in final_table.iterrows():
            ds, method = row["Dataset"], row["Method"]
            d = df_final[df_final["Dataset"] == ds]
            piv = d.pivot_table(index="Run", columns="Method", values="F1-Score")
            if piv.empty or method not in piv.columns:
                rank_col.append(np.nan)
                continue
            ranks = piv.rank(axis=1, ascending=False)
            mean_rank = ranks.mean(axis=0)
            rank_col.append(round(mean_rank.get(method, np.nan), 2))
        final_table["Ранг (Фрідман)"] = rank_col
        cols_order = [c for c in final_table.columns if c != "Ранг (Фрідман)"]
        idx = cols_order.index("Method") + 1
        final_table = final_table[cols_order[:idx] + ["Ранг (Фрідман)"] + cols_order[idx:]]
    if "NFE" in final_table.columns:
        final_table = final_table.rename(columns={"NFE": "NFE (тис.)"})
else:
    summary = df_plot.groupby("Method")[cols_present].agg(["mean", "std"])
    final_table = pd.DataFrame()
    for col in cols_present:
        if col == "NFE":
            final_table[col] = summary[col].apply(
                lambda r: f"{r['mean']/1000:.2f} ± {r['std']/1000:.2f}" if pd.notna(r.get("std")) and r.get("std", 0) > 0 else _fmt_table(r["mean"]/1000),
                axis=1
            )
        else:
            final_table[col] = summary[col].apply(
                lambda r: f"{r['mean']:.2f} ± {r['std']:.2f}" if pd.notna(r.get("std")) and r.get("std", 0) > 0 else _fmt_table(r["mean"]),
                axis=1
            )
    final_table = final_table.reset_index()
    if "NFE" in final_table.columns:
        final_table = final_table.rename(columns={"NFE": "NFE (тис.)"})
_col_ua = {
    "Dataset": "Набір даних",
    "Method": "Метод",
    "Accuracy": "Точність",
    "F1-Score": "F1-міра",
    "ROC-AUC": "ROC-AUC",
    "Log-Loss": "Логарифмічні втрати",
    "Time_RCU": "RCU (середнє ± ст. відх.)",
}
if "Dataset" in final_table.columns and len(datasets) > 1:
    for ds in datasets:
        tbl = final_table[final_table["Dataset"] == ds].drop(columns=["Dataset"], errors="ignore")
        tbl = tbl.rename(columns={c: _col_ua[c] for c in _col_ua if c in tbl.columns})
        safe_name = ds.replace(" ", "_")
        save_table_latex(tbl, RAW_DIR / f"ex03_table_summary_{safe_name}.tex")
        save_table_markdown(tbl, TABLES_DIR / f"ex03_table_summary_{safe_name}.md")
else:
    display_table = final_table.copy()
    display_table = display_table.rename(columns={c: _col_ua[c] for c in _col_ua if c in display_table.columns})
    save_table_latex(display_table, RAW_DIR / "ex03_table_summary.tex")
    save_table_markdown(display_table, TABLES_DIR / "ex03_table_summary.md")

# --- 5. Статистика: Friedman тест ---
def _effect_size_rank_biserial(u1, n1, n2):
    if n1 * n2 <= 0:
        return 0.0
    u = min(u1, n1 * n2 - u1)
    return 1.0 - (2.0 * u) / (n1 * n2)


def _vargha_delaney_a(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.5
    try:
        u_stat, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
        return u_stat / (len(a) * len(b))
    except Exception:
        return 0.5


def _cohens_d(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    try:
        mean_a, mean_b = np.mean(a), np.mean(b)
        std_a, std_b = np.std(a, ddof=1), np.std(b, ddof=1)
        n_a, n_b = len(a), len(b)
        pooled = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
        if pooled == 0:
            return 0.0
        return (mean_a - mean_b) / pooled
    except Exception:
        return 0.0


stats_lines = ["Ex03 — статистична оцінка", ""]
df_for_stats = df_final if "Run" in df_final.columns else df_plot
if len(methods) >= 3 and "Dataset" in df_for_stats.columns and "Run" in df_for_stats.columns:
    from scipy.stats import friedmanchisquare
    stats_lines.append("Тест Фрідмана (блок = Run).")
    stats_lines.append("")
    for ds in datasets:
        d = df_for_stats[df_for_stats["Dataset"] == ds]
        if d.groupby("Run").ngroups < 2:
            continue
        piv = d.pivot_table(index="Run", columns="Method", values="F1-Score" if "F1-Score" in d.columns else "Accuracy")
        piv = piv.dropna(how="all")
        if piv.shape[1] < 2:
            continue
        try:
            stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
            stats_lines.append(f"  {ds} (F1-Score): Friedman χ²={stat:.4f}, p={p:.4f}")
        except Exception:
            pass
        
        # Додавання CD, середніх рангів
        k_ds = piv.shape[1]
        n_ds = piv.shape[0]
        q_05_ds = studentized_range.ppf(0.95, k_ds, np.inf)
        cd_ds = q_05_ds * np.sqrt(k_ds * (k_ds + 1) / (6 * n_ds))
        
        ranks_ds = piv.rank(axis=1, ascending=False, method="average")
        mean_rank_ds = ranks_ds.mean(axis=0).sort_values()
        
        stats_lines.append(f"  Пост-хок Немені (CD, α=0.05): {cd_ds:.4f}")
        stats_lines.append("  Середні ранги:")
        for m, r in mean_rank_ds.items():
            stats_lines.append(f"    {m}: {r:.2f}")
        stats_lines.append("")

    stats_lines.append("")
else:
    stats_lines.append("Mann-Whitney U (два методи). Effect size: Vargha-Delaney A, Cohen's d.")
    stats_lines.append("")
    for metric in ["F1-Score", "Time_RCU"]:
        if metric not in df_plot.columns or len(methods) != 2:
            continue
        a = df_plot[df_plot["Method"] == methods[0]][metric].values
        b = df_plot[df_plot["Method"] == methods[1]][metric].values
        if len(a) < 2 or len(b) < 2:
            continue
        u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        vd_a = _vargha_delaney_a(a, b)
        cohens_d_val = _cohens_d(a, b)
        stats_lines.append(f"{metric}: p={p_val:.4f}, Vargha-Delaney A={vd_a:.4f}, Cohen's d={cohens_d_val:.4f}")
    stats_lines.append("")

stats_path = RAW_DIR / "ex03_stats.txt"
stats_path.write_text("\n".join(stats_lines), encoding="utf-8")
print(f"\nВізуалізація завершена. Результати: графіки в {FIGS_DIR}, таблиці в {TABLES_DIR}")
print("Статистика:", stats_path)
