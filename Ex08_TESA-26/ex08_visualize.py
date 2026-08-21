# Ex08: Візуалізація результатів бенчмарку
# Оновлено: розділено спільний графік на незалежні, оформлено відповідно до вимог
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
RESULTS_DIR = setup_experiment(EXPERIMENT_DIR)
TABLES_DIR = RESULTS_DIR / "tables"
FIGS_DIR = RESULTS_DIR / "figs"
RAW_DIR = RESULTS_DIR / "raw"
ensure_dir(TABLES_DIR)
ensure_dir(FIGS_DIR)
ensure_dir(RAW_DIR)
DATA_DIR = EXPERIMENT_DIR / "data"

# Map old internal names -> display names (for backward compatibility with existing data)
METHOD_DISPLAY_NAMES = {
    'Evo-SynFlow-K': 'Evo-SynFlow (SymWanda)',
    'Evo-SynFlow-KO': 'Evo-SynFlow (EnergyComp)',
    'Evo-SynFlow-KO-a': 'Evo-SynFlow (Adaptive)',
    'Evo-SynFlow-KO-A': 'Evo-SynFlow (EnergyComp-A)',
    'SoftMask': 'SoftMask',
}

# Кольорова гама (display names)
COLORS = {
    'Base CNN': '#d62728',      # Red
    'Magnitude': '#1f77b4',     # Blue
    'SET': '#9467bd',           # Purple
    'SoftMask': '#bcbd22',      # Olive
    'WANDA-CNN': '#7f7f7f',     # Gray
    'VPAM': '#aec7e8',          # Light blue
    'SoftMask-Grad': '#98df8a', # Light green
    'Evo-SynFlow': '#2ca02c',   # Green (як у Ex08)
    'Evo-SynFlow (SymWanda)': '#ff7f0e',   # Orange
    'Evo-SynFlow (EnergyComp)': '#e377c2', # Pink
    'Evo-SynFlow (EnergyComp-A)': '#17becf', # Cyan
    'Evo-SynFlow (Adaptive)': '#8c564b',   # Brown
    'EvoStruct': '#ff9896',     # Salmon
    'Evo-HMT': '#d62728',        # Red (НОВИЙ)
}

# Labels for plots (Ukrainian / English)
LABELS_UK = {
    'f1_vs_sparsity_title': "F1-міра vs Розрідження",
    'f1_vs_sparsity_x': "Рівень розрідження",
    'f1_vs_sparsity_y': "F1-міра",
    'method': "Метод",
    'time_cost_title': lambda sp: f"Час виконання ({int(sp*100)}% розрідження)",
    'time_cost_y': "Час (с)",
    'time_cost_95_98_title': "Час виконання (відрізок 95–98% розрідження)",
    'f1_at_98_title': "F1-міра при 98% розрідженні",
    'time_at_98_title': "Час виконання при 98% розрідженні",
    'friedman_ylabel': "Метод",
    'friedman_xlabel': "Середній ранг (1 = найкращий)",
    'friedman_title': "Ранжування Фрідмана–Немені (α=0.05)\n(одиниця спостереження: рівень розрідження × seed)",
}
LABELS_EN = {
    'f1_vs_sparsity_title': "F1-score vs Sparsity",
    'f1_vs_sparsity_x': "Sparsity",
    'f1_vs_sparsity_y': "F1-score",
    'method': "Method",
    'time_cost_title': lambda sp: f"Execution time ({int(sp*100)}% sparsity)",
    'time_cost_y': "Time (s)",
    'time_cost_95_98_title': "Execution time (95–98% sparsity range)",
    'f1_at_98_title': "F1-score at 98% sparsity",
    'time_at_98_title': "Execution time at 98% sparsity",
    'friedman_ylabel': "Method",
    'friedman_xlabel': "Mean rank (1 = best)",
    'friedman_title': "Friedman–Nemenyi ranking (α=0.05)\n(unit of observation: sparsity level × seed)",
}

parser = argparse.ArgumentParser(description='Ex08: Візуалізація бенчмарку')
parser.add_argument('--data', '-d', type=str, default=None, help='Шлях до JSON файлу або директорії з per-method JSON')
parser.add_argument('--english', action='store_true', help='Also save copies with English labels to results/en/')
args = parser.parse_args()

# English outputs go to results/en/ subdirectory
RESULTS_DIR_EN = RESULTS_DIR / "en"
if args.english:
    ensure_dir(RESULTS_DIR_EN)

# --- Data loading: scan per-method results_*.json from data/ ---
import json

def _load_per_method_results(data_dir: Path) -> pd.DataFrame:
    """Scan data/results_*.json and merge into single DataFrame."""
    all_rows = []
    for p in sorted(data_dir.glob("results_*.json")):
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_rows.extend(data.get('results', []))
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)

def _load_legacy(data_file: Path) -> tuple:
    """Legacy: load from single JSON file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'final' in data:
        return pd.DataFrame(data['final']), data.get('metadata', {})
    return pd.DataFrame(data), {}

if args.data:
    data_path = Path(args.data)
    if data_path.is_dir():
        df = _load_per_method_results(data_path)
        metadata = {}
    else:
        df, metadata = _load_legacy(data_path)
else:
    # Default: scan data/ directory for per-method files
    df = _load_per_method_results(DATA_DIR)
    metadata = {}
    if df.empty:
        # Fallback: try legacy single-file format
        for legacy in [DATA_DIR / 'ex08_benchmark_data.json', RESULTS_DIR / 'ex08_benchmark_data.json']:
            if legacy.exists():
                df, metadata = _load_legacy(legacy)
                break

if df.empty:
    print("Помилка: не знайдено жодних результатів в data/")
    sys.exit(1)

print(f"Завантажено {len(df)} записів, методи: {sorted(df['Method'].unique())}")

# --- Протокол запуску (run_info.txt) ---
_run_info_lines = [
    "Ex08 — протокол запуску",
    "",
    f"Кількість записів: {len(df)}",
    f"Кількість запусків (seeds): {len(df['Seed'].unique()) if 'Seed' in df.columns else '—'}",
    f"Методи ({len(df['Method'].unique())}): {', '.join(sorted(df['Method'].unique()))}",
    f"Рівні розрідження: {sorted(df['Sparsity'].unique().tolist())}",
    "Датасети: blobs, circles, cnn, moons, resnet, spirals",
    "Архітектура: CNN + ResNet",
]
_anchor = metadata.get('anchor_avg_ms')
if _anchor is not None:
    _run_info_lines.append(f"Еталон часу (с): {_anchor:.6f}")
(RAW_DIR / "ex08_run_info.txt").write_text("\n".join(_run_info_lines), encoding="utf-8")

# Normalize method names for display (backward compatibility)
df['Method'] = df['Method'].map(lambda m: METHOD_DISPLAY_NAMES.get(m, m))

# --- Підготовка даних ---
# Primary cost metric: Time_RCU (if present), fallback to Time
if 'Time_RCU' in df.columns:
    df['Time'] = df['Time_RCU']
    df['Time_sec'] = df['Time_RCU']  # RCU is already dimensionless; use as-is
else:
    anchor_avg_ms = metadata.get('anchor_avg_ms')
    if anchor_avg_ms is not None and anchor_avg_ms > 0:
        df['Time_sec'] = df['Time'] * anchor_avg_ms
    else:
        df['Time_sec'] = df['Time']

df['Efficiency'] = df['F1'] / df['Time'].clip(lower=1e-6) * 100
df_methods = df[df['Method'] != 'Base CNN']
max_sp = df['Sparsity'].max()

# Агрегація для графіків (з урахуванням seeds)
df_agg = df.groupby(['Sparsity', 'Method']).agg({
    'F1': ['mean', 'std'],
    'Time': ['mean', 'std'],
    'Efficiency': ['mean', 'std']
}).reset_index()
df_agg.columns = ['Sparsity', 'Method', 'F1_mean', 'F1_std', 'Time_mean', 'Time_std', 'Efficiency_mean', 'Efficiency_std']

# --- Per-dataset графіки вже згенеровані save_results.py / інтерактивно ---
# Копіюємо їх в figs/ для єдності
import shutil
for existing_png in RESULTS_DIR.glob("*_top5_f1_rcu.png"):
    dst = FIGS_DIR / existing_png.name
    if not dst.exists() or existing_png.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(existing_png, dst)
        print(f"[Копія] {existing_png.name} → figs/")
for existing_png in RESULTS_DIR.glob("*_table.png"):
    dst = FIGS_DIR / existing_png.name
    if not dst.exists() or existing_png.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(existing_png, dst)
        print(f"[Копія] {existing_png.name} → figs/")



# --- 4. Підсумкова таблиця (як у Ex08: без Base CNN, тільки методи з Sparsity) ---
if 'Sparsity' in df.columns and 'Method' in df.columns:
    # Виключаємо Base CNN з таблиці (як у Ex08)
    df_table = df[df['Method'] != 'Base CNN'].copy()
    methods = sorted([m for m in df_table['Method'].unique()])
    sparsities = sorted(df_table['Sparsity'].unique())
    
    agg_dict = {}
    if 'F1' in df_table.columns: agg_dict['F1'] = ['mean', 'std']
    if 'Time' in df_table.columns: agg_dict['Time'] = ['mean', 'std']
    
    if agg_dict:
        df_agg_table = df_table.groupby(['Sparsity', 'Method']).agg(agg_dict).reset_index()
        new_cols = ['Sparsity', 'Method']
        for col in agg_dict.keys():
            new_cols.append(f"{col}_mean")
            new_cols.append(f"{col}_std")
        df_agg_table.columns = new_cols
        
        # Pivot для таблиці
        pivot_f1_mean = df_agg_table.pivot(index='Sparsity', columns='Method', values='F1_mean').reindex(columns=methods) if 'F1_mean' in df_agg_table.columns else None
        pivot_f1_std = df_agg_table.pivot(index='Sparsity', columns='Method', values='F1_std').reindex(columns=methods).fillna(0) if 'F1_std' in df_agg_table.columns else None
        pivot_time_mean = df_agg_table.pivot(index='Sparsity', columns='Method', values='Time_mean').reindex(columns=methods) if 'Time_mean' in df_agg_table.columns else None
        pivot_time_std = df_agg_table.pivot(index='Sparsity', columns='Method', values='Time_std').reindex(columns=methods).fillna(0) if 'Time_std' in df_agg_table.columns else None
        
        # Ранги Фрідмана по кожному Sparsity (блок = Seed)
        friedman_ranks_per_sp = {}
        metric = 'F1'
        if 'Seed' in df_table.columns:
            for sp in sparsities:
                df_sp = df_table[df_table['Sparsity'] == sp]
                piv = df_sp.pivot_table(index='Seed', columns='Method', values=metric)
                piv = piv[[m for m in methods if m in piv.columns]].dropna(how='all')
                if len(piv) >= 2 and piv.shape[1] >= 2:
                    ranks = piv.rank(axis=1, ascending=False, method='average')
                    friedman_ranks_per_sp[sp] = ranks.mean(axis=0).to_dict()
                else:
                    friedman_ranks_per_sp[sp] = {m: np.nan for m in methods}
        
        if friedman_ranks_per_sp:
            pivot_rank = pd.DataFrame(friedman_ranks_per_sp).T.reindex(columns=methods).reindex(sparsities)
        else:
            pivot_rank = pd.DataFrame(index=sparsities, columns=methods)
        
        table_display = pd.DataFrame(index=sparsities)
        
        # F1-міра
        for m in methods:
            if pivot_f1_mean is not None and m in pivot_f1_mean.columns:
                table_display[f"F1-міра: {m}"] = [
                    f"{a:.2f} ± {s:.2f}" if s > 0 else f"{a:.2f}"
                    for a, s in zip(pivot_f1_mean[m], pivot_f1_std[m])
                ]
        
        # RCU
        for m in methods:
            if pivot_time_mean is not None and m in pivot_time_mean.columns:
                table_display[f"RCU: {m}"] = [
                    f"{t:.2f} ± {st:.2f}" if st > 0 else f"{t:.2f}"
                    for t, st in zip(pivot_time_mean[m], pivot_time_std[m])
                ]
        
        # Ранг Фрідмана
        for m in methods:
            if m in pivot_rank.columns:
                table_display[f"Ранг Фрідмана (F1): {m}"] = pivot_rank[m].apply(
                    lambda x: f"{float(x):.2f}" if pd.notna(x) and np.isfinite(x) else "—"
                )
        
        # Час заміру еталона (додаємо як окрему колонку)
        anchor_avg_ms = metadata.get('anchor_avg_ms')
        if anchor_avg_ms is not None:
            table_display["Anchor avg (ms)"] = [f"{anchor_avg_ms:.6f}"] * len(table_display)
        
        table_display.index = [f"{s*100:.0f}%" for s in table_display.index]
        table_display.index.name = "Рівень (%)"
        table_display = table_display.reset_index()
        
        save_table_latex(table_display, RAW_DIR / "ex08_summary.tex", caption="F1-міра, час (еталон) та ранг Фрідмана за F1 по методах (Ex08).")
        save_table_markdown(table_display, TABLES_DIR / "ex08_summary.md")
        print("\n=== Підсумкова таблиця ===")
        print(table_display.to_string(index=False))

# --- 5. Статистичний аналіз (Friedman + Nemenyi з CD та effect size, як у Ex06) ---
# Глобальний Friedman-Nemenyi тест з CD та Kendall's W (effect size)
df_final_fr = df[df['Method'] != 'Base CNN'].copy()
methods_for_stats = sorted([m for m in df_final_fr['Method'].unique()])

if 'Seed' in df_final_fr.columns and len(methods_for_stats) >= 2:
    # Pivot для глобального Friedman тесту (як у Ex06: index=["Sparsity", "Seed"])
    pivot_global = df_final_fr.pivot_table(index=["Sparsity", "Seed"], columns="Method", values="F1")
    methods_for_stats = [m for m in methods_for_stats if m in pivot_global.columns]
    pivot_global = pivot_global[methods_for_stats].dropna(how="all") if methods_for_stats else pivot_global
    
    if len(methods_for_stats) >= 2 and len(pivot_global) >= 2:
        stats_lines = ["Ex08 — статистична оцінка", ""]
        stats_lines.append(f"Метрика: F1. Блок: Sparsity × Seed.")
        stats_lines.append("")
        
        try:
            with np.errstate(invalid="ignore"):
                friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot_global[m].values for m in methods_for_stats])
            if not np.isfinite(friedman_stat):
                friedman_stat, friedman_p = np.nan, np.nan
        except Exception as e:
            friedman_stat, friedman_p = np.nan, np.nan
            stats_lines.append(f"Помилка обчислення тесту Фрідмана: {e}")
        
        # Обчислюємо ранги
        ranks_global = pivot_global.rank(axis=1, ascending=False, method="average")
        mean_rank_global = ranks_global.mean(axis=0).reindex(methods_for_stats)
        
        # Обчислюємо CD (Critical Difference) для Nemenyi test
        k_fr, N_fr = len(methods_for_stats), len(pivot_global)
        q_05_fr = studentized_range.ppf(0.95, k_fr, np.inf)
        CD_fr = q_05_fr * np.sqrt(k_fr * (k_fr + 1) / (6 * N_fr))
        
        # Обчислюємо Kendall's W (effect size)
        kendall_w = friedman_stat / (N_fr * (k_fr - 1)) if (N_fr * (k_fr - 1)) > 0 and np.isfinite(friedman_stat) else np.nan
        
        # Статистичний звіт
        p_str = f"{friedman_p:.4f}" if np.isfinite(friedman_p) else "—"
        stats_lines.append(f"Тест Фрідмана (глобальний):")
        stats_lines.append(f"  χ² = {friedman_stat:.4f}" if np.isfinite(friedman_stat) else "  χ² = —")
        stats_lines.append(f"  p = {p_str}")
        if np.isfinite(friedman_p):
            if friedman_p < 0.05:
                stats_lines.append("  Різниця статистично значуща (p < 0.05).")
            else:
                stats_lines.append("  Різниця не є статистично значущою (p >= 0.05).")
        stats_lines.append("")
        stats_lines.append(f"Пост-хок тест Немені:")
        stats_lines.append(f"  Critical Difference (CD) = {CD_fr:.3f}")
        stats_lines.append(f"  Методи з різницею в рангах менше CD не є статистично значуще різними.")
        stats_lines.append("")
        stats_lines.append(f"Effect size (Kendall's W):")
        stats_lines.append(f"  W = {kendall_w:.3f}" if np.isfinite(kendall_w) else "  W = —")
        if np.isfinite(kendall_w):
            if kendall_w < 0.1:
                stats_lines.append("  Інтерпретація: малий ефект")
            elif kendall_w < 0.3:
                stats_lines.append("  Інтерпретація: середній ефект")
            else:
                stats_lines.append("  Інтерпретація: великий ефект")
        stats_lines.append("")
        stats_lines.append(f"Середні ранги Фрідмана:")
        for m, r in mean_rank_global.items():
            stats_lines.append(f"  {m}: {r:.2f}")
        
        # Візуалізація рангу Фрідмана з CD та effect size
        df_rank_fr = mean_rank_global.reset_index()
        df_rank_fr.columns = ["Method", "Mean rank"]
        df_rank_fr = df_rank_fr.sort_values("Mean rank")
        
        n_repeats_str_uk = f"N = {N_fr} повторів"
        n_repeats_str_en = f"N = {N_fr} repeats"
        for lang_suffix, L in [('', LABELS_UK)] + ([('_en', LABELS_EN)] if args.english else []):
            out_figs_fr = (RESULTS_DIR_EN / "figs") if lang_suffix else FIGS_DIR
            ensure_dir(out_figs_fr)
            # Динамічна висота: 0.45 дюйми на кожен метод + відступи
            n_methods_fr = len(df_rank_fr)
            fig_h = max(6, 1.5 + n_methods_fr * 0.45)
            fig_fr, ax_fr = plt.subplots(figsize=(10, fig_h))
            
            # Чорно-білий палітра (відтінки сірого для B&W друку)
            grays = [plt.cm.Greys(0.3 + 0.5 * i / max(1, n_methods_fr - 1)) for i in range(n_methods_fr)]
            bars = ax_fr.barh(df_rank_fr["Method"], df_rank_fr["Mean rank"], color=grays, edgecolor='black', linewidth=0.5)
            
            for bar, rv in zip(bars, df_rank_fr["Mean rank"]):
                ax_fr.text(float(rv) + 0.15, bar.get_y() + bar.get_height() / 2, f"{rv:.1f}", 
                          va="center", ha="left", fontsize=9, fontweight="bold")
            
            best_rank_fr = df_rank_fr["Mean rank"].iloc[0]
            ax_fr.axvline(best_rank_fr + CD_fr, color="black", linestyle="--", linewidth=1.5, label=f"Еквівалентні (CD = {CD_fr:.2f})")
            
            handles, _ = ax_fr.get_legend_handles_labels()
            handles.append(Line2D([0], [0], color="none", label=f"W = {kendall_w:.3f}" if np.isfinite(kendall_w) else "W = —"))
            handles.append(Line2D([0], [0], color="none", label=n_repeats_str_uk if lang_suffix == '' else n_repeats_str_en))
            handles.append(Line2D([0], [0], color="none", label=f"p = {p_str}"))
            ax_fr.legend(handles=handles, loc="lower right", frameon=True, fontsize=10)
            ax_fr.set_xlim(0, df_rank_fr["Mean rank"].max() + 2.5)
            
            ax_fr.set_ylabel(L['friedman_ylabel'], fontsize=12)
            ax_fr.set_xlabel(L['friedman_xlabel'], fontsize=12)
            ax_fr.set_title(L['friedman_title'], fontsize=12, fontweight="bold", color='black')
            ax_fr.tick_params(axis='y', labelsize=10)
            ax_fr.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            save_figure(fig_fr, out_figs_fr / "ex08_friedman_nemenyi.png")
            plt.close(fig_fr)
            print(f"Збережено: {out_figs_fr / 'ex08_friedman_nemenyi.png'}")
        
        # Save Ranks Table (тільки LaTeX в raw/)
        df_rank_fr.columns = ["Метод", "Середній ранг"]
        # Без лапок
        # df_rank_fr["Метод"] = df_rank_fr["Метод"].apply(lambda x: f'«{x}»')
        save_table_latex(df_rank_fr, RAW_DIR / "ex08_ranks.tex", float_format="%.2f")
        
        (RAW_DIR / "ex08_stats.txt").write_text("\n".join(stats_lines), encoding="utf-8")
        print("\n=== Статистичний звіт ===")
        print("\n".join(stats_lines))


# (Парето-графік для 31 методу — нечитабельний. Per-dataset Top-5 графіки вже є.)

print(f"\nВізуалізація завершена. Результати: графіки в {FIGS_DIR}, таблиці в {TABLES_DIR}")
