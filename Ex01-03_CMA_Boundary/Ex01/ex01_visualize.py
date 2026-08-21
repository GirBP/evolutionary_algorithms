# Ex01: Візуалізація результатів експерименту
# Цей скрипт читає збережені дані та генерує графіки та таблиці.
# Може виконуватися окремо від експерименту для зміни візуалізації без перезапуску.

import sys
import argparse
from pathlib import Path

# Встановлюємо non-interactive backend для швидшого рендерингу без затримок курсора
import matplotlib
matplotlib.use('Agg')

# Налаштування шляху для імпорту common модулів
# (спільний common/ — на корені публічного репозиторію, на рівень вище за
# Ex01-03_CMA_Boundary/; тому тут на один .parent більше, ніж в оригіналі Ex01/)
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
from scipy import stats
from scipy.stats import studentized_range

# Спільні компоненти для всіх експериментів
from common import (
    ensure_dir,
    save_figure,
    save_table_latex,
    save_table_markdown,
    setup_experiment,
    load_experiment_data,
    set_dstu_style,
    legend_outside,
    create_figure,
)
# ДСТУ 3008:2015: графіки та таблиці — Times New Roman, 12 pt
set_dstu_style()

# Ініціалізація експерименту: results/ — графіки + .md таблиці, results/raw/ — .tex, .txt
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = setup_experiment(EXPERIMENT_DIR)
TABLES_DIR = RESULTS_DIR / "tables"
FIGS_DIR = RESULTS_DIR / "figs"
RAW_DIR = RESULTS_DIR / "raw"
ensure_dir(TABLES_DIR)
ensure_dir(FIGS_DIR)
ensure_dir(RAW_DIR)
DATA_DIR = EXPERIMENT_DIR / "data"

# Палітра: Multiple Restarts для обох градієнтних методів (10 restarts кожен)
PALETTE_EX01 = {"AdamW (MR)": "#D32F2F", "AdaBelief (MR)": "#1976D2", "CMA-ES (EA)": "#2E8B57", "Random Search": "#757575"}

# Парсинг аргументів командного рядка
parser = argparse.ArgumentParser(description='Ex01: Візуалізація результатів експерименту')
parser.add_argument('--data', '-d', type=str, default=None,
                    help='Шлях до JSON файлу з даними експерименту. За замовчуванням: автоматичний пошук останнього файлу')
parser.add_argument('--update-summary-only', action='store_true',
                    help='Лише оновити таблицю ex01_summary (2 знаки після коми) з існуючого .tex, без завантаження даних')
args = parser.parse_args()

# Режим "лише оновити таблицю" — без JSON, тільки парсинг існуючого ex01_summary.tex з raw
if args.update_summary_only:
    summary_tex = TABLES_DIR / "ex01_summary.tex"
    if not summary_tex.exists():
        print(f"Помилка: файл {summary_tex} не знайдено. Спочатку згенеруйте таблицю (запустіть експеримент і візуалізацію).")
        sys.exit(1)
    text = summary_tex.read_text(encoding='utf-8')
    # Витягуємо рядки таблиці (між \begin{tabular} і \end{tabular})
    start = text.find("\\begin{tabular}")
    end = text.find("\\end{tabular}")
    if start == -1 or end == -1:
        print("Помилка: не вдалося знайти таблицю у файлі.")
        sys.exit(1)
    block = text[start:end]
    raw_lines = []
    for part in block.split("\\\\"):
        for ln in part.split("\n"):
            raw_lines.append(ln.strip())
    lines = [ln for ln in raw_lines if ln and ln not in ("\\toprule", "\\midrule", "\\bottomrule")
             and not ln.startswith("\\begin{") and not ln.startswith("\\end{")]
    if len(lines) < 2:
        print("Помилка: недостатньо рядків у таблиці.")
        sys.exit(1)
    header = [c.strip() for c in lines[0].split("&")]
    rows = []
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.split("&")]
        row = []
        for c in cells:
            if c in ("---", "—", "–"):
                row.append("---")
            else:
                try:
                    row.append(round(float(c), 2))
                except ValueError:
                    row.append(c)
        rows.append(row)
    df_summary = pd.DataFrame(rows, columns=header)
    save_table_latex(df_summary, RAW_DIR / "ex01_summary.tex", float_format="%.2f")
    save_table_markdown(df_summary, TABLES_DIR / "ex01_summary.md")
    print(f"Таблицю ex01_summary оновлено (raw .tex + .md у results/).")
    sys.exit(0)

# Визначення файлу з даними
if args.data:
    data_file = Path(args.data)
    if not data_file.exists():
        print(f"Помилка: файл {data_file} не знайдено")
        sys.exit(1)
else:
    # Автоматичний пошук: спочатку Ex01/data, потім у всьому проєкті
    data_files = list(DATA_DIR.glob("ex01_data_*.json"))
    if not data_files:
        data_files = list(ROOT.glob("**/ex01_data_*.json"))
    if not data_files:
        print(f"Помилка: не знайдено файлів ex01_data_*.json (ex01_data_quick.json або ex01_data_n*.json) у проєкті.")
        print("  • Швидкий запуск: python ex01_run.py -q")
        print("  • Експериментальний: python ex01_run.py -n")
        print("  • Кастомно N прогонів: python ex01_run.py -n N")
        print("  • Якщо дані в іншому місці: python ex01_visualize.py --data /шлях/до/файл.json")
        print("  • Лише оновити таблицю ex01_summary: python ex01_visualize.py --update-summary-only")
        sys.exit(1)
    data_file = max(data_files, key=lambda p: p.stat().st_mtime)
    print(f"Використовується файл даних: {data_file}")

# Завантаження даних
print(f"Завантаження даних з {data_file}...")
data = load_experiment_data(data_file)

df_conv = data['convergence']
df_final = data['final']
metadata = data['metadata']

N_TRIALS = metadata['N_TRIALS']
L_TARGET = metadata.get('L_TARGET', 10.0)

print(f"Завантажено: {N_TRIALS} запусків")

# Очищення results/ не виконується тут — лише при запуску експерименту (ex01_run) перед viz.
# ==========================================
# ВІЗУАЛІЗАЦІЯ
# ==========================================

# --- Візуалізація ландшафту задачі (1D зріз Rastrigin) ---
fig_landscape, ax_land, _ = create_figure("landscape")
x_dummy = np.linspace(-5.12, 5.12, 1000)
y_dummy = 10 + x_dummy**2 - 10 * np.cos(2 * np.pi * x_dummy)
ax_land.plot(x_dummy, y_dummy, "k-", linewidth=1.5, alpha=0.7)
ax_land.set_title("Ландшафт задачі (1D зріз)", fontsize=12, fontweight="bold")
ax_land.set_xlabel("Простір параметрів", fontsize=12)
ax_land.set_ylabel("Цільова функція", fontsize=12)
ax_land.grid(True, alpha=0.2)
plt.tight_layout()
save_figure(fig_landscape, FIGS_DIR / "ex01_landscape.png")
plt.close(fig_landscape)

# --- Криві збіжності (окремий файл) ---
fig_conv, ax_conv, _ = create_figure("wide")
sns.lineplot(
    data=df_conv,
    x="FE",
    y="Loss",
    hue="Method",
    palette=PALETTE_EX01,
    estimator="median",
    errorbar=("pi", 50),
    ax=ax_conv,
)
ax_conv.set_yscale("log")
ax_conv.set_title("Швидкість збіжності (логарифмічна шкала)", fontsize=12, fontweight="bold")
ax_conv.set_xlabel("Кількість оцінок цільової функції", fontsize=12)
ax_conv.set_ylabel("Втрати / Помилка", fontsize=12)
ax_conv.legend(title="Метод")
ax_conv.grid(True, alpha=0.3)

# Додаємо вертикальні лінії для індикації рестартів у Multiple Restarts методах
MAX_FE = 3000
N_RESTARTS_MR = 10
FE_PER_RESTART = MAX_FE // N_RESTARTS_MR

for restart_idx in range(1, N_RESTARTS_MR):
    restart_fe = restart_idx * FE_PER_RESTART
    ax_conv.axvline(restart_fe, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

ax_conv.text(0.02, 0.98, 'Пунктирні лінії: рестарти\nдля MR методів (кожні 300 FE)',
             transform=ax_conv.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

plt.tight_layout()
save_figure(fig_conv, FIGS_DIR / "ex01_convergence.png")
plt.close(fig_conv)

# --- Розподіл фінальної якості (окремий файл) ---
fig_final, ax_final, _ = create_figure("wide")
sns.boxplot(data=df_final, x="Method", y="Final Loss", hue="Method", palette=PALETTE_EX01, legend=False, ax=ax_final)
ax_final.set_yscale("log")
ax_final.set_title("Розподіл фінальної точності", fontsize=12, fontweight="bold")
ax_final.set_ylabel("Фінальні втрати (логарифмічна шкала)", fontsize=12)
ax_final.set_xlabel("Метод", fontsize=12)
ax_final.tick_params(axis="x", rotation=45)
ax_final.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
save_figure(fig_final, FIGS_DIR / "ex01_final_distribution.png")
plt.close(fig_final)

# --- Розподіл точності залежно від обчислювальної вартості (Loss vs RCU) ---
# Краща область: нижня третина по втратах (quantile 0.33) і нижня третина по RCU.
cost_col_src = "Time_RCU"
cost_label = "Обчислювальна вартість (RCU)"

fig_acc_time, ax_acc_time, rect_acc = create_figure("wide", legend_outside=True)

loss_threshold = df_final["Final Loss"].quantile(0.33)
rcu_threshold = df_final[cost_col_src].quantile(0.33)

ax_acc_time.axhspan(0, loss_threshold, xmin=0, xmax=rcu_threshold/df_final[cost_col_src].max(), 
                     alpha=0.15, color='green', label='Краща область\n(низькі втрати та вартість)')

for method in df_final["Method"].unique():
    method_data = df_final[df_final["Method"] == method]
    ax_acc_time.scatter(
        method_data[cost_col_src],
        method_data["Final Loss"],
        label=method,
        color=PALETTE_EX01.get(method, "gray"),
        alpha=0.7,
        s=60,
        edgecolors='black',
        linewidths=0.5,
    )

ax_acc_time.set_yscale("log")
if "Anchor_avg_ms" in df_final.columns:
    _a = df_final["Anchor_avg_ms"]
    ax_acc_time.set_xlabel(f'{cost_label}\n(Anchor: {_a.mean():.2f}±{_a.std():.2f} мс)', fontsize=12)
else:
    ax_acc_time.set_xlabel(cost_label, fontsize=12)
ax_acc_time.set_ylabel("Фінальні втрати (логарифмічна шкала)", fontsize=12)
ax_acc_time.set_title("Точність vs Обчислювальна вартість\n(Краще: нижчі втрати та вартість → лівий нижній кут)", fontsize=12, fontweight="bold")
legend_outside(ax_acc_time, side="right", title="Метод")
ax_acc_time.grid(True, alpha=0.3)

ax_acc_time.text(0.02, 0.98, '← Менша вартість\n(Краще)', transform=ax_acc_time.transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax_acc_time.text(0.98, 0.02, 'Менші втрати\n(Краще) ↓', transform=ax_acc_time.transAxes,
                 fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout(rect=rect_acc)
save_figure(fig_acc_time, FIGS_DIR / "ex01_accuracy_vs_time.png")
plt.close(fig_acc_time)

# --- Комплексний підхід: Friedman → ранжування → Nemenyi → візуалізація ---
method_list = df_final["Method"].unique().tolist()
k, N = len(method_list), N_TRIALS
pivot_loss = df_final.pivot(index="Trial", columns="Method", values="Final Loss")

friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot_loss[m].values for m in method_list])
ranks = pivot_loss.rank(axis=1, method="average")
mean_rank = ranks.mean(axis=0).reindex(method_list)

# Effect size: Kendall's W (сумарний ефект для тесту Фрідмана), 0 ≤ W ≤ 1
kendall_w = friedman_stat / (N * (k - 1)) if N * (k - 1) > 0 else 0.0


def vargha_delaney_a(x, y):
    """Vargha-Delaney A: ймовірність, що випадкове значення з x менше ніж з y. A > 0.5 ⇒ x краще (нижчі значення)."""
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return 0.5
    count = 0.0
    for xi in x:
        for yj in y:
            if xi < yj:
                count += 1.0
            elif xi == yj:
                count += 0.5
    return count / (n * m)


# Попарний effect size Vargha-Delaney A (по методах)
vda_pairs = []
for i, ma in enumerate(method_list):
    for j, mb in enumerate(method_list):
        if i >= j:
            continue
        a_ab = vargha_delaney_a(pivot_loss[ma].values, pivot_loss[mb].values)
        vda_pairs.append((ma, mb, a_ab))

# Статистичний звіт: Friedman тест + effect size
stats_lines = ["Ex01 — статистична оцінка", ""]
stats_lines.append("Тест Фрідмана (блок = Trial). Метрика: Final Loss (менше = краще).")
stats_lines.append("")
stats_lines.append(f"Friedman χ²={friedman_stat:.4f}, p={friedman_p:.4f}")
stats_lines.append("")
stats_lines.append("Effect size (сумарний): Kendall's W (конкорданс рангів, 0–1):")
stats_lines.append(f"  W = {kendall_w:.4f}")
stats_lines.append("")
stats_lines.append("Effect size (попарно): Vargha-Delaney A (ймовірність, що метод A має нижчу втрату за B; A>0.5 ⇒ A краще):")
for ma, mb, a_ab in vda_pairs:
    stats_lines.append(f"  A({ma}, {mb}) = {a_ab:.4f}")
stats_lines.append("")

q_05 = studentized_range.ppf(0.95, k, np.inf)
CD = q_05 * np.sqrt(k * (k + 1) / (6 * N))
stats_lines.append(f"Пост-хок Немені: CD (α=0.05) = {CD:.4f}")
stats_lines.append("")
(RAW_DIR / "ex01_stats.txt").write_text("\n".join(stats_lines), encoding="utf-8")

df_rank = mean_rank.reset_index()
df_rank.columns = ["Method", "Mean rank"]
df_rank = df_rank.sort_values("Mean rank")
fig_friedman, ax_f, _ = create_figure("friedman")
colors_f = [PALETTE_EX01.get(m, "gray") for m in df_rank["Method"]]
bars = ax_f.barh(df_rank["Method"], df_rank["Mean rank"], color=colors_f)

for i, (bar, rank_val) in enumerate(zip(bars, df_rank["Mean rank"])):
    ax_f.text(rank_val + 0.05, bar.get_y() + bar.get_height()/2, 
              f'{rank_val:.2f}', 
              va='center', ha='left', fontsize=12, fontweight='bold')

ax_f.axvline(CD, color="red", linestyle="--", linewidth=1.5, label=f"CD = {CD:.3f}")
ax_f.set_xlabel("Середній ранг (1 = найкращий)", fontsize=12)
ax_f.set_title("Ранжування Фрідмана та критична різниця Немені (α=0.05)", fontsize=12, fontweight="bold")
handles, _ = ax_f.get_legend_handles_labels()
handles.append(Line2D([0], [0], color="none", label=f"W = {kendall_w:.3f}"))
handles.append(Line2D([0], [0], color="none", label=f"N = {N} повторів"))
handles.append(Line2D([0], [0], color="none", label=f"p = {friedman_p:.4f}"))
ax_f.legend(handles=handles)
ax_f.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
save_figure(fig_friedman, FIGS_DIR / "ex01_friedman_nemenyi.png")
plt.close(fig_friedman)

df_friedman_report = pd.DataFrame({
    "Method": method_list,
    "Mean rank": [mean_rank[m] for m in method_list],
})
df_friedman_report_ukr = df_friedman_report.copy()
df_friedman_report_ukr.columns = ["Метод", "Середній ранг"]

# --- Таблиця результатів ---
df_final_ukr = df_final.copy()
df_final_ukr = df_final_ukr.rename(columns={
    "Method": "Метод",
    "Final Loss": "Фінальні втрати",
    "Time_RCU": "RCU",
})

summary = (
    df_final_ukr.groupby("Метод")
    .agg({
        "Фінальні втрати": ["mean", "std", "min"],
        "RCU": ["mean", "std"],
    })
    .reset_index()
)

summary.columns = ["Метод", "Фінальні втрати (середнє)", "Фінальні втрати (ст. відх.)", "Фінальні втрати (мін.)", "RCU (середнє)", "RCU (ст. відх.)"]

summary_with_ranks = summary.merge(
    df_friedman_report_ukr,
    on="Метод",
    how="left"
)

summary_rounded = summary_with_ranks.round(2)

save_table_latex(summary_rounded.round(2), RAW_DIR / "ex01_summary.tex")
save_table_markdown(summary_rounded, TABLES_DIR / "ex01_summary.md")

# ==========================================
# ГРАФІК: Обчислювальна вартість по методах (RCU)
# ==========================================
print("Генерація графіка обчислювальної вартості (RCU)...")

rcu_agg = df_final.groupby("Method").agg({
    "Time_RCU": ["mean", "std"],
}).reset_index()
rcu_agg.columns = ["Method", "RCU_mean", "RCU_std"]
rcu_agg = rcu_agg.sort_values("RCU_mean")

fig_rcu, ax_rcu, _ = create_figure("wide")
x_pos = np.arange(len(rcu_agg))
bar_width = 0.5

bars = ax_rcu.bar(x_pos, rcu_agg["RCU_mean"], bar_width,
                  yerr=rcu_agg["RCU_std"], capsize=4,
                  color=[PALETTE_EX01.get(m, "gray") for m in rcu_agg["Method"]],
                  edgecolor="black", linewidth=0.5, alpha=0.85)

for bar, val, std in zip(bars, rcu_agg["RCU_mean"], rcu_agg["RCU_std"]):
    ax_rcu.text(bar.get_x() + bar.get_width()/2., val + std + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax_rcu.set_xticks(x_pos)
ax_rcu.set_xticklabels(rcu_agg["Method"], rotation=30, ha="right")
ax_rcu.set_ylabel("Обчислювальна вартість (RCU)", fontsize=12)
if "Anchor_avg_ms" in df_final.columns:
    _a = df_final["Anchor_avg_ms"]
    ax_rcu.set_title(f'Середня обчислювальна вартість методів (RCU)\nAnchor: {_a.mean():.2f}±{_a.std():.2f} мс',
                     fontsize=12, fontweight='bold')
else:
    ax_rcu.set_title("Середня обчислювальна вартість методів (RCU)", fontsize=12, fontweight="bold")
ax_rcu.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
save_figure(fig_rcu, FIGS_DIR / "ex01_computational_cost.png")
plt.close(fig_rcu)
print(f"  → ex01_computational_cost.png")

# Anchor статистика до stats файлу
if "Anchor_avg_ms" in df_final.columns:
    _a = df_final["Anchor_avg_ms"]
    _mean, _std = _a.mean(), _a.std()
    _cv = (_std / _mean * 100) if _mean > 0 else 0.0
    anchor_stats_lines = [
        "", "Anchor RCU Stability:",
        f"  Global mean: {_mean:.3f} ms",
        f"  Global std:  {_std:.3f} ms",
        f"  CV:          {_cv:.1f}%",
        f"  N samples:   {len(_a)}", "",
    ]
    stats_file = TABLES_DIR / "ex01_stats.txt"
    if stats_file.exists():
        existing = stats_file.read_text(encoding='utf-8')
        stats_file.write_text(existing + "\n".join(anchor_stats_lines), encoding='utf-8')

print(f"\nВізуалізація завершена. Результати: графіки в {FIGS_DIR}, таблиці в {TABLES_DIR}")
