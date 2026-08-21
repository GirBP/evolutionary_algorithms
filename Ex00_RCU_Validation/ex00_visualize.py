#!/usr/bin/env python3
"""
Візуалізація результатів Ex00: RCU Metric Validation.
Читає CSV з results/ і генерує графіки. Можна змінювати без перезапуску експерименту.
Запуск: python Ex00/visualize.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import save_figure, set_dstu_style, ensure_dir

set_dstu_style()

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "figs"

N_BOOTSTRAP = 10000

# ============================================================
# МІТКИ ТА КОЛЬОРИ (єдиний стиль для всього проєкту)
# ============================================================

METRIC_LABELS = {
    "RCU": "RCU",
    "Wall_ms": "Реальний час (мс)",
    "Process_ms": "Процесорний час (мс)",
    "Thread_ms": "Час потоку (мс)",
}
METRIC_COLORS = {
    "RCU": "#2E7D32",
    "Wall_ms": "#1565C0",
    "Process_ms": "#E65100",
    "Thread_ms": "#6A1B9A",
}

CONDITIONS = [
    ("Чисто", None, 0),
    ("CPU шум (4×)", "cpu", 4),
    ("MEM шум (4×)", "mem", 4),
    ("Mixed шум (4×)", "mixed", 4),
    ("Heavy (8×CPU)", "cpu", 8),
]

# ============================================================
# СТАТИСТИКА
# ============================================================

def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=0.95):
    data = np.asarray(data)
    means = np.array([np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / max(pooled, 1e-12)

def vargha_delaney_a(x, y):
    nx, ny = len(x), len(y)
    r = sp_stats.rankdata(np.concatenate([x, y]))
    return (r[:nx].sum() / nx - (nx + 1) / 2) / ny

def cv(s):
    m = s.mean()
    return s.std() / max(abs(m), 1e-12) * 100

# ============================================================
# ГРАФІКИ
# ============================================================

def plot_drift_bars(df):
    """Дрифт (зсув середнього) кожної метрики під стресом відносно чистих умов."""
    metrics = ["RCU", "Wall_ms", "Process_ms", "Thread_ms"]
    stress_conds = [c[0] for c in CONDITIONS if c[1] is not None]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stress_conds))
    n_metrics = len(metrics)
    w = 0.8 / n_metrics

    for j, m in enumerate(metrics):
        clean_mean = df[df["Condition"] == "Чисто"][m].mean()
        drifts = []
        for cond in stress_conds:
            noisy_mean = df[df["Condition"] == cond][m].mean()
            drift = abs(noisy_mean - clean_mean) / max(clean_mean, 1e-9) * 100
            drifts.append(drift)

        bars = ax.bar(x + j * w - (n_metrics - 1) * w / 2, drifts, w,
                      label=METRIC_LABELS[m], color=METRIC_COLORS[m],
                      edgecolor="black", linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, drifts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(stress_conds)
    ax.set_xlabel("Стресова умова")
    ax.set_ylabel("Зсув середнього, %", fontsize=12)
    ax.set_title("Зсув метрик під стресовими умовами відносно чистих умов", fontsize=12, fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=5, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(len(stress_conds) - 0.5, 6, "поріг 5 %",color="green", alpha=0.7)
    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / "rcu_validation_drift_bars.png")
    plt.close(fig)


def plot_effect_sizes(df):
    """Cohen's d та Vargha-Delaney A для кожної метрики: чисті vs стрес-умови."""
    metrics = ["RCU", "Wall_ms", "Process_ms", "Thread_ms"]
    stress_conds = [c[0] for c in CONDITIONS if c[1] is not None]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cohen's d
    ax = axes[0]
    x = np.arange(len(stress_conds))
    w = 0.8 / len(metrics)
    for j, m in enumerate(metrics):
        clean = df[df["Condition"] == "Чисто"][m].values
        ds = [abs(cohens_d(clean, df[df["Condition"] == c][m].values)) for c in stress_conds]
        ax.bar(x + j * w - (len(metrics) - 1) * w / 2, ds, w,
               label=METRIC_LABELS[m], color=METRIC_COLORS[m],
               edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stress_conds)
    ax.set_xlabel("Стресова умова")
    ax.set_ylabel("|Cohen's d|")
    ax.set_title("Cohen's d: порівняння чистих та стресових умов", fontweight="bold")
    ax.axhline(0.2, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(0.8, color="orange", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Vargha-Delaney A
    ax = axes[1]
    for j, m in enumerate(metrics):
        clean = df[df["Condition"] == "Чисто"][m].values
        vdas = [vargha_delaney_a(clean, df[df["Condition"] == c][m].values) for c in stress_conds]
        ax.bar(x + j * w - (len(metrics) - 1) * w / 2, vdas, w,
               label=METRIC_LABELS[m], color=METRIC_COLORS[m],
               edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(stress_conds)
    ax.set_xlabel("Стресова умова")
    ax.set_ylabel("Vargha-Delaney A")
    ax.set_title("Vargha-Delaney A: ймовірність погіршення під стресом", fontweight="bold")
    ax.axhline(0.5, color="green", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(0.71, color="orange", linestyle=":", linewidth=1, alpha=0.5)
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / "rcu_validation_effect_sizes.png")
    plt.close(fig)


def plot_core_heterogeneity(df):
    """P-ядра vs E-ядра — RCU інваріантність при різній швидкості ядер."""
    if df.empty:
        return

    metrics = ["RCU", "Thread_ms", "Wall_ms"]
    titles = ["RCU", "Час потоку, мс", "Реальний час, мс"]
    core_types = df["Core_Type"].unique()
    workloads = df["Workload"].unique()
    colors_core = {"P-ядра (Interactive)": "#1976D2", "Default": "#757575", "E-ядра (Background)": "#E53935"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, title in zip(axes, metrics, titles):
        x = np.arange(len(workloads))
        w = 0.8 / len(core_types)

        for j, ct in enumerate(core_types):
            means, ci_errs = [], []
            for wl in workloads:
                vals = df[(df["Core_Type"] == ct) & (df["Workload"] == wl)][metric].values
                m = vals.mean()
                lo, hi = bootstrap_ci(vals)
                means.append(m)
                ci_errs.append(([m - lo], [hi - m]))

            err_lo = [e[0][0] for e in ci_errs]
            err_hi = [e[1][0] for e in ci_errs]
            ax.bar(x + j * w - (len(core_types) - 1) * w / 2, means, w,
                   yerr=[err_lo, err_hi], capsize=3,
                   label=ct, color=colors_core.get(ct, "gray"),
                   edgecolor="black", linewidth=0.5, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(workloads,rotation=20, ha="right")
        ax.set_xlabel("Навантаження")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Порівняння метрик на продуктивних та енергоефективних ядрах", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / "rcu_validation_core_heterogeneity.png")
    plt.close(fig)


def plot_anchor_stability(df):
    """Стабільність анкера RCU під різними стресовими умовами."""
    if "Anchor_ns" not in df.columns:
        print("  ⚠️  Дані анкера відсутні (старий формат CSV). Пропускаємо.")
        return

    cond_names = [c[0] for c in CONDITIONS]
    workloads = df["Workload"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Лівий: середній анкер по умовах ---
    ax = axes[0]
    x = np.arange(len(cond_names))
    w = 0.8 / len(workloads)
    colors_wl = ["#2E7D32", "#1565C0", "#E65100", "#6A1B9A"]

    for j, wl in enumerate(workloads):
        means, errs_lo, errs_hi = [], [], []
        for cond in cond_names:
            vals = df[(df["Condition"] == cond) & (df["Workload"] == wl)]["Anchor_ns"].values / 1e6  # → мс
            m = vals.mean()
            lo, hi = bootstrap_ci(vals)
            means.append(m)
            errs_lo.append(m - lo)
            errs_hi.append(hi - m)

        ax.bar(x + j * w - (len(workloads) - 1) * w / 2, means, w,
               yerr=[errs_lo, errs_hi], capsize=3,
               label=wl, color=colors_wl[j % len(colors_wl)],
               edgecolor="black", linewidth=0.5, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names)
    ax.set_xlabel("Умова вимірювання")
    ax.set_ylabel("Тривалість еталонного обчислення, мс")
    ax.set_title("Середня тривалість еталонного обчислення RCU за умовами", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Правий: CV анкера по умовах ---
    ax = axes[1]
    for j, wl in enumerate(workloads):
        cvs = []
        for cond in cond_names:
            vals = df[(df["Condition"] == cond) & (df["Workload"] == wl)]["Anchor_ns"].values
            cvs.append(cv(pd.Series(vals)))

        bars = ax.bar(x + j * w - (len(workloads) - 1) * w / 2, cvs, w,
               label=wl, color=colors_wl[j % len(colors_wl)],
               edgecolor="black", linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, cvs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(cond_names)
    ax.set_xlabel("Умова вимірювання")
    ax.set_ylabel("КВ еталонного обчислення, %")
    ax.set_title("Варіативність еталонного обчислення RCU", fontweight="bold")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_figure(fig, RESULTS_DIR / "rcu_validation_anchor_stability.png")
    plt.close(fig)


# ============================================================
# СТАТИСТИЧНИЙ ЗВІТ
# ============================================================

def print_stats_report(df_stab, df_scale, df_cores=None):
    metrics = ["RCU", "Wall_ms", "Process_ms", "Thread_ms"]
    stress_conds = [c[0] for c in CONDITIONS if c[1] is not None]

    print("\n" + "=" * 70)
    print("📊 СТАТИСТИЧНИЙ ЗВІТ")
    print("=" * 70)

    # --- Тест 1: CV + Дрифт ---
    print("\n═══ Тест 1: Стабільність (CV та Дрифт) ═══")
    print(f"{'Умова':<18} | {'RCU CV':>8} | {'Wall CV':>8} | {'Proc CV':>8} | {'Thr CV':>8}")
    print("-" * 62)
    for cond_name, _, _ in CONDITIONS:
        sub = df_stab[df_stab["Condition"] == cond_name]
        cvs = [f"{cv(sub[m]):.1f}%" for m in metrics]
        print(f"{cond_name:<18} | {cvs[0]:>8} | {cvs[1]:>8} | {cvs[2]:>8} | {cvs[3]:>8}")

    print(f"\n{'Дрифт':>18} | {'RCU':>8} | {'Wall':>8} | {'Process':>8} | {'Thread':>8}")
    print("-" * 62)
    for cond in stress_conds:
        drifts = []
        for m in metrics:
            c_mean = df_stab[df_stab["Condition"] == "Чисто"][m].mean()
            n_mean = df_stab[df_stab["Condition"] == cond][m].mean()
            d = abs(n_mean - c_mean) / max(c_mean, 1e-9) * 100
            drifts.append(f"{d:.1f}%")
        print(f"{cond:<18} | {drifts[0]:>8} | {drifts[1]:>8} | {drifts[2]:>8} | {drifts[3]:>8}")

    # --- Тест 2: Wilcoxon + Effect Size ---
    print("\n═══ Тест 2: Статистична значущість (Wilcoxon + Effect Sizes) ═══")
    print(f"{'Умова':<18} | {'Метрика':<12} | {'p-value':>10} | {'Значущість':>11} | {'Cohen d':>9} | {'VD-A':>8}")
    print("-" * 78)

    for cond in stress_conds:
        for m in metrics:
            clean = df_stab[df_stab["Condition"] == "Чисто"][m].values
            noisy = df_stab[df_stab["Condition"] == cond][m].values
            n_min = min(len(clean), len(noisy))
            try:
                stat, p = sp_stats.wilcoxon(clean[:n_min], noisy[:n_min])
            except ValueError:
                p = 1.0
            d = abs(cohens_d(clean, noisy))
            vda = vargha_delaney_a(clean, noisy)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
            print(f"{cond:<18} | {METRIC_LABELS[m]:<12} | {p:>10.4f} | {sig:>11} | {d:>9.3f} | {vda:>8.3f}")

    # --- Тест 3: Лінійність ---
    if df_scale is not None and not df_scale.empty:
        print("\n═══ Тест 3: Лінійність масштабування ═══")
        print(f"{'Метрика':<15} | {'R²':>8} | {'Нахил':>10} | {'Перетин':>10} | {'SE slope':>10}")
        print("-" * 60)
        for m in metrics[:3]:
            means = df_scale.groupby("Scale")[m].mean()
            scales = means.index.values.astype(float)
            slope, intercept, r_value, p_value, std_err = sp_stats.linregress(scales, means.values)
            r2 = r_value ** 2
            status = "✅" if r2 > 0.999 else ("✅" if r2 > 0.99 else "⚠️")
            print(f"{METRIC_LABELS[m]:<15} | {r2:>8.4f} | {slope:>10.4f} | {intercept:>10.4f} | {std_err:>10.6f} {status}")

    # --- Bootstrap CI ---
    print("\n═══ Bootstrap 95% CI для RCU (по умовах, усереднено по навантаженнях) ═══")
    for cond_name, _, _ in CONDITIONS:
        vals = df_stab[df_stab["Condition"] == cond_name]["RCU"].values
        lo, hi = bootstrap_ci(vals)
        print(f"  {cond_name:<20}: mean={vals.mean():.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], width={hi-lo:.4f}")

    # --- Тест 4: P-core vs E-core ---
    if df_cores is not None and not df_cores.empty:
        print("\n═══ Тест 4: P-ядра vs E-ядра (гетерогенні ядра Apple Silicon) ═══")
        core_types = df_cores["Core_Type"].unique()
        wls = df_cores["Workload"].unique()

        print(f"{'Навантаження':<12} | {'Тип ядра':<22} | {'RCU mean':>10} | {'Thread ms':>10} | {'Wall ms':>10} | {'RCU CV':>8}")
        print("-" * 85)
        for wl in wls:
            for ct in core_types:
                sub = df_cores[(df_cores["Core_Type"] == ct) & (df_cores["Workload"] == wl)]
                print(f"{wl:<12} | {ct:<22} | {sub['RCU'].mean():>10.3f} | {sub['Thread_ms'].mean():>10.2f} | {sub['Wall_ms'].mean():>10.2f} | {cv(sub['RCU']):>7.1f}%")

        print("\n  RCU дрифт P-ядра vs E-ядра (по навантаженнях):")
        for wl in wls:
            p_rcu = df_cores[(df_cores["Core_Type"].str.contains("P-")) & (df_cores["Workload"] == wl)]["RCU"]
            e_rcu = df_cores[(df_cores["Core_Type"].str.contains("E-")) & (df_cores["Workload"] == wl)]["RCU"]
            if len(p_rcu) > 0 and len(e_rcu) > 0:
                drift = abs(e_rcu.mean() - p_rcu.mean()) / max(p_rcu.mean(), 1e-9) * 100
                d = abs(cohens_d(p_rcu.values, e_rcu.values))
                thr_p = df_cores[(df_cores["Core_Type"].str.contains("P-")) & (df_cores["Workload"] == wl)]["Thread_ms"].mean()
                thr_e = df_cores[(df_cores["Core_Type"].str.contains("E-")) & (df_cores["Workload"] == wl)]["Thread_ms"].mean()
                thr_ratio = thr_e / max(thr_p, 1e-9)
                status = "✅" if drift < 15 else "⚠️"
                print(f"    {wl:<10}: RCU дрифт={drift:.1f}% {status} | |d|={d:.2f} | Thread ratio E/P={thr_ratio:.2f}x")

    # --- Підсумок ---
    clean_rcu = df_stab[df_stab["Condition"] == "Чисто"]["RCU"]
    rcu_cv_clean = cv(clean_rcu)

    max_rcu_drift = 0
    max_wall_drift = 0
    for cond in stress_conds:
        noisy_rcu = df_stab[df_stab["Condition"] == cond]["RCU"]
        noisy_wall = df_stab[df_stab["Condition"] == cond]["Wall_ms"]
        d_rcu = abs(noisy_rcu.mean() - clean_rcu.mean()) / max(clean_rcu.mean(), 1e-9) * 100
        c_wall = df_stab[df_stab["Condition"] == "Чисто"]["Wall_ms"]
        d_wall = abs(noisy_wall.mean() - c_wall.mean()) / max(c_wall.mean(), 1e-9) * 100
        max_rcu_drift = max(max_rcu_drift, d_rcu)
        max_wall_drift = max(max_wall_drift, d_wall)

    print(f"\n{'=' * 70}")
    print("📝 ВИСНОВОК")
    print("=" * 70)
    print(f"  RCU CV (чисто):           {rcu_cv_clean:.1f}%")
    print(f"  Макс. RCU дрифт:          {max_rcu_drift:.1f}%")
    print(f"  Макс. реальний час дрифт: {max_wall_drift:.1f}%")
    print(f"  N bootstrap ітерацій:     {N_BOOTSTRAP}")

    if max_rcu_drift < 15 and rcu_cv_clean < 10:
        print(f"\n  ✅ RCU ВАЛІДНА: макс. дрифт {max_rcu_drift:.1f}% при реальному часі {max_wall_drift:.1f}%.")
        print(f"     Покращення стабільності: {max_wall_drift / max(max_rcu_drift, 0.1):.0f}x порівняно з реальним часом.")
    else:
        print(f"\n  ⚠️  Деякі показники виходять за межі очікувань.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Читаємо збережені дані
    DATA_DIR = Path(__file__).resolve().parent / "data"
    stab_path = DATA_DIR / "data_stability.csv"
    scale_path = DATA_DIR / "data_scaling.csv"
    cores_path = DATA_DIR / "data_cores.csv"

    if not stab_path.exists():
        print(f"❌ Файл {stab_path} не знайдено. Спочатку запустіть ex00.py для збору даних.")
        sys.exit(1)

    df_stab = pd.read_csv(stab_path)
    df_scale = pd.read_csv(scale_path) if scale_path.exists() else pd.DataFrame()
    df_cores = pd.read_csv(cores_path) if cores_path.exists() else pd.DataFrame()

    print("📊 Генерація графіків з збережених даних...")

    plot_drift_bars(df_stab)
    plot_effect_sizes(df_stab)
    plot_anchor_stability(df_stab)
    if not df_cores.empty:
        plot_core_heterogeneity(df_cores)

    print_stats_report(df_stab, df_scale, df_cores)

    print(f"\n  Графіки збережено у {RESULTS_DIR}/")
    for f in sorted(RESULTS_DIR.glob("rcu_validation_*.png")):
        print(f"    • {f.name}")
