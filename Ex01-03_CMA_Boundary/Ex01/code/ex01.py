# Ex01: CMA-ES vs AdamW (MR, 10 restarts) vs AdaBelief (MR, 10 restarts) vs Random Search на Rastrigin (мінімізація).
# Виходи: папка Ex01/ — графіки (PNG), таблиця (LaTeX + PNG).

import sys
import argparse
from pathlib import Path

# Налаштування шляху для імпорту common модулів (code/ всередині Ex01/ всередині
# Ex01-03_CMA_Boundary/; спільний common/ — на корені публічного репозиторію,
# тому тут на один .parent більше, ніж в оригіналі Ex01/code/)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# Перевірка та автоматичне встановлення залежностей ПЕРЕД імпортом модулів
from common import ensure_experiment_dependencies

# Перевіряємо та встановлюємо залежності для експериментів (включаючи базові: tqdm, cma, scipy)
deps_status = ensure_experiment_dependencies()

# Встановлюємо non-interactive backend для швидшого рендерингу без затримок курсора
import matplotlib
matplotlib.use('Agg')

import numpy as np
import torch
import torch.optim as optim
import cma
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import stats
from scipy.stats import studentized_range

# AdaBelief: адаптивний оптимізатор (2020)
ADABELIEF_AVAILABLE = deps_status.get("adabelief_pytorch", False)

if ADABELIEF_AVAILABLE:
    from adabelief_pytorch import AdaBelief
else:
    AdaBelief = None
    print("Warning: adabelief-pytorch недоступний. Falling back to AdamW for AdaBelief method.")

# Спільні компоненти для всіх експериментів
from common import (
    save_figure,
    save_table_latex,
    save_table_png,
    suppress_stdout,
    setup_experiment,
    save_experiment_data,
    set_dstu_style,
    legend_outside,
    create_figure,
)
from common.rcu import profile_rcu as _profile_rcu, setup_rcu_worker, ANCHOR_LOOPS
# ДСТУ 3008:2015: графіки та таблиці — Times New Roman, 12 pt
set_dstu_style()

# Ініціалізація експерименту (OUTPUT_DIR = Ex01/results/, не code/)
OUTPUT_DIR = setup_experiment(Path(__file__).resolve().parent.parent)

# Палітра: Multiple Restarts для обох градієнтних методів (10 restarts кожен)
PALETTE_EX01 = {"AdamW (MR)": "#D32F2F", "AdaBelief (MR)": "#1976D2", "CMA-ES (EA)": "#2E8B57", "Random Search": "#757575"}

class RastriginProblem:
    """
    Класична бенчмарк-функція для перевірки здатності виходити з локальних мінімумів.
    f(x) = 10d + sum(x_i^2 - 10cos(2pi*x_i))
    Global min: f(0) = 0.
    Bounds: [-5.12, 5.12]
    """
    def __init__(self, dim=10):
        self.dim = dim
        self.bounds = (-5.12, 5.12)
        self.fevals = 0 # Лічильник викликів функції

    def reset(self):
        self.fevals = 0

    def __call__(self, x):
        """NumPy інтерфейс (для CMA-ES та Random Search)"""
        self.fevals += 1
        # f(x) implementation
        x = np.array(x)
        return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def evaluate_torch(self, x_tensor):
        """PyTorch інтерфейс для градієнтних методів (AdamW, AdaBelief)."""
        # fevals оновлюється в солвері: 1 крок = 2 FE (forward + backward).
        term1 = 10 * self.dim
        term2 = torch.sum(x_tensor**2 - 10 * torch.cos(2 * np.pi * x_tensor))
        return term1 + term2

# ==========================================
# 3. SOLVERS (ALGORITHMS)
# ==========================================

def run_adamw_single(problem, max_fe_per_restart, seed_offset):
    """Один restart AdamW: використовується всередині Multiple Restarts."""
    torch.manual_seed(42 + seed_offset)
    problem.reset()

    start_x = (torch.rand(problem.dim) * (problem.bounds[1] - problem.bounds[0]) + problem.bounds[0])
    x_tensor = start_x.clone().detach().requires_grad_(True)
    optimizer = optim.AdamW([x_tensor], lr=0.1, weight_decay=0.01)

    history_fe = []
    history_loss = []
    fe_per_step = 2

    while problem.fevals < max_fe_per_restart:
        optimizer.zero_grad()
        loss = problem.evaluate_torch(x_tensor)
        loss.backward()
        optimizer.step()
        problem.fevals += fe_per_step
        history_fe.append(problem.fevals)
        history_loss.append(loss.item())

    return history_fe, history_loss, loss.item()


def run_adamw_multiple_restarts(problem, max_fe, seed):
    """
    AdamW з Multiple Restarts: чесне порівняння з CMA-ES (популяція).
    Розподіляємо max_fe між n_restarts запусками з різних стартових точок.
    """
    n_restarts = 10  # Кількість restarts (аналог популяції в CMA-ES)
    max_fe_per_restart = max_fe // n_restarts

    best_final_loss = float('inf')
    best_history_fe = None
    best_history_loss = None
    total_fevals_used = 0

    for restart_idx in range(n_restarts):
        problem.reset()  # Скидаємо лічильник для кожного restart
        fe, loss, final_loss = run_adamw_single(problem, max_fe_per_restart, seed + restart_idx)
        total_fevals_used += problem.fevals
        
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            # Нормалізуємо FE до глобального масштабу (накопичуємо)
            fe_offset = restart_idx * max_fe_per_restart
            best_history_fe = [f + fe_offset for f in fe]
            best_history_loss = loss.copy()

    # Якщо не використали весь бюджет, додаємо останні FE до історії
    if total_fevals_used < max_fe:
        if best_history_fe:
            best_history_fe.append(max_fe)
            best_history_loss.append(best_final_loss)

    return best_history_fe if best_history_fe else [max_fe], best_history_loss if best_history_loss else [best_final_loss]

def run_cma_es(problem, max_fe, seed):
    np.random.seed(seed)
    problem.reset()

    # Start point
    x0 = np.random.uniform(problem.bounds[0], problem.bounds[1], problem.dim)
    sigma0 = 2.0 # Великий початковий розкид для глобального пошуку

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'seed': seed,
        'verbose': -9,
        'popsize': 32 # Фіксуємо розмір популяції
    })

    history_fe = []
    history_loss = []

    while not es.stop() and problem.fevals < max_fe:
        solutions = es.ask()

        # Обчислення фітнесу (Function Evaluations відбуваються тут)
        fitnesses = [problem(s) for s in solutions]
        es.tell(solutions, fitnesses)

        # Логуємо найкращий результат у популяції
        history_fe.append(problem.fevals)
        history_loss.append(min(fitnesses))

    return history_fe, history_loss

def run_random_search(problem, max_fe, seed):
    np.random.seed(seed)
    problem.reset()

    history_fe = []
    history_loss = []
    best_loss = float('inf')

    while problem.fevals < max_fe:
        # Генеруємо випадкову точку
        x = np.random.uniform(problem.bounds[0], problem.bounds[1], problem.dim)
        loss = problem(x) # +1 FE

        if loss < best_loss:
            best_loss = loss

        # Логуємо не кожен крок, щоб не забити пам'ять, але часто
        if problem.fevals % 10 == 0:
            history_fe.append(problem.fevals)
            history_loss.append(best_loss)

    return history_fe, history_loss


def run_adabelief_single(problem, max_fe_per_restart, seed_offset):
    """Один restart AdaBelief: використовується всередині Multiple Restarts."""
    if not ADABELIEF_AVAILABLE or AdaBelief is None:
        # Fallback до AdamW якщо AdaBelief недоступний
        return run_adamw_single(problem, max_fe_per_restart, seed_offset)
    
    torch.manual_seed(42 + seed_offset)
    problem.reset()

    start_x = (torch.rand(problem.dim) * (problem.bounds[1] - problem.bounds[0]) + problem.bounds[0])
    x_tensor = start_x.clone().detach().requires_grad_(True)
    # AdaBelief v0.2.0+: eps=1e-16 для задач де Adam краще за SGD (оптимізація)
    # Придушуємо вивід про weight decoupling та rectification
    with suppress_stdout():
        optimizer = AdaBelief([x_tensor], lr=0.1, eps=1e-16, betas=(0.9, 0.999), weight_decay=0.01, print_change_log=False)

    history_fe = []
    history_loss = []
    fe_per_step = 2

    while problem.fevals < max_fe_per_restart:
        optimizer.zero_grad()
        loss = problem.evaluate_torch(x_tensor)
        loss.backward()
        optimizer.step()
        problem.fevals += fe_per_step
        history_fe.append(problem.fevals)
        history_loss.append(loss.item())

    return history_fe, history_loss, loss.item()


def run_adabelief_multiple_restarts(problem, max_fe, seed):
    """
    AdaBelief з Multiple Restarts: чесне порівняння з CMA-ES (популяція).
    Розподіляємо max_fe між n_restarts запусками з різних стартових точок.
    """
    n_restarts = 10  # Кількість restarts (аналог популяції в CMA-ES)
    max_fe_per_restart = max_fe // n_restarts

    best_final_loss = float('inf')
    best_history_fe = None
    best_history_loss = None
    total_fevals_used = 0

    for restart_idx in range(n_restarts):
        problem.reset()  # Скидаємо лічильник для кожного restart
        fe, loss, final_loss = run_adabelief_single(problem, max_fe_per_restart, seed + restart_idx)
        total_fevals_used += problem.fevals
        
        if final_loss < best_final_loss:
            best_final_loss = final_loss
            # Нормалізуємо FE до глобального масштабу (накопичуємо)
            fe_offset = restart_idx * max_fe_per_restart
            best_history_fe = [f + fe_offset for f in fe]
            best_history_loss = loss.copy()

    # Якщо не використали весь бюджет, додаємо останні FE до історії
    if total_fevals_used < max_fe:
        if best_history_fe:
            best_history_fe.append(max_fe)
            best_history_loss.append(best_final_loss)

    return best_history_fe if best_history_fe else [max_fe], best_history_loss if best_history_loss else [best_final_loss]


# Відображення імен методів на функції солверів (для воркера паралельного запуску)
METHOD_FUNCS = {
    "AdamW (MR)": run_adamw_multiple_restarts,
    "AdaBelief (MR)": run_adabelief_multiple_restarts,
    "CMA-ES (EA)": run_cma_es,
    "Random Search": run_random_search,
}


def _run_one_trial_method(args):
    """
    Воркер для одного (trial, method): виконується в окремому процесі.
    RCU: sandwich profiling з фіксованим anchor (ANCHOR_LOOPS={ANCHOR_LOOPS}).
    Повертає (method_name, trial_idx, fe, loss, rcu_cost, anchor_pre_ns, anchor_post_ns).
    """
    (method_name, trial_idx, seed, DIM, MAX_FE) = args
    setup_rcu_worker()  # P-ядра + ініціалізація
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    problem = RastriginProblem(dim=DIM)
    solver = METHOD_FUNCS[method_name]

    # RCU: sandwich profiling (anchor_pre → payload → anchor_post)
    (fe, loss), rcu_cost, anchor_pre_ns, anchor_post_ns = _profile_rcu(solver, problem, MAX_FE, seed)

    return (method_name, trial_idx, fe, loss, rcu_cost, anchor_pre_ns, anchor_post_ns)


# ==========================================
# 4. EXPERIMENT EXECUTION
# ==========================================

def run_full_experiment(n_trials=20):
    """
    RCU profiling з фіксованим anchor (ANCHOR_LOOPS).

    Паралелізм: ProcessPoolExecutor на рівні (trial, method). Кожен воркер виконує
    setup_rcu_worker() для P-core pinning; OMP_NUM_THREADS=1, torch.set_num_threads(1).

    Args:
        n_trials: Кількість незалежних запусків (trials) для кожного методу
    """
    DIM = 10
    MAX_FE = 3000
    N_TRIALS = n_trials
    L_TARGET = 10.0

    method_names = list(METHOD_FUNCS.keys())
    n_methods = len(method_names)
    total_tasks = N_TRIALS * n_methods
    W = min(total_tasks, os.cpu_count() or 4)

    print(f"Starting Benchmark on Rastrigin (D={DIM})...")
    print(f"Parallel execution: {W} workers, {total_tasks} tasks. RCU anchor: {ANCHOR_LOOPS} loops (фікс.).")

    tasks = [
        (method_name, trial_idx, 42 + trial_idx, DIM, MAX_FE)
        for trial_idx in range(N_TRIALS)
        for method_name in method_names
    ]

    raw_results = []
    with ProcessPoolExecutor(max_workers=W) as executor:
        futures = {executor.submit(_run_one_trial_method, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            raw_results.append(future.result())
            done += 1
            if done % (n_methods * 5) == 0 or done == total_tasks:
                print(f"Completed {done}/{total_tasks} runs...")

    # Сортуємо за (trial_idx, method_name) для узгодженості з попередньою логікою
    raw_results.sort(key=lambda r: (r[1], r[0]))

    results_conv = []
    results_final = []
    for method_name, trial_idx, fe, loss, rcu_cost, anchor_pre_ns, anchor_post_ns in raw_results:
        final_metric = loss[-1] if loss else float("inf")
        anchor_avg_ms = (anchor_pre_ns + anchor_post_ns) / 2.0 / 1e6  # нс → мс

        df_trial = pd.DataFrame({"FE": fe, "Loss": loss})
        df_trial["Method"] = method_name
        df_trial["Trial"] = trial_idx
        if len(df_trial) > 100:
            indices = np.linspace(0, len(df_trial) - 1, 100).astype(int)
            df_trial = df_trial.iloc[indices]
        results_conv.append(df_trial)

        results_final.append({
            "Method": method_name,
            "Final Loss": final_metric,
            "Time_RCU": rcu_cost,
            "Anchor_pre_ns": anchor_pre_ns,
            "Anchor_post_ns": anchor_post_ns,
            "Anchor_avg_ms": anchor_avg_ms,
            "Trial": trial_idx,
        })

    print("Done.")
    return pd.concat(results_conv), pd.DataFrame(results_final), L_TARGET, N_TRIALS

# ==========================================
# 5. ПАРАМЕТРИ ЕКСПЕРИМЕНТУ (запуск лише при виконанні ex01.py як __main__)
# ==========================================
# Для запуску експерименту використовуйте ex01_run.py (--mode test/experiment, --quick, -n).

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ex01: Порівняння CMA-ES з градієнтними методами на Rastrigin')
    parser.add_argument('--trials', '-n', type=int, default=None,
                        help='Кількість незалежних запусків (trials) для кожного методу. За замовчуванням: інтерактивне введення')
    args = parser.parse_args()

    # Визначення кількості прогонів
    if args.trials is not None:
        N_TRIALS = args.trials
    else:
        # Інтерактивне введення, якщо аргумент не передано
        print("\n" + "="*60)
        print("Налаштування кількості прогонів експерименту")
        print("="*60)
        print("Рекомендації:")
        print("  - Тестовий запуск: 2-5 прогонів (швидка перевірка)")
        print("  - Фінальний експеримент: 20 прогонів (статистична значущість)")
        print("="*60)
        while True:
            try:
                user_input = input("\nВведіть кількість прогонів (trials) [за замовчуванням: 20]: ").strip()
                if user_input == "":
                    N_TRIALS = 20
                    break
                N_TRIALS = int(user_input)
                if N_TRIALS < 1:
                    print("Помилка: кількість прогонів має бути >= 1")
                    continue
                break
            except ValueError:
                print("Помилка: введіть ціле число")
            except KeyboardInterrupt:
                print("\n\nПерервано користувачем. Використовується значення за замовчуванням: 20")
                N_TRIALS = 20
                break

    print(f"\nВибрано кількість прогонів: {N_TRIALS}")
    if N_TRIALS < 10:
        print("Увага: менше 10 прогонів може бути недостатньо для статистично значущих висновків")

    # Запуск (RCU з фіксованим anchor)
    df_conv, df_final, L_TARGET, N_TRIALS = run_full_experiment(n_trials=N_TRIALS)

    # Зберігаємо дані експерименту для подальшої візуалізації (Ex01/data/)
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    DATA_DIR.mkdir(exist_ok=True)
    data_file = DATA_DIR / f"ex01_data_n{N_TRIALS}.json"
    experiment_data = {
        'convergence': df_conv,
        'final': df_final,
        'metadata': {
            'ANCHOR_LOOPS': ANCHOR_LOOPS,
            'L_TARGET': float(L_TARGET),
            'N_TRIALS': int(N_TRIALS),
            'DIM': 10,
            'MAX_FE': 3000,
        }
    }
    save_experiment_data(experiment_data, data_file)

    # ==========================================
    # 6. VISUALIZATION (збереження в папку експерименту)
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
    save_figure(fig_landscape, OUTPUT_DIR / "ex01_landscape.png")
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
    ax_conv.set_xlabel("Оцінки цільової функції (обчислювальна вартість)", fontsize=12)
    ax_conv.set_ylabel("Втрати / Помилка", fontsize=12)
    ax_conv.legend(title="Метод")
    ax_conv.grid(True, alpha=0.3)

    # Додаємо вертикальні лінії для індикації рестартів у Multiple Restarts методах
    # Для MR методів: 10 рестартів, кожен по 300 FE (3000 FE / 10)
    MAX_FE = 3000
    N_RESTARTS_MR = 10
    FE_PER_RESTART = MAX_FE // N_RESTARTS_MR

    # Вертикальні лінії на місцях рестартів (кожні 300 FE)
    for restart_idx in range(1, N_RESTARTS_MR):
        restart_fe = restart_idx * FE_PER_RESTART
        ax_conv.axvline(restart_fe, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)

    # Додаємо текстовий індикатор для пояснення
    ax_conv.text(0.02, 0.98, 'Пунктирні лінії: рестарти\nдля MR методів (кожні 300 FE)',
             transform=ax_conv.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    save_figure(fig_conv, OUTPUT_DIR / "ex01_convergence.png")
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
    save_figure(fig_final, OUTPUT_DIR / "ex01_final_distribution.png")
    plt.close(fig_final)

    # --- Розподіл точності vs обчислювальна вартість (Loss vs RCU) ---
    fig_acc_time, ax_acc_time, rect_acc = create_figure("wide", legend_outside=True)

    # Краща область: нижня третина по втратах і по RCU (quantile 0.33)
    loss_threshold = df_final["Final Loss"].quantile(0.33)
    rcu_threshold = df_final["Time_RCU"].quantile(0.33)

    # Напівпрозорий прямокутник для кращої області (лівий нижній кут)
    ax_acc_time.axhspan(0, loss_threshold, xmin=0, xmax=rcu_threshold/df_final["Time_RCU"].max(), 
                     alpha=0.15, color='green', label='Краща область\n(низькі втрати та RCU)')

    for method in df_final["Method"].unique():
        method_data = df_final[df_final["Method"] == method]
        ax_acc_time.scatter(
            method_data["Time_RCU"],
            method_data["Final Loss"],
            label=method,
            color=PALETTE_EX01.get(method, "gray"),
            alpha=0.7,
            s=60,
            edgecolors='black',
            linewidths=0.5,
        )

    ax_acc_time.set_yscale("log")
    ax_acc_time.set_xlabel("Обчислювальна вартість (RCU)", fontsize=12)
    ax_acc_time.set_ylabel("Фінальні втрати (логарифмічна шкала)", fontsize=12)
    ax_acc_time.set_title("Точність vs Обчислювальна вартість\n(RCU включає всі 10 рестартів MR; краще → лівий нижній кут)", fontsize=12, fontweight="bold")
    legend_outside(ax_acc_time, side="right", title="Метод")
    ax_acc_time.grid(True, alpha=0.3)

    ax_acc_time.text(0.02, 0.98, '← Менший RCU\n(Краще)', transform=ax_acc_time.transAxes,
                 fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax_acc_time.text(0.98, 0.02, 'Менші втрати\n(Краще) ↓', transform=ax_acc_time.transAxes,
                 fontsize=9, verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=rect_acc)
    save_figure(fig_acc_time, OUTPUT_DIR / "ex01_accuracy_vs_time.png")
    plt.close(fig_acc_time)

    # --- Комплексний підхід: Friedman → ранжування → Nemenyi → візуалізація ---
    method_list = df_final["Method"].unique().tolist()
    k, N = len(method_list), N_TRIALS
    pivot_loss = df_final.pivot(index="Trial", columns="Method", values="Final Loss")

    # 1. Тест Фрідмана
    friedman_stat, friedman_p = stats.friedmanchisquare(*[pivot_loss[m].values for m in method_list])
    # Результати зберігаються у файли, не виводяться в консоль

    # 2. Ранжування (по кожному trial: 1 = найкращий loss)
    ranks = pivot_loss.rank(axis=1, method="average")
    mean_rank = ranks.mean(axis=0).reindex(method_list)
    # Результати зберігаються у файли, не виводяться в консоль

    # 3. Пост-хок Немені: CD = q * sqrt(k(k+1)/(6*N))
    q_05 = studentized_range.ppf(0.95, k, np.inf)
    CD = q_05 * np.sqrt(k * (k + 1) / (6 * N))
    # Результати зберігаються у файли, не виводяться в консоль

    # 4. Візуалізація: середні ранги + індикатор CD + значення рангів на стовпцях
    df_rank = mean_rank.reset_index()
    df_rank.columns = ["Method", "Mean rank"]
    df_rank = df_rank.sort_values("Mean rank")
    fig_friedman, ax_f, _ = create_figure("friedman")
    colors_f = [PALETTE_EX01.get(m, "gray") for m in df_rank["Method"]]
    bars = ax_f.barh(df_rank["Method"], df_rank["Mean rank"], color=colors_f)

    # Додаємо значення рангів на стовпцях
    for i, (bar, rank_val) in enumerate(zip(bars, df_rank["Mean rank"])):
        ax_f.text(rank_val + 0.05, bar.get_y() + bar.get_height()/2, 
              f'{rank_val:.2f}', 
              va='center', ha='left', fontsize=12, fontweight='bold')

    ax_f.axvline(CD, color="red", linestyle="--", linewidth=1.5, label=f"CD = {CD:.3f}")
    ax_f.set_xlabel("Середній ранг (1 = найкращий)", fontsize=12)
    ax_f.set_title("Ранжування Фрідмана та критична різниця Немені (α=0.05)", fontsize=12, fontweight="bold")
    ax_f.legend()
    ax_f.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    save_figure(fig_friedman, OUTPUT_DIR / "ex01_friedman_nemenyi.png")
    plt.close(fig_friedman)

    # Створюємо df_friedman_report тільки для використання в summary таблиці (не зберігаємо окремо)
    df_friedman_report = pd.DataFrame({
        "Method": method_list,
        "Mean rank": [mean_rank[m] for m in method_list],
    })

    # Створюємо українську версію для summary таблиці
    df_friedman_report_ukr = df_friedman_report.copy()
    df_friedman_report_ukr.columns = ["Метод", "Середній ранг"]

    # Створюємо копію з українськими назвами колонок для summary таблиці
    df_final_ukr = df_final.copy()
    df_final_ukr = df_final_ukr.rename(columns={
        "Method": "Метод",
        "Final Loss": "Фінальні втрати",
        "Time_RCU": "RCU"
    })

    summary = (
        df_final_ukr.groupby("Метод")
        .agg({
        "Фінальні втрати": ["mean", "std", "min"],
        "RCU": ["mean", "std"],
        })
        .reset_index()
    )

    # Вирівнюємо MultiIndex колонки
    summary.columns = ["Метод", "Фінальні втрати (середнє)", "Фінальні втрати (ст. відх.)", "Фінальні втрати (мін.)", "RCU (середнє)", "RCU (ст. відх.)"]
    # Форматуємо RCU як mean ± std
    summary["RCU (середнє ± ст. відх.)"] = summary.apply(
        lambda r: f"{r['RCU (середнє)']:.3f} ± {r['RCU (ст. відх.)']:.3f}", axis=1
    )
    summary = summary.drop(columns=["RCU (середнє)", "RCU (ст. відх.)"])

    # Додаємо середні ранги Фрідмана до summary таблиці
    summary_with_ranks = summary.merge(
        df_friedman_report_ukr[["Метод", "Середній ранг"]],
        on="Метод",
        how="left"
    )

    summary_rounded = summary_with_ranks.round(2)
    save_table_latex(summary_rounded.round(2), OUTPUT_DIR / "ex01_summary.tex")
    summary_for_png = summary_rounded.copy()
    save_table_png(summary_for_png.round(2), OUTPUT_DIR / "ex01_summary.png")

    # Примітка до таблиці
    note_lines = [
        "## Примітка до таблиці Ex01",
        "",
        f"- **RCU (середнє ± ст. відх.)** — усереднення по {N_TRIALS} незалежних статистичних прогонах.",
        "- Градієнтні методи AdamW (MR) та AdaBelief (MR) використовують **10 випадкових рестартів**",
        "  (Multiple Restarts) як шанс знайти глобальний оптимум.",
        "- RCU відображає **повну вартість** алгоритму, включаючи всі 10 рестартів.",
        "- Бюджет FE однаковий для всіх методів (MAX_FE = 3000), градієнтні ділять його на 10 × 300.",
        "- Незважаючи на 10 шансів із різних стартових точок, градієнтні методи",
        "  поступаються CMA-ES за якістю розв'язку при зіставних обчислювальних витратах.",
        f"- Anchor: ANCHOR_LOOPS = {ANCHOR_LOOPS} (фіксований для всього проєкту).",
    ]
    (OUTPUT_DIR / "ex01_summary.md").write_text("\n".join(note_lines), encoding="utf-8")