#!/usr/bin/env python3
"""
Універсальний диспетчер для всіх HPO ешелонів (L0-L4)

Використання:
    python3 run_benchmark.py <tier> [seeds] [--force]

Приклади:
    python3 run_benchmark.py L0 5         # Запустити L0 на 5 сідів
    python3 run_benchmark.py ALL 3        # Запустити всі ешелоні
"""

import os
import sys
import time
import concurrent.futures
import subprocess
import warnings
import logging
from tqdm import tqdm

# Глушимо всі спам-ворнінги (Sklearn, Optuna, SMAC, YAHPO, тощо)
warnings.filterwarnings('ignore')
logging.getLogger('optuna').setLevel(logging.ERROR)
logging.getLogger('smac').setLevel(logging.ERROR)
logging.getLogger('dehb').setLevel(logging.ERROR)
logging.getLogger('yahpo_gym').setLevel(logging.ERROR)

# Обмежуємо внутрішні потоки PyTorch/OpenMP на рівні ініціалізації процесу
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from benchmark import TIERS, ACTIVE_METHODS
from run_method import run_single_cell, load_method


def runner(task):
    """Обертка для воркера, яка перехоплює помилки."""
    m_name, tier, dataset, mdl, seed, budget, force = task
    try:
        method_fn = load_method(m_name)
    except Exception as e:
        return task, False, f"load_method error: {e}"

    try:
        res = run_single_cell(m_name, method_fn, tier, dataset, mdl, seed, budget, force)
        return task, True, res
    except Exception as e:
        return task, False, str(e)


def process_tier(tier, seeds, force):
    """Створює завдання для конкретного ешелону."""
    if tier not in TIERS:
        print(f"Помилка: невідомий рівень {tier}. Доступні: {list(TIERS.keys())}")
        return []

    cfg = TIERS[tier]
    budget = cfg['budget']

    if tier == 'L_ABLATION':
        methods_to_run = ["sacma_v3", "sacma_v3_no_adapt", "sacma_v3_no_virtual", "sacma_base"]
    else:
        methods_to_run = ACTIVE_METHODS

    tasks = []
    for m in methods_to_run:
        for d in cfg['datasets']:
            for mdl in cfg['models']:
                for s in range(seeds):
                    tasks.append((m, tier, d, mdl, s, budget, force))
    return tasks


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target_tier = sys.argv[1].upper()
    force = '--force' in sys.argv

    # Розрахунок сідів
    if target_tier in TIERS:
        default_s = TIERS[target_tier].get('default_seeds', 1)
    else:
        default_s = 1

    seeds = default_s
    for a in sys.argv[2:]:
        if a.isdigit():
            seeds = int(a)

    tiers_to_run = list(TIERS.keys()) if target_tier == 'ALL' else [target_tier]

    print("=" * 80)
    print("    GLOBAL HPO BENCHMARK PIPELINE")
    print("=" * 80)
    print(f"  Ешелони:     {tiers_to_run}")
    print(f"  Методів:     {len(ACTIVE_METHODS)}")
    print(f"  Seeds:       {seeds}")
    print(f"  Force cache: {force}")
    print("=" * 80)
    print()

    # Збираємо всі задачі
    all_tasks = []
    for t in tiers_to_run:
        all_tasks.extend(process_tier(t, seeds, force))

    if not all_tasks:
        sys.exit(1)

    print(f"Сформовано {len(all_tasks)} завдань. Запускаємо глобальний пул...")

    t0 = time.time()
    done = 0
    skipped = 0
    errors = 0

    # Визначаємо кількість воркерів (безпечно для L4 - макс 10-12, для інших можна більше)
    max_w = min(os.cpu_count() or 4, 10)

    # Запускаємо через ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_w) as pool:
        futures = {pool.submit(runner, t): t for t in all_tasks}

        with tqdm(total=len(all_tasks), desc="HPO Benchmark", unit="task", dynamic_ncols=True) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                task, success, msg = fut.result()
                m_name, tier, dataset, mdl, seed, budget, _ = task

                if success:
                    if msg is None:
                        skipped += 1
                        pbar.set_postfix({'done': done, 'cached': skipped, 'err': errors, 'last': f'{m_name[:8]} (cache)'})
                    else:
                        done += 1
                        loss = msg.get('loss', 1.0)
                        pbar.set_postfix({'done': done, 'cached': skipped, 'err': errors, 'last_loss': f'{loss:.3f}'})
                else:
                    errors += 1
                    pbar.write(f"   {m_name[:12]:12s} | {tier:2s} | {dataset[:20]:20s} | s{seed} | ERROR: {msg}")
                    pbar.set_postfix({'done': done, 'cached': skipped, 'err': errors, 'last': 'ERROR'})

                pbar.update(1)

    print("\n" + "=" * 80)
    print(f"Done: {done} computed, {skipped} cached, {errors} errors | Time: {time.time()-t0:.1f}s")

    # Генерація звітів для всіх запущених ешелонів
    for t in tiers_to_run:
        print(f"\n▶ Generating Reports for {t}...")
        subprocess.run(["python3", "report.py", t])
        subprocess.run(["python3", "analyze_visuals.py", t])

    # Звукове сповіщення про завершення тестування на Mac
    try:
        os.system('afplay /System/Library/Sounds/Glass.aiff &')
    except Exception:
        pass


if __name__ == '__main__':
    main()
