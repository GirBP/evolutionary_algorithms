#!/usr/bin/env python3
"""
Wall-Clock Time Analysis Runner — L2_WCT
=========================================
Запускає 5 цільових методів на LCBench (YAHPO) з захопленням
часу навчання кожної конфігурації (train_time_curve).

Зберігає результати в results/L2_WCT/ — НЕ конфліктує з results/L2/

Запуск:
    python3 run_wct.py
"""
import os
import sys
import time
import concurrent.futures

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import TIERS, WCT_METHODS
from run_method import run_single_cell, load_method


def runner(task):
    m_name, tier, dataset, mdl, seed, budget = task
    try:
        method_fn = load_method(m_name)
        res = run_single_cell(m_name, method_fn, tier, dataset, mdl, seed, budget, force=True)
        return task, True, res
    except Exception as e:
        return task, False, str(e)


def main():
    tier = 'L2_WCT'
    cfg = TIERS[tier]
    budget = cfg['budget']
    seeds = cfg['default_seeds']

    tasks = [
        (m, tier, d, mdl, s, budget)
        for m in WCT_METHODS
        for d in cfg['datasets']
        for mdl in cfg['models']
        for s in range(seeds)
    ]

    print("=" * 70)
    print("  Wall-Clock Time Analysis — L2_WCT")
    print(f"  Методи:   {WCT_METHODS}")
    print(f"  Датасети: {len(cfg['datasets'])} LCBench instances")
    print(f"  Seeds:    {seeds} | Budget: {budget}")
    print(f"  Задач:    {len(tasks)}")
    print(f"  Результати → results/L2_WCT/ (НЕ перезаписує results/L2/)")
    print("=" * 70)

    t0 = time.time()
    done, errors = 0, 0
    max_workers = min(10, os.cpu_count() or 4)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
        for fut in concurrent.futures.as_completed(
            {pool.submit(runner, t): t for t in tasks}
        ):
            task, success, msg = fut.result()
            m_name, tier, dataset, mdl, seed, budget = task
            ds_short = dataset.split('__')[-1]
            if success and msg is not None:
                loss = msg.get('loss', 1.0)
                has_tc = msg.get('train_time_curve') is not None
                tc_len = len(msg.get('train_time_curve') or [])
                print(f"   {m_name:<14} | {ds_short:<10} | s{seed} | loss={loss:.4f} | train_curve={'OK '+str(tc_len)+'pts' if has_tc else 'MISSING'}")
                done += 1
            elif success and msg is None:
                print(f"  ⏭  {m_name:<14} | {ds_short:<10} | s{seed} | cached")
                done += 1
            else:
                print(f"   {m_name:<14} | {ds_short:<10} | s{seed} | ERROR: {msg}")
                errors += 1

    print(f"\nDone: {done} computed, {errors} errors | Time: {time.time()-t0:.1f}s")
    print("Аналіз результатів: python3 analyze_wct.py")


if __name__ == '__main__':
    main()
