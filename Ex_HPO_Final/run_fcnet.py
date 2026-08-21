#!/usr/bin/env python3
"""
FCNet Wall-Clock Analysis Runner — L5_FCNET
============================================
Запускає 5 цільових методів на FCNet (Klein & Hutter, 2019)
з записом реального часу тренування кожної конфігурації.

4 датасети × 5 методів × 10 сідів = 200 задач
Результати → results/L5_FCNET/

Перед запуском потрібно скачати дані:
    curl -L http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz | tar xz -C data/fcnet/

Запуск:
    python3 run_fcnet.py
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
    tier = 'L5_FCNET'
    cfg = TIERS[tier]
    budget = cfg['budget']
    seeds = cfg['default_seeds']

    # Перевірка що дані є
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'fcnet')
    expected_files = [
        'fcnet_protein_structure_data.hdf5',
        'fcnet_slice_localization_data.hdf5',
        'fcnet_naval_propulsion_data.hdf5',
        'fcnet_parkinsons_telemonitoring_data.hdf5',
    ]
    missing = [f for f in expected_files if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print(f" Відсутні HDF5 файли у {data_dir}:")
        for f in missing:
            print(f"   - {f}")
        print(f"\nСкачайте дані командою:")
        print(f"  curl -L http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz | tar xz -C {data_dir}/")
        sys.exit(1)

    tasks = [
        (m, tier, d, mdl, s, budget)
        for m in WCT_METHODS
        for d in cfg['datasets']
        for mdl in cfg['models']
        for s in range(seeds)
    ]

    print("=" * 70)
    print("  FCNet Wall-Clock Analysis — L5_FCNET")
    print(f"  Методи:   {WCT_METHODS}")
    print(f"  Датасети: {cfg['datasets']}")
    print(f"  Seeds:    {seeds} | Budget: {budget} | dim=9")
    print(f"  Задач:    {len(tasks)}")
    print(f"  Результати → results/L5_FCNET/")
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
            if success and msg is not None:
                loss = msg.get('loss', 1.0)
                has_tc = msg.get('train_time_curve') is not None
                tc_len = len(msg.get('train_time_curve') or [])
                wc_last = msg.get('wall_clock_curve', [None])[-1] if msg.get('wall_clock_curve') else 'N/A'
                print(f"   {m_name:<14} | {dataset:<18} | s{seed} | loss={loss:.6f} | runtime={'OK '+str(tc_len)+'pts' if has_tc else 'MISS'} | wc_total={wc_last}")
                done += 1
            elif success and msg is None:
                done += 1
            else:
                print(f"   {m_name:<14} | {dataset:<18} | s{seed} | ERROR: {msg}")
                errors += 1

    elapsed = time.time() - t0
    print(f"\nDone: {done} computed, {errors} errors | Time: {elapsed:.1f}s")
    print("Наступний крок: python3 analyze_visuals.py L5_FCNET")


if __name__ == '__main__':
    main()
