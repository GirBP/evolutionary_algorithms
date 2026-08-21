#!/usr/bin/env python3
"""
HPO Benchmark — Незалежний Запуск Одного Методу.

Використання:
    python3 run_method.py <method> <tier> [seeds] [--force]
    python3 run_method.py <method> <tier> [seeds] --dataset <name> --model <name> [--force]

Приклади:
    # Весь тір (всі датасети × всі моделі):
    python3 run_method.py shade L0
    python3 run_method.py sacma_v3 L1 20

    # Конкретний датасет + модель:
    python3 run_method.py sacma_v3 L1 10 --dataset california --model hgb
    python3 run_method.py shade L1 5 --dataset digits --model rf

    # Тільки один датасет, всі моделі:
    python3 run_method.py sacma_v3 L1 10 --dataset california

    # Тільки одна модель, всі датасети:
    python3 run_method.py sacma_v3 L1 10 --model svm

    # Перезапуск (ігнорувати кеш):
    python3 run_method.py sacma_v3 L1 20 --force

Результати зберігаються в results/<tier>/<method>__<dataset>__<model>__seed<NN>.json
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import importlib
import concurrent.futures
from datetime import datetime
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmark import TIERS
from benchmark.datasets import load_dataset, get_task_type
from benchmark.models import get_model_space
from benchmark.profiler import run_with_rcu


def load_method(method_name):
    """Динамічний імпорт методу з methods/<name>.py"""
    mod = importlib.import_module(f"methods.{method_name}")
    return mod.run


def result_path(tier, method, dataset, model, seed):
    d = os.path.join(os.path.dirname(__file__), "results", tier)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{method}__{dataset}__{model}__seed{seed:02d}.json")


def run_single_cell(method_name, method_fn, tier, dataset, model, seed, budget, force):
    """Запускає один {method × dataset × model × seed} і зберігає JSON."""
    path = result_path(tier, method_name, dataset, model, seed)

    if os.path.exists(path) and not force:
        return None  # cached

    # PD1 surrogate tasks (L3)
    if model == 'pd1':
        from benchmark.pd1_adapter import get_pd1_objective
        dim, make_obj = get_pd1_objective(dataset)
        obj_fn = make_obj()
    # YAHPO Gym surrogate tasks (L2)
    elif model == 'yahpo':
        from benchmark.yahpo_adapter import get_yahpo_objective
        dim, make_obj = get_yahpo_objective(dataset)
        obj_fn = make_obj()
    # L4: Real PyTorch micro-network training
    elif model == 'l4':
        from benchmark.l4_objective import get_l4_objective
        dim, make_obj = get_l4_objective(dataset)
        obj_fn = make_obj()
    # L5: FCNet tabular benchmark (Klein & Hutter)
    elif model == 'fcnet':
        from benchmark.fcnet_adapter import get_fcnet_objective
        dim, make_obj = get_fcnet_objective(dataset)
        obj_fn = make_obj()
    else:
        task_type = get_task_type(dataset)
        dim, make_obj = get_model_space(model, task_type)
        Xt, Xv, yt, yv = load_dataset(dataset, seed)
        obj_fn = make_obj(Xt, yt, Xv, yv)

    res, rcu_hpo, rcu_total = run_with_rcu(method_fn, seed, obj_fn, dim, budget)

    # RCU_train: час навчання фінальної моделі (з метаданих YAHPO, якщо доступно)
    rcu_train_best = getattr(obj_fn, 'last_train_time', None)

    # Wall-clock curve: кумулятивна сума часу навчання кожної конфіг. + RCU_hpo накладних
    # Дозволяє будувати Loss vs T_wall замість Loss vs n_eval
    train_time_curve = getattr(obj_fn, 'time_curve', None)
    wall_clock_curve = None
    if train_time_curve:
        # Фільтруємо None (failed evals) → замінюємо на 0
        cleaned = [t if t is not None else 0.0 for t in train_time_curve]
        cumsum = 0.0
        wall_clock_curve = []
        for t in cleaned:
            cumsum += t
            wall_clock_curve.append(round(cumsum, 3))

    # Processing x_best
    x_best = res.get('x_best', None)
    best_config = None
    test_metrics = {}

    if x_best is not None:
        if hasattr(obj_fn, 'decode'):
            try:
                best_config = obj_fn.decode(np.array(x_best))
            except Exception as e:
                import traceback
                traceback.print_exc()
                best_config = None

        if hasattr(obj_fn, 'get_test_metrics'):
            try:
                test_metrics = obj_fn.get_test_metrics(np.array(x_best))
                # Convert np values to python types for json serialization
                for k, v in test_metrics.items():
                    if hasattr(v, 'item'):
                        test_metrics[k] = v.item()
            except Exception as e:
                import traceback
                traceback.print_exc()
                pass

    record = {
        'method': method_name,
        'dataset': dataset,
        'model': model,
        'seed': seed,
        'loss': res['loss'],
        'curve': res['curve'],
        'x_best': x_best,                          # [NEW] Best found hyperparameter vector
        'best_config': best_config,                # [NEW] Decoded configuration dict
        'test_metrics': test_metrics,              # [NEW] Dict containing test loss/runtime/memory
        'rcu_hpo': rcu_hpo,                       # RCU чистого алгоритму (безрозмірне, відносно anchor)
        'rcu_total': rcu_total,                    # RCU повного циклу (алгоритм + тренування, безрозмірне)
        'rcu_train_best': rcu_train_best,          # RCU навчання найкращої конфігурації
        'train_time_curve': train_time_curve,      # [NEW] Час тренування кожної конфіг. (сек)
        'wall_clock_curve': wall_clock_curve,      # [NEW] Кумулятивний wall-clock (сек) до i-ї оцінки
        'timestamp': datetime.now().isoformat(),
    }

    with open(path, 'w') as f:
        json.dump(record, f, indent=2)

    return record


def parse_args(argv):
    method_name = argv[1]
    tier = argv[2].upper()
    force = '--force' in argv

    if tier not in TIERS:
        print(f"Error: unknown tier '{tier}'. Use: {list(TIERS.keys())}")
        sys.exit(1)

    tier_cfg = TIERS[tier]

    # Seeds: --seeds 984,1457,3544 або числове значення після tier
    seeds_list = None
    if '--seeds' in argv:
        idx = argv.index('--seeds')
        if idx + 1 < len(argv):
            seeds_list = [int(x) for x in argv[idx + 1].split(',')]

    if seeds_list is None:
        n_seeds = tier_cfg['default_seeds']
        for a in argv[3:]:
            if a.isdigit():
                n_seeds = int(a)
                break
        seeds_list = list(range(n_seeds))

    # --dataset filter
    ds_filter = None
    if '--dataset' in argv:
        idx = argv.index('--dataset')
        if idx + 1 < len(argv):
            ds_filter = argv[idx + 1]

    # --model filter
    mdl_filter = None
    if '--model' in argv:
        idx = argv.index('--model')
        if idx + 1 < len(argv):
            mdl_filter = argv[idx + 1]

    datasets = [ds_filter] if ds_filter else tier_cfg['datasets']
    models = [mdl_filter] if mdl_filter else tier_cfg['models']
    budget = tier_cfg['budget']

    return method_name, tier, seeds_list, datasets, models, budget, force


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    method_name, tier, seeds_list, datasets, models, budget, force = parse_args(sys.argv)
    method_fn = load_method(method_name)

    total_cells = len(datasets) * len(models) * len(seeds_list)
    print(f"{'='*70}")
    print(f"HPO Benchmark: {method_name} @ {tier} (Sequential Debug Mode)")
    print(f"Datasets: {datasets}")
    print(f"Models:   {models}")
    print(f"Seeds:    {seeds_list} | Budget: {budget} | Force: {force}")
    print(f"Total cells: {total_cells}")
    print(f"{'='*70}")
    print(f"Попередження: Ви використовуєте run_method.py напряму. Це послідовний режим без об'єднаного паралелізму. Для бойового запуску використовуйте run_benchmark.py")

    t0 = time.time()
    done, skipped = 0, 0

    for ds in datasets:
        for mdl in models:
            for s in seeds_list:
                try:
                    rec = run_single_cell(method_name, method_fn, tier, ds, mdl, s, budget, force)
                    if rec is None:
                        skipped += 1
                    else:
                        done += 1
                        rcu = rec.get('rcu_hpo', rec.get('rcu_method', 0))
                        print(f"   {ds:20s} | {mdl:4s} | seed {s:2d} | loss={rec['loss']:.4f} | RCU={rcu:.0f}")
                except Exception as e:
                    done += 1
                    import traceback
                    print(f"   {ds:20s} | {mdl:4s} | seed {s:2d} | ERROR: {e}")
                    traceback.print_exc()

    elapsed = int(time.time() - t0)
    print(f"\n{'='*70}")
    print(f"Done: {done} computed, {skipped} cached | Time: {elapsed}s")

if __name__ == '__main__':
    main()
