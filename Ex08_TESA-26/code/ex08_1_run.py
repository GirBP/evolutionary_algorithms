#!/usr/bin/env python
# ex08_1_run.py — Ex08.1: TESA-26 vs DSA / LAMP / ERK
# Запуск: python ex08_1_run.py
#         python ex08_1_run.py --quick   (3 sparsities, 1 seed, for smoke-test)
#
# Результати: Ex08_1/data/results_{method}.json
# Зведена таблиця + CSV виводяться після завершення.

from __future__ import annotations
import sys, os, json, copy, argparse
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

torch.set_num_threads(1)

# ── Paths ──────────────────────────────────────────────────────────────
THIS_DIR   = Path(__file__).resolve().parent          # Ex08/code/
EX08_DIR   = THIS_DIR.parent                          # Ex08/
ROOT       = EX08_DIR.parent                          # cs_dev/
EX08_1_DIR = ROOT / 'Ex08_1'                         # Ex08_1/ (results go here)
DATA_DIR   = EX08_1_DIR / 'data'
BASE_DIR   = DATA_DIR / 'base_models'

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'common'))
sys.path.insert(0, str(THIS_DIR))

from common import ensure_experiment_dependencies
ensure_experiment_dependencies()

from common.rcu import profile_rcu, setup_rcu_worker
from or08_01 import (create_model, set_model_class, set_seed,
                     get_dataloaders, evaluate_full, measure_actual_sparsity)
import or08_01 as _or08_mod

import config_ex08_1 as cfg_module
from methods import list_methods, get_method, get_display_name


# ── Helpers ────────────────────────────────────────────────────────────
def get_or_train_teacher(seed: int, config: dict) -> dict:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    cache = BASE_DIR / f"teacher_SimpleMLP_seed{seed}.pt"
    set_model_class('SimpleMLP')

    if cache.exists():
        state = torch.load(cache, map_location='cpu', weights_only=True)
        print(f"  [Teacher {seed}] loaded from cache")
        return state

    print(f"  [Teacher {seed}] training {config['epochs_pretrain']} epochs…")
    set_seed(seed)
    train_dl, _, test_dl = get_dataloaders(seed, 'moons')
    teacher = create_model()
    opt = optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config['epochs_pretrain'], eta_min=1e-5)
    crit = torch.nn.CrossEntropyLoss()
    for _ in range(config['epochs_pretrain']):
        teacher.train()
        for X, y in train_dl:
            opt.zero_grad(); crit(teacher(X), y).backward(); opt.step()
        sch.step()
    _, f1, _, _ = evaluate_full(teacher, test_dl)
    print(f"  [Teacher {seed}] F1={f1:.4f}")
    state = copy.deepcopy(teacher.state_dict())
    torch.save(state, cache)
    return state


def run_one(method_key, teacher_state, sp, seed, config):
    setup_rcu_worker()
    set_model_class('SimpleMLP')
    os.environ.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    torch.set_num_threads(1)

    train_dl, val_dl, test_dl = get_dataloaders(seed, 'moons')
    run_func = get_method(method_key)['func']

    _cap = {'sp': None, 'model': None}
    _orig = evaluate_full

    def _patched(model, loader):
        _cap['sp'] = measure_actual_sparsity(model)
        _cap['model'] = model
        return _orig(model, loader)

    _or08_mod.evaluate_full = _patched
    import methods as _mpkg
    for attr in dir(_mpkg):
        mod = getattr(_mpkg, attr, None)
        if hasattr(mod, 'evaluate_full') and mod.evaluate_full is _orig:
            mod.evaluate_full = _patched

    try:
        def _payload():
            return run_func(teacher_state, sp, seed, config, train_dl, test_dl)
        result, rcu, apre, apost = profile_rcu(_payload)
    finally:
        _or08_mod.evaluate_full = _orig
        for attr in dir(_mpkg):
            mod = getattr(_mpkg, attr, None)
            if hasattr(mod, 'evaluate_full') and mod.evaluate_full is _patched:
                mod.evaluate_full = _orig

    act_sp = _cap['sp'] if _cap['sp'] is not None else sp
    f1_val = result['F1']
    if val_dl and _cap['model']:
        _, f1_val, _, _ = _orig(_cap['model'], val_dl)

    return {
        'Seed': seed, 'Sparsity': sp, 'Actual_Sparsity': round(act_sp, 6),
        'Method': get_method(method_key)['display_name'],
        'F1': f1_val, 'F1_test': result['F1'],
        'Time_RCU': rcu,
    }


def save_results(method_key, rows, config):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"results_{method_key}.json"
    existing = []
    if path.exists():
        existing = json.loads(path.read_text()).get('results', [])
    new_keys = {(r['Seed'], r['Sparsity']) for r in rows}
    merged = [r for r in existing if (r['Seed'], r['Sparsity']) not in new_keys] + rows
    merged.sort(key=lambda r: (r['Sparsity'], r['Seed']))
    path.write_text(json.dumps({
        'method_key': method_key,
        'results': merged,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }, indent=2, default=str))
    print(f"  Saved: {path} ({len(merged)} rows)")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()

    config = cfg_module.CONFIG.copy()
    if args.quick:
        config['sparsities'] = [0.70, 0.90, 0.95]
        config['seeds'] = [42]
        config['epochs_pretrain'] = 20
        config['max_evals'] = 20

    print(f"\n{'='*60}")
    print(f"Ex08.1 — {cfg_module.MODE_LABEL}")
    print(f"Methods: {config['methods']}")
    print(f"Sparsities: {config['sparsities']}")
    print(f"Seeds: {config['seeds']}")
    print(f"{'='*60}\n")

    set_model_class('SimpleMLP')

    # Phase 1: teachers
    from joblib import Parallel, delayed
    teachers = dict(zip(
        config['seeds'],
        Parallel(n_jobs=len(config['seeds']))(
            delayed(get_or_train_teacher)(s, config) for s in config['seeds']
        )
    ))

    # Phase 2: run methods
    all_rows = []
    for method_key in config['methods']:
        print(f"\n{'─'*50}")
        print(f"METHOD: {get_display_name(method_key)}")
        print(f"{'─'*50}")

        tasks = [(s, sp) for s in config['seeds'] for sp in config['sparsities']]
        n_jobs = 1 if method_key == 'tesa26' else min(config['max_workers'], len(tasks))

        rows = Parallel(n_jobs=n_jobs)(
            delayed(run_one)(method_key, teachers[s], sp, s, config)
            for s, sp in tasks
        )
        for r in sorted(rows, key=lambda x: (x['Sparsity'], x['Seed'])):
            print(f"  Sp={r['Sparsity']:.2f} Seed={r['Seed']} "
                  f"F1={r['F1']:.4f} RCU={r['Time_RCU']:.1f}")
        save_results(method_key, rows, config)
        all_rows.extend(rows)

    # Phase 3: summary
    df = pd.DataFrame(all_rows)
    print(f"\n{'='*60}")
    print("SUMMARY — Mean F1 by Method × Sparsity")
    print(f"{'='*60}")
    pivot = df.groupby(['Method', 'Sparsity'])['F1'].mean().unstack('Sparsity').round(3)
    print(pivot.to_string())

    # Friedman test
    from scipy import stats
    methods = df['Method'].unique()
    groups = [df[df['Method'] == m]['F1'].values for m in methods]
    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
        try:
            stat, p = stats.friedmanchisquare(*groups)
            print(f"\nFriedman: χ²={stat:.2f}, p={p:.4f}")
        except Exception as e:
            print(f"\nFriedman: {e}")

    # Mean rank
    print("\nMean F1 per method:")
    print(df.groupby('Method')['F1'].agg(['mean', 'std']).round(4).to_string())

    # Save CSV
    csv_path = EX08_1_DIR / 'ex08_1_results.csv'
    EX08_1_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")


if __name__ == '__main__':
    main()
