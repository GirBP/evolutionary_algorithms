# ex08_run.py — Modular Ex08 Benchmark Runner
# Кожен метод зберігає результати в data/results_{method_key}.json
# Teacher моделі кешуються в data/base_models/
# Використовує RCU (profile_rcu) замість etalon
#
# Приклади:
#   python ex08_run.py --method magnitude          # один метод
#   python ex08_run.py --method all                # всі методи
#   python ex08_run.py --method wanda-cnn -q       # швидкий тест
#   python ex08_run.py --method all --list         # список методів
from __future__ import annotations

import sys
import os
import json
import copy
import time
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# --- CPU parallelization: enforce single-thread to match RCU protocol ---
# Workers set torch.set_num_threads(1) explicitly in run_one_task().
# Module-level stays at 1 to prevent hidden multi-threading leaking
# into teacher training or early joblib workers.
torch.set_num_threads(1)

# --- Paths ---
CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'common'))
sys.path.insert(0, str(ROOT / 'Ex07' / 'code'))
sys.path.insert(0, str(CODE_DIR))

from common import ensure_experiment_dependencies
ensure_experiment_dependencies()

from common.rcu import profile_rcu, setup_rcu_worker
from or08_01 import (CompactCNN, create_model, set_model_class, set_seed,
                      get_dataloaders, evaluate_full, measure_actual_sparsity)
import or08_01 as _or08_mod  # for monkey-patching evaluate_full

# --- Config ---
import config_experiment
import config_test
import config_baseline
import config_preview
import config_circles
import config_spirals
import config_blobs
import config_cnn
import config_resnet

# --- Method registry (triggers all method registrations on import) ---
from methods import list_methods, get_method, get_display_name
_reg_keys = list_methods  # alias for save_method_results fallback

DATA_DIR = EXPERIMENT_DIR / 'data'
BASE_MODELS_DIR = DATA_DIR / 'base_models'


# =====================================================
# TEACHER CACHE
# =====================================================
def get_or_train_teacher(seed: int, config: dict) -> dict:
    """Train teacher or load from cache."""
    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = config.get('model', 'CompactCNN')
    cache_path = BASE_MODELS_DIR / f"teacher_{model_name}_seed{seed}.pt"

    if cache_path.exists():
        print(f"  [Teacher seed={seed}] Loading from cache...", end=" ", flush=True)
        state = torch.load(cache_path, map_location='cpu', weights_only=True)
        # Quick eval to print teacher F1
        set_model_class(model_name)
        teacher = create_model()
        teacher.load_state_dict(state)
        _, test_dl_tmp = get_dataloaders(seed, config.get('dataset', 'fashionmnist'))[0], get_dataloaders(seed, config.get('dataset', 'fashionmnist'))[2]
        _, f1_t, _, _ = evaluate_full(teacher, test_dl_tmp)
        print(f"F1={f1_t:.4f}")
        return state

    print(f"  [Teacher seed={seed}] Training {config['epochs_pretrain']} epochs...", flush=True)
    set_seed(seed)
    train_dl, _, test_dl = get_dataloaders(seed, config.get('dataset', 'fashionmnist'))
    set_model_class(model_name)
    teacher = create_model()
    n_epochs = config['epochs_pretrain']

    opt = optim.Adam(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)
    crit = torch.nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        teacher.train()
        for X, y in train_dl:
            opt.zero_grad()
            loss = crit(teacher(X), y)
            loss.backward()
            opt.step()
        scheduler.step()

    _, f1_teacher, _, _ = evaluate_full(teacher, test_dl)
    print(f"  [Teacher seed={seed}] F1={f1_teacher:.4f}")

    state = copy.deepcopy(teacher.state_dict())
    torch.save(state, cache_path)
    print(f"  [Teacher seed={seed}] Done (cached)")
    return state


# =====================================================
# PER-SEED-SPARSITY WORKER
# =====================================================
def run_one_task(method_key: str, teacher_state: dict, sp: float,
                 seed: int, config: dict) -> dict:
    """Run one (method, seed, sparsity) task with RCU profiling.

    Captures actual sparsity via monkey-patching evaluate_full to intercept
    the model when the method calls it for final evaluation.
    """
    setup_rcu_worker()
    # Set active model architecture from config in child process
    model_name = config.get('model', 'CompactCNN')
    set_model_class(model_name)

    # Full thread isolation (matching Ex01 protocol)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    train_dl, val_dl, test_dl = get_dataloaders(seed, config.get('dataset', 'fashionmnist'))
    method_info = get_method(method_key)
    run_func = method_info['func']

    # Monkey-patch evaluate_full in EVERY method module to capture actual sparsity
    # AND the final model for held-out validation.
    _captured = {'actual_sp': None, 'model': None}
    _original_eval = evaluate_full  # the real function

    def _eval_with_sparsity(model, loader):
        _captured['actual_sp'] = measure_actual_sparsity(model)
        _captured['model'] = model  # capture for val evaluation
        return _original_eval(model, loader)

    # Collect all method modules that have evaluate_full in their namespace
    import methods as _methods_pkg
    _patched_modules = []
    for attr_name in dir(_methods_pkg):
        mod = getattr(_methods_pkg, attr_name, None)
        if hasattr(mod, 'evaluate_full') and getattr(mod, 'evaluate_full', None) is _original_eval:
            _patched_modules.append(mod)
    # Also patch or08_01 module itself (for methods that call it via module)
    _patched_modules.append(_or08_mod)

    def _payload():
        # Patch all modules
        for mod in _patched_modules:
            mod.evaluate_full = _eval_with_sparsity
        try:
            return run_func(teacher_state, sp, seed, config, train_dl, test_dl)
        finally:
            for mod in _patched_modules:
                mod.evaluate_full = _original_eval

    result_dict, rcu, anchor_pre_ns, anchor_post_ns = profile_rcu(_payload)

    anchor_avg_ms = (anchor_pre_ns + anchor_post_ns) / 2.0 / 1e6  # ns → ms

    actual_sp = _captured['actual_sp']
    if actual_sp is None:
        actual_sp = sp  # fallback if method didn't call evaluate_full

    # Held-out validation: evaluate on val_dl (never seen by method)
    f1_test = result_dict['F1']  # F1 on test_dl (used by method)
    f1_val = f1_test  # fallback
    if val_dl is not None and _captured['model'] is not None:
        _, f1_val, _, _ = _original_eval(_captured['model'], val_dl)

    return {
        'Seed': seed,
        'Sparsity': sp,
        'Actual_Sparsity': round(actual_sp, 6),
        'Method': method_info['display_name'],
        'F1': f1_val,           # held-out validation F1
        'F1_test': f1_test,     # test F1 (for reference)
        'Time_RCU': rcu,
        'Anchor_pre_ns': int(anchor_pre_ns),
        'Anchor_post_ns': int(anchor_post_ns),
        'Anchor_avg_ms': anchor_avg_ms,
    }


# =====================================================
# SAVE / LOAD PER-METHOD RESULTS
# =====================================================
def save_method_results(method_key: str, rows: list, config: dict):
    """Save results for one method — INCREMENTAL: merges with existing sparsity levels."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"results_{method_key}.json"

    # Load existing results (if any) and merge
    existing_rows = []
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        existing_rows = existing_data.get('results', [])

    # Build set of (seed, sparsity) keys from new rows to replace
    new_keys = {(r['Seed'], r['Sparsity']) for r in rows}
    # Keep existing rows that are NOT being replaced
    merged = [r for r in existing_rows if (r['Seed'], r['Sparsity']) not in new_keys]
    merged.extend(rows)
    # Sort by sparsity then seed for readability
    merged.sort(key=lambda r: (r['Sparsity'], r['Seed']))

    data = {
        'method_key': method_key,
        'display_name': get_display_name(method_key) if method_key in _reg_keys() else method_key,
        'results': merged,
        'metadata': {
            'seeds': config['seeds'],
            'sparsities': sorted(set(r['Sparsity'] for r in merged)),
            'config': {k: v for k, v in config.items()
                       if isinstance(v, (int, float, str, list, bool))},
        },
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)
    n_sp = len(set(r['Sparsity'] for r in merged))
    print(f"  Saved: {path} ({len(merged)} results, {n_sp} sparsity levels)")


def load_all_results() -> pd.DataFrame:
    """Scan data/results_*.json and merge into a single DataFrame."""
    all_rows = []
    if not DATA_DIR.exists():
        return pd.DataFrame()
    for p in sorted(DATA_DIR.glob("results_*.json")):
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f)
        all_rows.extend(data.get('results', []))
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# =====================================================
# BASE CNN (runs without per-method save)
# =====================================================
def run_base_cnn(config: dict) -> list:
    """Train & eval plain CompactCNN (no pruning). Saves to results_base-cnn.json"""
    rows = []
    for run_idx, seed in enumerate(config['seeds'][:config.get('n_runs_base', 3)]):
        set_seed(seed)
        # Thread isolation (matching Ex01)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        train_dl, _, test_dl = get_dataloaders(seed, config.get('dataset', 'fashionmnist'))
        print(f"  [Base CNN run {run_idx+1}] ", end="", flush=True)

        def _payload():
            teacher = create_model()
            opt = optim.SGD(teacher.parameters(), lr=0.01, momentum=0.9)
            crit = torch.nn.CrossEntropyLoss()
            for _ in range(config['epochs_pretrain']):
                for X, y in train_dl:
                    opt.zero_grad()
                    loss = crit(teacher(X), y)
                    loss.backward()
                    opt.step()
            _, f1, _, _ = evaluate_full(teacher, test_dl)
            return f1

        f1, rcu, anchor_pre_ns, anchor_post_ns = profile_rcu(_payload)
        anchor_avg_ms = (anchor_pre_ns + anchor_post_ns) / 2.0 / 1e6
        print(f"F1={f1:.4f}  RCU={rcu:.1f}")

        rows.append({
            'Seed': seed, 'Sparsity': 0.0, 'Method': 'Base CNN',
            'F1': f1, 'Time_RCU': rcu,
            'Anchor_pre_ns': int(anchor_pre_ns),
            'Anchor_post_ns': int(anchor_post_ns),
            'Anchor_avg_ms': anchor_avg_ms,
        })

    save_method_results('base-cnn', rows, config)
    return rows


# =====================================================
# MAIN
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Ex08 Modular Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available methods: {', '.join(list_methods())}"
    )
    parser.add_argument(
        '--method', '-m', type=str, default='all',
        help="Method key to run (or 'all' for everything). Use --list to see options."
    )
    parser.add_argument('--list', action='store_true', help="List available methods and exit")
    parser.add_argument('-q', '--quick', action='store_true', help="Run in fast test mode")
    parser.add_argument('--config', type=str, default=None,
                        choices=['experiment', 'test', 'baseline', 'preview',
                                 'circles', 'spirals', 'blobs', 'cnn', 'resnet'],
                        help="Config to use (overrides --quick)")
    parser.add_argument('--base', action='store_true', help="Also run Base CNN evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        print("Available methods:")
        for key in list_methods():
            m = get_method(key)
            print(f"  {key:30s} → {m['display_name']}")
        return

    # Config selection: --config overrides --quick
    config_map = {
        'experiment': (config_experiment.CONFIG, config_experiment.MODE_LABEL),
        'test': (config_test.CONFIG, config_test.MODE_LABEL),
        'baseline': (config_baseline.CONFIG, config_baseline.MODE_LABEL),
        'preview': (config_preview.CONFIG, config_preview.MODE_LABEL),
        'circles': (config_circles.CONFIG, config_circles.MODE_LABEL),
        'spirals': (config_spirals.CONFIG, config_spirals.MODE_LABEL),
        'blobs': (config_blobs.CONFIG, config_blobs.MODE_LABEL),
        'cnn': (config_cnn.CONFIG, config_cnn.MODE_LABEL),
        'resnet': (config_resnet.CONFIG, config_resnet.MODE_LABEL),
    }
    if args.config:
        config, mode_label = config_map[args.config]
    elif args.quick:
        config, mode_label = config_map['test']
    else:
        config, mode_label = config_map['experiment']

    # Per-dataset/model data directory
    global DATA_DIR, BASE_MODELS_DIR
    ds = config.get('dataset', 'moons')
    model_name_lower = config.get('model', 'SimpleMLP').lower()
    # Determine subdirectory: for non-default configs, use dataset or model name
    if ds == 'moons' and model_name_lower == 'simplemlp':
        pass  # keep default DATA_DIR = data/
    elif ds in ('circles', 'spirals', 'blobs'):
        DATA_DIR = EXPERIMENT_DIR / 'data' / ds
        BASE_MODELS_DIR = DATA_DIR / 'base_models'
    else:
        # CNN or other model configs
        subdir = args.config if args.config else model_name_lower
        DATA_DIR = EXPERIMENT_DIR / 'data' / subdir
        BASE_MODELS_DIR = DATA_DIR / 'base_models'
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Set active model architecture from config
    model_name = config.get('model', 'CompactCNN')
    set_model_class(model_name)
    model_cls = create_model().__class__
    n_params = sum(p.numel() for p in create_model().parameters())

    print(f"{'=' * 60}")
    print(f"Ex08 Modular Benchmark — {mode_label}")
    print(f"Model: {model_name} ({n_params:,} params)")
    print(f"Seeds: {config['seeds']}, Sparsities: {config['sparsities']}")
    print(f"{'=' * 60}")

    # Determine methods to run
    if args.method == 'all':
        # Use config-level filter if specified
        if 'methods' in config:
            method_keys = config['methods']
        else:
            method_keys = list_methods()
    else:
        method_keys = [args.method]

    if args.base:
        print("\n--- Phase 0: Base CNN ---")
        run_base_cnn(config)

    # --- Phase 1: Train & cache teachers (parallel) ---
    print("\n--- Phase 1: Caching teachers ---")
    from joblib import Parallel, delayed
    teacher_states = Parallel(n_jobs=len(config['seeds']))(
        delayed(get_or_train_teacher)(seed, config)
        for seed in config['seeds']
    )
    teachers = dict(zip(config['seeds'], teacher_states))

    # --- Phase 2: Run methods ---
    from joblib import Parallel, delayed

    n_jobs_outer = min(
        config.get('max_workers', multiprocessing.cpu_count()),
        multiprocessing.cpu_count()
    )

    # Methods with internal CMA-ES parallelism — run sequentially at outer level
    # to avoid CPU oversubscription (inner Parallel already uses all cores)
    EVO_METHODS = {
        'evo-synflow', 'evo-synflow-adaptive', 'evo-synflow-symwanda',
        'evo-synflow-energycomp', 'evo-synflow-energycomp-a',
        'evostruct', 'evo-hmt',
        'evo-hmt-no-erk', 'evo-hmt-no-bn', 'evo-hmt-mag-only',
    }

    for method_key in method_keys:
        method_info = get_method(method_key)
        display = method_info['display_name']
        print(f"\n{'=' * 60}")
        print(f"=== METHOD: {display} ({method_key}) ===")
        print(f"{'=' * 60}")

        try:
            # Early stopping: if enabled, run sparsities sequentially and skip
            # after N consecutive failures
            es_max = config.get('early_stop_consecutive_fails', 0)
            es_thresh = config.get('early_stop_f1_threshold', 0.15)

            if es_max > 0:
                # Sequential per-sparsity with early stopping
                results = []
                consecutive_fails = 0
                stopped_at = None
                for sp in config['sparsities']:
                    if stopped_at:
                        # Fill remaining with sentinel
                        for seed in config['seeds']:
                            results.append({
                                'Method': display, 'Seed': seed, 'Sparsity': sp,
                                'F1': 0.0, 'Actual_Sparsity': sp,
                                'Time_RCU': 0, 'Skipped': True,
                            })
                        continue

                    sp_tasks = [(seed, sp) for seed in config['seeds']]
                    effective_n_jobs = 1 if method_key in EVO_METHODS else n_jobs_outer
                    sp_results = Parallel(n_jobs=effective_n_jobs)(
                        delayed(run_one_task)(
                            method_key, teachers[seed], sp, seed, config
                        ) for seed, sp in sp_tasks
                    )
                    results.extend(sp_results)

                    avg_f1 = np.mean([r['F1'] for r in sp_results])
                    if avg_f1 < es_thresh:
                        consecutive_fails += 1
                        if consecutive_fails >= es_max:
                            stopped_at = sp
                            print(f"   EARLY STOP: {display} broke at sp={sp:.2f} "
                                  f"(F1={avg_f1:.3f}, {es_max} consecutive fails)")
                    else:
                        consecutive_fails = 0
            else:
                # Original: all tasks in parallel
                tasks = [
                    (seed, sp) for seed in config['seeds']
                    for sp in config['sparsities']
                ]
                effective_n_jobs = 1 if method_key in EVO_METHODS else n_jobs_outer
                results = Parallel(n_jobs=effective_n_jobs)(
                    delayed(run_one_task)(
                        method_key, teachers[seed], sp, seed, config
                    )
                    for seed, sp in tasks
                )

            # Filter out skipped results before saving
            real_results = [r for r in results if not r.get('Skipped')]

            # Print results
            for r in real_results:
                sp_delta = r['Actual_Sparsity'] - r['Sparsity']
                flag = ' ' if abs(sp_delta) > 0.01 else ''
                print(f"  Seed={r['Seed']} Sp={r['Sparsity']:.2f} "
                      f"ActSp={r['Actual_Sparsity']:.4f} "
                      f"F1={r['F1']:.4f} RCU={r['Time_RCU']:.1f}{flag}")

            # Save per-method
            save_method_results(method_key, real_results, config)

        except Exception as e:
            print(f"   SKIPPED: {display} — {type(e).__name__}: {e}")

    # --- Final summary ---
    print(f"\n{'=' * 60}")
    print("=== SUMMARY ===")
    df = load_all_results()
    if not df.empty and 'Time_RCU' in df.columns:
        agg = df.groupby(['Sparsity', 'Method']).agg(
            F1_mean=('F1', 'mean'), F1_std=('F1', 'std'),
            RCU_mean=('Time_RCU', 'mean'), RCU_std=('Time_RCU', 'std')
        ).reset_index()
        agg['F1'] = agg.apply(
            lambda r: f"{r['F1_mean']:.4f} ± {r['F1_std']:.4f}"
            if pd.notna(r['F1_std']) else f"{r['F1_mean']:.4f}", axis=1
        )
        agg['RCU'] = agg.apply(
            lambda r: f"{r['RCU_mean']:.1f} ± {r['RCU_std']:.1f}"
            if pd.notna(r['RCU_std']) else f"{r['RCU_mean']:.1f}", axis=1
        )
        print(agg[['Sparsity', 'Method', 'F1', 'RCU']].to_string(index=False))
    print(f"\nDone. Per-method results in {DATA_DIR}/")


if __name__ == '__main__':
    main()
