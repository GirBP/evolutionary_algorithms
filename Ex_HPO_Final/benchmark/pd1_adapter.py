"""
PD1 Benchmark Adapter — Surrogate-based objective for deep learning HPO.
=========================================================================
Перетворює дані Google PD1 (Transformer, ResNet, WideResNet, CNN)
у формат obj_fn(v) → loss, сумісний із нашим фреймворком.

Стратегія: XGBoost сурогат навчений на matched + unmatched даних PD1.
Це дозволяє нашим HPO-алгоритмам працювати з трансформерами та ResNet
без навчання реальних моделей (оцінка ~0.1 мс).

Автор: benchmark framework, 2025
"""

import os
import json
import gzip
import numpy as np
import pickle
from pathlib import Path

# Шлях до даних PD1
PD1_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pd1', 'pd1')

# 4D простір пошуку (спільний для всіх задач PD1)
PD1_DIM = 4
PD1_HP_KEYS = [
    'hps.lr_hparams.initial_value',       # learning rate
    'hps.lr_hparams.decay_steps_factor',   # lr schedule decay
    'hps.lr_hparams.power',               # polynomial power
    'hps.opt_hparams.momentum',           # optimizer momentum
]

# Діапазони для нормалізації v ∈ [0,1]^4 → реальні HP
PD1_HP_RANGES = {
    'hps.lr_hparams.initial_value':     (1e-5, 10.0, 'log'),     # логарифмічна шкала
    'hps.lr_hparams.decay_steps_factor': (0.01, 1.0, 'linear'),
    'hps.lr_hparams.power':             (0.1, 2.0, 'linear'),
    'hps.opt_hparams.momentum':         (0.1, 0.999, 'linear'),
}

# Доступні задачі PD1
PD1_TASKS = {
    # ResNet / WideResNet  (Computer Vision)
    'imagenet_resnet':          {'dataset': 'imagenet',       'model': 'resnet',              'task': 'classification'},
    'cifar10_wresnet':          {'dataset': 'cifar10',        'model': 'wide_resnet',         'task': 'classification'},
    'cifar100_wresnet':         {'dataset': 'cifar100',       'model': 'wide_resnet',         'task': 'classification'},
    'svhn_wresnet':             {'dataset': 'svhn_no_extra',  'model': 'wide_resnet',         'task': 'classification'},
    # CNN (Computer Vision)
    'mnist_cnn':                {'dataset': 'mnist',          'model': 'max_pooling_cnn',     'task': 'classification'},
    'fashion_cnn':              {'dataset': 'fashion_mnist',  'model': 'max_pooling_cnn',     'task': 'classification'},
    # Transformer  (NLP / Protein)
    'lm1b_transformer':         {'dataset': 'lm1b',          'model': 'transformer',         'task': 'language_model'},
    'translate_transformer':    {'dataset': 'translate_wmt',  'model': 'xformer_translate',   'task': 'translation'},
    'uniref50_transformer':     {'dataset': 'uniref50',       'model': 'transformer',         'task': 'protein_lm'},
}


def _decode_v_to_hps(v):
    """Перетворює v ∈ [0,1]^4 → словник реальних гіперпараметрів."""
    hps = {}
    for i, key in enumerate(PD1_HP_KEYS):
        lo, hi, scale = PD1_HP_RANGES[key]
        vi = np.clip(v[i], 0, 1)
        if scale == 'log':
            hps[key] = float(np.exp(np.log(lo) + vi * (np.log(hi) - np.log(lo))))
        else:
            hps[key] = float(lo + vi * (hi - lo))
    return hps


def _hps_to_v(hps_dict):
    """Зворотне перетворення: реальні HP → v ∈ [0,1]^4."""
    v = np.zeros(PD1_DIM)
    for i, key in enumerate(PD1_HP_KEYS):
        lo, hi, scale = PD1_HP_RANGES[key]
        val = hps_dict.get(key, (lo + hi) / 2)
        if scale == 'log':
            v[i] = (np.log(val) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            v[i] = (val - lo) / (hi - lo)
        v[i] = np.clip(v[i], 0, 1)
    return v


def _load_pd1_data(task_name):
    """Завантажує всі рядки PD1 для задачі (matched + unmatched, phase0 + phase1)."""
    spec = PD1_TASKS[task_name]
    target_ds = spec['dataset']
    target_model = spec['model']

    all_rows = []
    files = [
        'pd1_matched_phase0_results.jsonl.gz',
        'pd1_matched_phase1_results.jsonl.gz',
        'pd1_unmatched_phase0_results.jsonl.gz',
        'pd1_unmatched_phase1_results.jsonl.gz',
    ]

    for fname in files:
        fpath = os.path.join(PD1_DATA_DIR, fname)
        if not os.path.exists(fpath):
            continue
        with gzip.open(fpath, 'rt') as f:
            for line in f:
                r = json.loads(line)
                if r.get('dataset') == target_ds and r.get('model') == target_model:
                    if r.get('status') == 'done':
                        all_rows.append(r)

    return all_rows


def _build_surrogate(task_name, cache_dir=None):
    """
    Будує XGBoost сурогат для задачі PD1.
    Кешує навчену модель у pickle для повторного використання.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pd1', 'surrogates')
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f'{task_name}_surrogate.pkl')

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # Завантажуємо дані
    rows = _load_pd1_data(task_name)
    if len(rows) < 10:
        raise ValueError(f"PD1 task '{task_name}': too few rows ({len(rows)})")

    # Перетворюємо у масиви
    X = np.array([_hps_to_v({k: r[k] for k in PD1_HP_KEYS if k in r}) for r in rows])
    # Використовуємо best_valid/ce_loss як метрику (є для всіх задач)
    y = np.array([r.get('best_valid/ce_loss', r.get('valid/ce_loss', 1e6)) for r in rows])

    # Замінюємо NaN/inf
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]

    if len(X) < 10:
        raise ValueError(f"PD1 task '{task_name}': too few valid rows ({len(X)})")

    # Навчаємо ExtraTrees замість XGBoost (менше залежностей)
    from sklearn.ensemble import ExtraTreesRegressor
    surrogate = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=1,
    )
    surrogate.fit(X, y)

    # Кешуємо
    with open(cache_path, 'wb') as f:
        pickle.dump(surrogate, f)

    print(f"  PD1 surrogate built for '{task_name}': {len(X)} points, "
          f"y_range=[{y.min():.4f}, {y.max():.4f}]")

    return surrogate


def get_pd1_objective(task_name):
    """
    Повертає (dim, make_objective) для PD1 задачі.
    Інтерфейс ідентичний до benchmark/models.py → get_model_space().

    Використання:
        dim, make_obj = get_pd1_objective('imagenet_resnet')
        obj_fn = make_obj(None, None, None, None)  # для PD1 дані не потрібні
        loss = obj_fn(np.random.rand(4))
    """
    surrogate = _build_surrogate(task_name)

    def make_objective(Xt=None, yt=None, Xv=None, yv=None):
        import time
        def objective(v):
            st = time.thread_time_ns()
            v_clipped = np.clip(v, 0, 1).reshape(1, -1)
            try:
                loss = float(surrogate.predict(v_clipped)[0])
            except Exception:
                loss = 1e6
            objective.total_time_ns += (time.thread_time_ns() - st)
            return loss
        objective.total_time_ns = 0
        return objective

    return PD1_DIM, make_objective
