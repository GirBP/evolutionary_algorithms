"""HPO Benchmark — Dataset Registry
Кожен датасет повертає (Xt, Xv, yt, yv, task_type).
task_type: 'regression' або 'classification'.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Registry ─────────────────────────────────────────────────────────────────
DATASET_INFO = {
    # L0 (synthetic)
    'synth_regression':     {'task': 'regression',      'tier': 'L0'},
    'synth_classification': {'task': 'classification',  'tier': 'L0'},
    'synth_friedman':       {'task': 'regression',      'tier': 'L0'},
    # L1 (real, tabular)
    'california':           {'task': 'regression',      'tier': 'L1'},
    'diabetes':             {'task': 'regression',      'tier': 'L1'},
    'digits':               {'task': 'classification',  'tier': 'L1'},
    'breast_cancer':        {'task': 'classification',  'tier': 'L1'},
}

def get_task_type(name):
    return DATASET_INFO[name]['task']

def load_dataset(name, seed):
    """Завантажує датасет і розбиває на train/val.
    Returns: (Xt, Xv, yt, yv)
    """
    X, y = _load_raw(name)
    # Нормалізація
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return train_test_split(X, y, test_size=0.3, random_state=seed)


def _load_raw(name):
    if name == 'synth_regression':
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
        return X, y

    elif name == 'synth_classification':
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=500, n_features=15, n_informative=8,
                                   n_redundant=3, random_state=42)
        return X, y

    elif name == 'synth_friedman':
        from sklearn.datasets import make_friedman1
        X, y = make_friedman1(n_samples=500, n_features=10, noise=1.0, random_state=42)
        return X, y

    elif name == 'california':
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing()
        return d.data, d.target

    elif name == 'diabetes':
        from sklearn.datasets import load_diabetes
        d = load_diabetes()
        return d.data, d.target

    elif name == 'digits':
        from sklearn.datasets import load_digits
        d = load_digits()
        return d.data, d.target

    elif name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        return d.data, d.target

    else:
        raise ValueError(f"Unknown dataset: {name}")
