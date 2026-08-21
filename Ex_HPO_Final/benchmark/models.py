"""HPO Benchmark — Model Space Registry
Кожна модель визначає: dim, decode(v) -> params, regressor/classifier class.
Усі вектори v ∈ [0,1]^dim.
"""
import numpy as np
from sklearn.ensemble import (HistGradientBoostingRegressor, HistGradientBoostingClassifier,
                              RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import root_mean_squared_error, accuracy_score

# ═══════════════════════════════════════════════════════════════════════════════
# Decode functions: v ∈ [0,1]^dim → sklearn params dict
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_hgb(v):
    return {
        'learning_rate': 10**(v[0] * 2 + (-3)),           # [1e-3, 1e-1]
        'max_iter': int(20 + v[1] * 230),                  # [20, 250]
        'max_leaf_nodes': int(10 + v[2] * 90),             # [10, 100]
        'max_depth': int(3 + v[3] * 12) if v[3] > 0.1 else None,  # [3,15] or None
        'min_samples_leaf': int(1 + v[4] * 49),            # [1, 50]
        'l2_regularization': 10**(v[5] * 5 + (-3)) if v[5] > 0.1 else 0,  # [1e-3,100]
        'max_features': 0.3 + v[6] * 0.7,                 # [0.3, 1.0]
        'max_bins': int(10 + v[7] * 245),                  # [10, 255]
        'early_stopping': False,
        'scoring': 'loss',
        'random_state': 42,
    }

def _decode_rf(v):
    return {
        'n_estimators': int(10 + v[0] * 290),              # [10, 300]
        'max_depth': int(2 + v[1] * 28) if v[1] > 0.05 else None,  # [2,30] or None
        'min_samples_split': int(2 + v[2] * 18),           # [2, 20]
        'min_samples_leaf': int(1 + v[3] * 19),            # [1, 20]
        'max_features': 0.3 + v[4] * 0.7,                 # [0.3, 1.0]
        'bootstrap': v[5] > 0.5,
        'random_state': 42,
        'n_jobs': 1,
    }

def _decode_mlp(v):
    # Вибір кількості шарів та нейронів
    n_layers = int(1 + v[0] * 2)  # 1, 2, or 3 layers
    n_neurons = int(16 + v[1] * 240)  # [16, 256]
    if n_layers == 1:
        hidden = (n_neurons,)
    elif n_layers == 2:
        hidden = (n_neurons, n_neurons // 2)
    else:
        hidden = (n_neurons, n_neurons // 2, n_neurons // 4)

    return {
        'hidden_layer_sizes': hidden,
        'learning_rate_init': 10**(v[2] * 3 + (-4)),       # [1e-4, 1e-1]
        'alpha': 10**(v[3] * 4 + (-5)),                    # [1e-5, 1e-1]
        'batch_size': int(16 + v[4] * 240),                # [16, 256]
        'max_iter': int(50 + v[5] * 450),                  # [50, 500]
        'activation': 'relu' if v[5] > 0.5 else 'tanh',
        'random_state': 42,
    }

def _decode_svm(v):
    return {
        'C': 10**(v[0] * 5 + (-2)),                        # [1e-2, 1e3]
        'gamma': 10**(v[1] * 5 + (-4)),                    # [1e-4, 1e1]
        'kernel': 'rbf',
    }

def _decode_svm_reg(v):
    params = _decode_svm(v)
    params['epsilon'] = 10**(v[2] * 3 + (-3))              # [1e-3, 1.0]
    return params

def _decode_gb(v):
    return {
        'n_estimators': int(20 + v[0] * 280),              # [20, 300]
        'learning_rate': 10**(v[1] * 2.5 + (-3)),          # [1e-3, 0.3]
        'max_depth': int(2 + v[2] * 8),                    # [2, 10]
        'min_samples_split': int(2 + v[3] * 18),           # [2, 20]
        'min_samples_leaf': int(1 + v[4] * 14),            # [1, 15]
        'subsample': 0.5 + v[5] * 0.5,                    # [0.5, 1.0]
        'max_features': 0.3 + v[6] * 0.7,                 # [0.3, 1.0]
        'random_state': 42,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Model Registry
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    'hgb': {
        'dim': 8,
        'decode': _decode_hgb,
        'reg': HistGradientBoostingRegressor,
        'clf': HistGradientBoostingClassifier,
    },
    'rf': {
        'dim': 6,
        'decode': _decode_rf,
        'reg': RandomForestRegressor,
        'clf': RandomForestClassifier,
    },
    'mlp': {
        'dim': 6,
        'decode': _decode_mlp,
        'reg': MLPRegressor,
        'clf': MLPClassifier,
    },
    'svm': {
        'dim': 3,
        'decode_reg': _decode_svm_reg,
        'decode_clf': _decode_svm,
        'reg': SVR,
        'clf': SVC,
    },
    'gb': {
        'dim': 7,
        'decode': _decode_gb,
        'reg': GradientBoostingRegressor,
        'clf': GradientBoostingClassifier,
    },
}


def get_model_space(model_name, task_type):
    """
    Returns (dim, obj_fn_factory)
    obj_fn_factory(Xt, yt, Xv, yv) -> obj_fn(v) -> float (loss to minimize)
    """
    spec = MODEL_REGISTRY[model_name]
    dim = spec['dim']

    # SVM has separate decode for reg/clf
    if model_name == 'svm':
        decode = spec['decode_reg'] if task_type == 'regression' else spec['decode_clf']
    else:
        decode = spec['decode']

    ModelClass = spec['reg'] if task_type == 'regression' else spec['clf']

    def make_objective(Xt, yt, Xv, yv):
        import time
        def objective(v):
            st = time.thread_time_ns()
            try:
                params = decode(v)
                model = ModelClass(**params)
                model.fit(Xt, yt)
                if task_type == 'regression':
                    res = float(root_mean_squared_error(yv, model.predict(Xv)))
                else:
                    res = 1.0 - float(accuracy_score(yv, model.predict(Xv)))
            except Exception:
                res = 1e6  # Penalty for crashed configs
            objective.total_time_ns += (time.thread_time_ns() - st)
            return res
        
        # Ініціалізація лічильника
        objective.total_time_ns = 0
        return objective

    return dim, make_objective
