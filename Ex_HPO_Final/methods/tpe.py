"""TPE (Optuna) — Industry Standard Surrogate Baseline"""
import optuna
import logging

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run(seed, obj_fn, dim, budget):
    """
    Запускає Tree-structured Parzen Estimator (TPE) через Optuna.
    TPE — індустріальний стандарт для пошуку з сурогатною моделлю.
    """
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(sampler=sampler, direction='minimize')
    
    curve = []
    bl = float('inf')
    x_best = None
    
    def objective(trial):
        # Відновлюємо вектор v ∈ [0, 1]^dim
        v = [trial.suggest_float(f'v{i}', 0.0, 1.0) for i in range(dim)]
        import numpy as np
        v = np.array(v)
        nonlocal bl, x_best
        fl = obj_fn(v)
        if fl < bl:
            bl = fl
            x_best = v.copy()
        curve.append(bl)
        return fl
        
    study.optimize(objective, n_trials=budget, n_jobs=1)
    
    # Якщо Optuna якось завершила зарано (наприклад через помилкові return)
    while len(curve) < budget:
        curve.append(curve[-1] if curve else 1e9)
        
    import numpy as np
    x_best_arr = np.array(x_best) if x_best is not None else None
        
    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best_arr.tolist() if x_best_arr is not None else None}
