"""Bayesian Optimization (GP) — Industrial BO Baseline
Matern 5/2 kernel + Expected Improvement + L-BFGS-B acquisition optimization.
"""
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.exceptions import ConvergenceWarning
from scipy.optimize import minimize
from scipy.stats import norm
from benchmark.init import sobol_init

warnings.filterwarnings('ignore', category=ConvergenceWarning)


def expected_improvement(X, gpr, y_best, xi=0.01):
    """Computes EI at points X."""
    mu, sigma = gpr.predict(np.atleast_2d(X), return_std=True)
    with np.errstate(divide='warn'):
        imp = y_best - mu - xi
        Z = imp / (sigma + 1e-9)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
    return ei


def maximize_ei_lbfgsb(gpr, y_best, dim, rng, n_restarts=5, n_random=2000):
    """Максимізує EI через multi-start L-BFGS-B + random backup.

    1. Генеруємо n_random випадкових точок, знаходимо top-n_restarts
    2. Запускаємо L-BFGS-B з кожної стартової точки
    3. Повертаємо найкращу
    """
    bounds_list = [(0.0, 1.0)] * dim
    
    # Випадкове сканування для стартових точок
    X_random = rng.random((n_random, dim))
    ei_random = expected_improvement(X_random, gpr, y_best)
    
    # Top-n_restarts стартових точок для L-BFGS-B
    top_idx = np.argsort(-ei_random)[:n_restarts]
    
    best_x = X_random[top_idx[0]]
    best_ei = ei_random[top_idx[0]]
    
    for idx in top_idx:
        x0 = X_random[idx].copy()
        try:
            result = minimize(
                lambda x: -expected_improvement(x.reshape(1, -1), gpr, y_best)[0],
                x0,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': 50, 'ftol': 1e-9}
            )
            if result.success or result.fun < -best_ei:
                ei_val = -result.fun
                if ei_val > best_ei:
                    best_ei = ei_val
                    best_x = np.clip(result.x, 0, 1)
        except Exception:
            pass
    
    return best_x


def run(seed, obj_fn, dim, budget):
    """
    Класична BO з GP(Matern 5/2) + EI, оптимізація acquisition через L-BFGS-B.
    """
    rng = np.random.default_rng(seed)
    n_init = 10
    
    # ── Sobol Initialization ───────────────────────────────────────────
    init_pts = sobol_init(seed, dim, n_init)
    
    X_hist = []
    y_hist = []
    curve = []
    bl = float('inf')
    x_best = None

    for v in init_pts:
        y = obj_fn(v)
        X_hist.append(v)
        y_hist.append(y)
        if y < bl: 
            bl = y
            x_best = v.copy()
        curve.append(bl)
        
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2, random_state=seed, alpha=1e-4)

    # ── Bayesian Optimization Loop ───────────────────────────────────
    for ec in range(n_init, budget):
        Xh = np.array(X_hist)
        yh = np.array(y_hist)
        # Нормалізуємо y для стабільності GP
        y_mean = np.mean(yh)
        y_std = np.std(yh) + 1e-9
        yh_norm = (yh - y_mean) / y_std

        try:
            gpr.fit(Xh, yh_norm)
            y_best_norm = np.min(yh_norm)
            
            # L-BFGS-B оптимізація acquisition function
            next_x = maximize_ei_lbfgsb(gpr, y_best_norm, dim, rng,
                                        n_restarts=5, n_random=2000)
            
        except Exception:
            # Fallback
            next_x = rng.random(dim)

        y = obj_fn(next_x)
        
        X_hist.append(next_x)
        y_hist.append(y)
        if y < bl: 
            bl = y
            x_best = next_x.copy()
        curve.append(bl)

    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
