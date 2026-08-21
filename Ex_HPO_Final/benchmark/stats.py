"""HPO Benchmark — Statistical Utilities"""
import numpy as np
from scipy.stats import wilcoxon

def compute_aucc(curve, budget):
    """Area Under Convergence Curve (lower is better)."""
    c = list(curve)[:budget]
    while len(c) < budget:
        c.append(c[-1])
    return float(np.mean(c))

def compute_aucc_post_init(curve, budget, n_init=10):
    """AUCC після фази ініціалізації (коректне порівняння між методами
    з різними стратегіями init — Sobol, Random, LHS тощо).
    Рахує середнє тільки по кроках [n_init, budget).
    """
    c = list(curve)[:budget]
    while len(c) < budget:
        c.append(c[-1])
    post = c[n_init:]
    return float(np.mean(post)) if post else float(np.mean(c))


def safe_wilcoxon(a, b):
    """Wilcoxon signed-rank test з обробкою крайніх випадків."""
    a, b = np.array(a), np.array(b)
    diff = a - b
    if np.all(diff == 0) or len(a) < 5:
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return 1.0
