"""
HPO Benchmark — Уніфікована ініціалізація (Sobol QMC).
=====================================================
Усі методи, що мають фазу "random init", використовують цю функцію
для забезпечення рівних стартових умов.

- Sobol: квазівипадкова послідовність з рівномірним покриттям [0,1]^d
- Детермінована: фіксований seed → однаковий набір точок
- Scrambled: Owen scrambling запобігає корелятивним артефактам
"""
import numpy as np
from scipy.stats import qmc


def sobol_init(seed: int, dim: int, n_points: int) -> np.ndarray:
    """
    Генерує n_points квазівипадкових точок у [0,1]^dim
    через Scrambled Sobol послідовність.
    
    Args:
        seed: random seed (контролює scrambling)
        dim: розмірність простору
        n_points: кількість початкових точок
        
    Returns:
        np.ndarray shape (n_points, dim) у [0,1]^dim
    """
    sampler = qmc.Sobol(d=dim, seed=seed, scramble=True)
    # Sobol вимагає 2^k точок; генеруємо мінімальну потужність 2, обрізаємо
    n_pow2 = max(n_points, 2)
    k = int(np.ceil(np.log2(n_pow2)))
    points = sampler.random_base2(m=k)  # 2^k points
    return points[:n_points]
