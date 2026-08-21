"""Random Search — Lower Bound Baseline"""
import numpy as np

def run(seed, obj_fn, dim, budget):
    """
    Запускає чистий Random Search.
    Args:
        seed: random seed
        obj_fn: callable, v ∈ [0,1]^dim → float (minimize)
        dim: dimensionality
        budget: max evaluations
    Returns:
        {'loss': float, 'curve': list, 'seed': int}
    """
    rng = np.random.default_rng(seed)
    bl = float('inf')
    curve = []
    x_best = None

    for _ in range(budget):
        v = rng.random(dim)
        fl = obj_fn(v)
        if fl < bl:
            bl = fl
            x_best = v.copy()
        curve.append(bl)

    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
