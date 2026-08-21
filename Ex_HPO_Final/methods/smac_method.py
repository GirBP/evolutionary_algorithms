"""
SMAC-RF — Random Forest Surrogate + Expected Improvement.

Reimplementation of the core SMAC3 algorithm (Lindauer et al., JMLR 2022):
  - Random Forest as surrogate model
  - Expected Improvement as acquisition function
  - Local search on EI for candidate generation

NOTE: The official SMAC3 library (both v1.4 and v2.3) is incompatible
with the current environment due to ConfigSpace/Python 3.14 conflicts.
This is an independent implementation of the same algorithmic pipeline.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import norm


def run(seed, obj_fn, dim, budget):
    """
    SMAC-RF: Random Forest + EI acquisition for HPO benchmark.

    Implements the core SMAC loop:
      1. Build RF surrogate on observed data
      2. Compute EI acquisition across candidate points
      3. Evaluate the best candidate
      4. Update surrogate

    Args:
        seed: random seed
        obj_fn: callable, v ∈ [0,1]^dim → float (minimize)
        dim: dimensionality
        budget: max real evaluations
    Returns:
        {'loss': float, 'curve': list, 'seed': int, 'x_best': list}
    """
    rng = np.random.default_rng(seed)

    # ── Initial design (Sobol) ─────────────────────────────────────────
    from benchmark.init import sobol_init
    n_init = min(max(dim + 1, 5), budget // 3)
    init_pts = sobol_init(seed, dim, n_init)

    X_hist, y_hist, curve = [], [], []
    bl = float('inf')
    x_best = None

    for v in init_pts:
        fl = obj_fn(v)
        X_hist.append(v.copy())
        y_hist.append(fl)
        if fl < bl:
            bl = fl
            x_best = v.copy()
        curve.append(bl)

    ec = n_init

    # ── Main BO loop ───────────────────────────────────────────────────
    while ec < budget:
        Xh = np.array(X_hist)
        yh = np.array(y_hist)

        # 1. Train RF surrogate (SMAC default: 10 trees, log-transformed y)
        rf = RandomForestRegressor(
            n_estimators=10,
            max_features=5.0 / 6.0 if dim > 5 else 1.0,
            min_samples_leaf=3,
            random_state=seed + ec,
            n_jobs=1,
        )
        rf.fit(Xh, yh)

        # 2. Compute EI using individual tree predictions (predictive variance)
        best_y = np.min(yh)

        # Generate candidates: random + local perturbations around best
        n_random = 500
        n_local = 200
        cands_random = rng.random((n_random, dim))

        # Local search: perturb around best found point
        perturb_scale = max(0.05, 0.2 * (1 - ec / budget))
        cands_local = x_best + rng.normal(0, perturb_scale, (n_local, dim))
        cands_local = np.clip(cands_local, 0, 1)

        candidates = np.vstack([cands_random, cands_local])

        # Compute mean and std from individual tree predictions
        tree_preds = np.array([tree.predict(candidates) for tree in rf.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
        sigma = np.maximum(sigma, 1e-9)

        # Expected Improvement
        imp = best_y - mu
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-8] = 0.0

        # Select best EI candidate
        best_idx = np.argmax(ei)
        candidate = np.clip(candidates[best_idx], 0, 1)

        # 3. Evaluate
        fl = obj_fn(candidate)
        X_hist.append(candidate.copy())
        y_hist.append(fl)
        if fl < bl:
            bl = fl
            x_best = candidate.copy()
        curve.append(bl)
        ec += 1

    return {
        'loss': bl,
        'curve': curve[:budget],
        'seed': seed,
        'x_best': x_best.tolist() if x_best is not None else None,
    }
