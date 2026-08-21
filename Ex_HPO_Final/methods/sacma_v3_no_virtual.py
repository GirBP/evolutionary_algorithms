"""SA-CMA-ES v3 — Surrogate-Assisted CMA-ES with Fitness Gradient Adaptation"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _cma_on_surrogate(surrogate, X_hist, y_hist, dim, rng,
                      n_gens=50, lam=10):
    """Чисте еволюційне ядро CMA-ES на поверхні сурогату."""
    best_idx = np.argmin(y_hist)
    mean = X_hist[best_idx].copy()
    sigma = 0.15
    mu = max(2, lam // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)
    cc = 4.0 / (dim + 4)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2.0 / ((dim + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    pc, ps, C = np.zeros(dim), np.zeros(dim), np.eye(dim)
    chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))
    best_x, best_f = mean.copy(), float('inf')

    for gen in range(n_gens):
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            invsqrtC = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
        except Exception:
            sqrtC, invsqrtC = np.eye(dim), np.eye(dim)

        arz = rng.standard_normal((lam, dim))
        arx = np.array([np.clip(mean + sigma * sqrtC @ z, 0, 1) for z in arz])

        fits = surrogate.predict(arx)

        idx = np.argsort(fits)
        r_best = np.argmin(fits)
        if fits[r_best] < best_f:
            best_f, best_x = fits[r_best], arx[r_best].copy()

        old_mean = mean.copy()
        mean = sum(weights[i] * arx[idx[i]] for i in range(mu))
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (mean - old_mean) / sigma
        hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1))) < (1.4+2/(dim+1))*chiN
        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
        artmp = (arx[idx[:mu]] - old_mean) / sigma
        C = (1-c1-cmu)*C + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
        for i in range(mu):
            C += cmu * weights[i] * np.outer(artmp[i], artmp[i])
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = np.clip(sigma, 1e-8, 0.5)
        if sigma < 1e-6:
            break

    return np.clip(best_x, 0, 1)


def run(seed, obj_fn, dim, budget):
    """
    SA-CMA-ES v3 з адаптацією через Градієнт Покращення Фітнесу (Delta F).
    Args:
        seed: random seed
        obj_fn: callable, v ∈ [0,1]^dim → float (minimize)
        dim: dimensionality
        budget: max real evaluations
    Returns:
        {'loss': float, 'curve': list, 'seed': int}
    """
    rng = np.random.default_rng(seed)
    n_init = min(10, budget // 3)

    X_hist, y_hist, curve = [], [], []
    bl = float('inf')
    x_best = None

    # ── 1. Sobol init (уніфікована ініціалізація) ────────────────────────
    from benchmark.init import sobol_init
    init_pts = sobol_init(seed, dim, n_init)
    for v in init_pts:
        fl = obj_fn(v)
        X_hist.append(v.copy())
        y_hist.append(fl)
        if fl < bl:
            bl = fl
            x_best = v.copy()
        curve.append(bl)

    ec = n_init

    # ── Delta F state ────────────────────────────────────────────────────
    w = 5
    lam, n_gens = 10, 50
    LAM_MIN, LAM_MAX = 4, 30
    GEN_MIN, GEN_MAX = 5, 120
    best_history = [bl] * w

    # ── 2. Surrogate-assisted loop ───────────────────────────────────────
    while ec < budget:
        Xh, yh = np.array(X_hist), np.array(y_hist)
        progress = ec / budget

        # Delta F: адаптація популяції та генерацій
        f_old = best_history[0]
        delta_f = max(0.0, (f_old - bl) / (f_old + 1e-12))
        if delta_f < 0.001:   # Стагнація
            lam = min(LAM_MAX, int(lam * 1.5))
            n_gens = max(GEN_MIN, int(n_gens * 0.8))
        else:                 # Прогрес
            lam = max(LAM_MIN, int(lam * 0.8))
            n_gens = min(GEN_MAX, int(n_gens * 1.2))

        # Surrogate (Random Forest)
        rf = RandomForestRegressor(
            n_estimators=30, max_depth=8,
            random_state=seed + ec, n_jobs=1
        )
        rf.fit(Xh, yh)

        # CMA-ES на сурогаті
        cma_best = _cma_on_surrogate(rf, Xh, yh, dim, rng, n_gens=n_gens, lam=lam)

        # No Surrogate Filtering (Pure CMA candidate)
        candidate = np.clip(cma_best, 0, 1)

        # Реальна оцінка
        fl = obj_fn(candidate)
        X_hist.append(candidate.copy())
        y_hist.append(fl)
        if fl < bl:
            bl = fl
            x_best = candidate.copy()
        curve.append(bl)
        ec += 1

        # Зсув вікна
        best_history.pop(0)
        best_history.append(bl)

    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
