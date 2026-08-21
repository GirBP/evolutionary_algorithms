"""SA-CMA v1 — Parsimony Pressure (+ penalty do HPO)"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def _cma_on_surrogate(surrogate, X_hist, y_hist, dim, rng, n_gens=50, lam=10, parsimony_beta=0.1):
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

    # Parsimony heuristic (using sum of vector as proxy for complexity)
    def complexity(x):
        return np.mean(x)

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
        
        base_fits = surrogate.predict(arx)
        # Parsimony pressure applied to surrogate predictions
        fits = np.array([bf + parsimony_beta * complexity(x) for bf, x in zip(base_fits, arx)])
        
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

    return np.clip(best_x, 0, 1)

def run(seed, obj_fn, dim, budget):
    rng = np.random.default_rng(seed)
    n_init = min(10, budget // 3)
    X_hist, y_hist, curve = [], [], []
    bl = float('inf')

    for _ in range(n_init):
        v = rng.random(dim)
        fl = obj_fn(v)
        X_hist.append(v)
        y_hist.append(fl)
        if fl < bl: bl = fl
        curve.append(bl)

    ec = n_init
    while ec < budget:
        rf = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=seed+ec, n_jobs=1)
        rf.fit(X_hist, y_hist)
        
        # CMA-ES Proxy with Parsimony
        beta_p = 0.05 * (ec / budget)
        candidate = _cma_on_surrogate(rf, X_hist, y_hist, dim, rng, n_gens=50, lam=10, parsimony_beta=beta_p)
        
        fl = obj_fn(candidate)
        X_hist.append(candidate)
        y_hist.append(fl)
        if fl < bl: bl = fl
        curve.append(bl)
        ec += 1

    return {'loss': bl, 'curve': curve, 'seed': seed}
