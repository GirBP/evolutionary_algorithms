"""SA-CMA Golden — Метод із контролем кількості дерев через золоте переріз (0.382, 0.618)"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def _cma_on_surrogate_lazy(surrogate, X_hist, y_hist, dim, rng, n_gens=50, lam=10):
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
    _sqrtC, _invsqrtC = np.eye(dim), np.eye(dim)
    _eigen_at = -1

    for gen in range(n_gens):
        eigen_interval = max(1, int(dim / (10 * lam)))
        if gen - _eigen_at >= eigen_interval:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                _sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                _invsqrtC = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
            except Exception:
                _sqrtC, _invsqrtC = np.eye(dim), np.eye(dim)
            _eigen_at = gen

        arz = rng.standard_normal((lam, dim))
        arx = np.array([np.clip(mean + sigma * _sqrtC @ z, 0, 1) for z in arz])
        
        try:
            fits = surrogate.predict(arx)
        except Exception:
            fits = np.ones(lam) * 1e6

        idx = np.argsort(fits)
        r_best = np.argmin(fits)
        if fits[r_best] < best_f:
            best_f, best_x = fits[r_best], arx[r_best].copy()

        old_mean = mean.copy()
        mean = sum(weights[i] * arx[idx[i]] for i in range(mu))
        ps = (1-cs)*ps + np.sqrt(cs*(2-cs)*mueff) * _invsqrtC @ (mean - old_mean) / sigma
        hsig = np.linalg.norm(ps)/np.sqrt(1-(1-cs)**(2*(gen+1))) < (1.4+2/(dim+1))*chiN
        pc = (1-cc)*pc + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma
        artmp = (arx[idx[:mu]] - old_mean) / sigma
        C = (1-c1-cmu)*C + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C)
        for i in range(mu):
            C += cmu * weights[i] * np.outer(artmp[i], artmp[i])
        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = np.clip(sigma, 1e-8, 0.5)

    return np.clip(best_x, 0, 1)

# ═══════════ SA-CMA-ES (φ-Adaptive) ═══════════
# Коефіцієнти золотого перерізу
GS_CONTRACT = 0.382   # при покращенні — стискаємо
GS_EXPAND   = 0.618   # при стагнації — розширюємо

LAM_MIN, LAM_MAX = 4, 30
GEN_MIN, GEN_MAX = 5, 120

def run(seed, obj_fn, dim, budget):
    """
    Точна адаптація φ-Adaptive SA-CMA-ES з експерименту Ex64_Corporative.
    """
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
    
    # φ-адаптивні параметри (стартуємо з середини діапазону)
    alpha_lam = 0.5    # нормалізована позиція λ 
    alpha_gen = 0.5    # нормалізована позиція n_gens
    
    # Використовуємо наш найшвидший AdaptiveRF із попередніх версій,
    # щоб RCU не злітало в космос, але логіка еволюції буде оригінальна 
    from sklearn.ensemble import RandomForestRegressor
    import warnings
    class _AdaptiveRF:
        def __init__(self, seed):
            self.seed = seed
            self._rf = None
            self._n_last = 0
            self.oob_ = 0.0

        def fit_if_needed(self, Xh, yh, ec):
            n = len(Xh)
            needs_refit = (self._rf is None or (n - self._n_last) >= 3 or self.oob_ < 0.15)
            if not needs_refit: return self._rf, self.oob_
            rf = RandomForestRegressor(n_estimators=30, max_depth=8, oob_score=True, random_state=self.seed + ec, n_jobs=1)
            try:
                rf.fit(Xh, yh)
                self.oob_ = max(0.0, getattr(rf, 'oob_score_', 0.0))
            except Exception: self.oob_ = 0.0
            self._rf = rf
            self._n_last = n
            return self._rf, self.oob_
            
    arf = _AdaptiveRF(seed)

    while ec < budget:
        Xh, yh = np.array(X_hist), np.array(y_hist)
        
        # Поточні значення з нормалізованих позицій
        lam = int(LAM_MIN + alpha_lam * (LAM_MAX - LAM_MIN))
        n_gens = int(GEN_MIN + alpha_gen * (GEN_MAX - GEN_MIN))

        rf, oob = arf.fit_if_needed(Xh, yh, ec)
        
        if oob < 0.1:
            candidate = rng.random(dim)
        else:
            candidate = _cma_on_surrogate_lazy(rf, Xh, yh, dim, rng, n_gens=n_gens, lam=lam)

        fl = obj_fn(candidate)
        X_hist.append(candidate)
        y_hist.append(fl)
        
        # φ-адаптація золотим перерізом
        if fl < bl:
            # Покращення → стискаємо (менше популяції, менше генерацій)
            bl = fl
            alpha_lam = alpha_lam * GS_CONTRACT
            alpha_gen = alpha_gen * GS_CONTRACT
        else:
            # Стагнація → розширюємо (більше популяції, більше генерацій)
            alpha_lam = alpha_lam + (1 - alpha_lam) * GS_EXPAND
            alpha_gen = alpha_gen + (1 - alpha_gen) * GS_EXPAND
            
        curve.append(bl)
        ec += 1

    return {'loss': bl, 'curve': curve, 'seed': seed}
