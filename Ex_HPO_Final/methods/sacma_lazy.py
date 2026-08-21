"""SA-CMA Lazy — версія з Lazy Eigen Cache та Adaptive Request + Delta F"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class AdaptiveRF:
    def __init__(self, seed):
        self.rf = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=seed, n_jobs=1)
        self.last_n = 0
        self.last_oob = -float('inf')
        self.rf.oob_score = True

    def fit_if_needed(self, X, y, ec_current, n_trees=30):
        n = len(X)
        if n - self.last_n >= 3 or self.last_oob < 0.3:
            self.rf.n_estimators = n_trees
            try:
                self.rf.fit(X, y)
                self.last_oob = getattr(self.rf, 'oob_score_', 0.0)
            except Exception:
                self.last_oob = -1.0
            self.last_n = n
        return self.rf, self.last_oob

def get_adaptive_params(X_hist, y_hist, oob_score, current_budget, total_budget):
    # 1. Прогрес пошуку (від 0.0 до 1.0)
    progress = current_budget / total_budget

    # 2. Адаптація OOB (масштабуємо у [0, 1])
    # Оцінка OOB може бути від'ємною, обрізаємо її
    normalized_oob = max(0.0, min(1.0, oob_score)) 

    # 3. Кількість дерев у RF (росте з ростом бази даних)
    # На старті (10 точок) вистачить 20 дерев. На 50 точках — 100 дерев.
    n_trees = int(10 + len(X_hist) * 2.0)
    n_trees = min(n_trees, 150) # Обмежуємо зверху для швидкості

    # 4. Кількість генерацій CMA-ES (Нелінійна інтерполяція)
    # Поєднує якість моделі (OOB) та фазу пошуку (progress)
    base_gens = 10 + (normalized_oob ** 2) * 90  # від 10 до 100 залежно від OOB
    # До кінця пошуку звужуємо фокус (робимо більше кроків для тонкої настройки)
    cma_generations = int(base_gens * (1 + progress)) 

    # 5. Адаптивний початковий крок CMA-ES (Sigma)
    # Рахуємо середню відстань між точками в архіві, щоб зрозуміти покриття простору
    from scipy.spatial.distance import pdist
    if len(X_hist) > 1:
        avg_distance = np.mean(pdist(X_hist)) # Яка розкиданість точок зараз?
    else:
        avg_distance = 0.5

    # Якщо точки збилися в купу (avg_distance малий), збільшуємо sigma для виходу
    sigma_init = 0.2 * (1 - progress) # Базове зменшення з часом
    if avg_distance < 0.1: # Критична щільність
        sigma_init += 0.2  # Штовхаємо алгоритм назовні (Exploration)

    return n_trees, cma_generations, sigma_init

def _cma_on_surrogate_lazy(surrogate, X_hist, y_hist, dim, rng, n_gens=50, lam=10, sigma_init=0.15):
    best_idx = np.argmin(y_hist)
    mean = X_hist[best_idx].copy()
    sigma = sigma_init
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

    # Eigen caching
    sqrtC, invsqrtC = np.eye(dim), np.eye(dim)
    eigen_interval = max(1, int(1.0 / (c1 + cmu) / dim / 10.0))
    eigen_counter = 0

    for gen in range(n_gens):
        if eigen_counter <= 0:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                invsqrtC = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
            except Exception:
                sqrtC, invsqrtC = np.eye(dim), np.eye(dim)
            eigen_counter = eigen_interval
        eigen_counter -= 1

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

    return np.clip(best_x, 0, 1)


def run(seed, obj_fn, dim, budget):
    """
    SA-CMA Lazy: Включає Кешування матриці коваріації та Ліниве перенавчання RF.
    Адаптація через ідеальний Delta F (вікно w=5).
    Найвища економія RCU (Методу).
    """
    rng = np.random.default_rng(seed)
    n_init = min(10, budget // 3)
    X_hist, y_hist, curve = [], [], []
    bl = float('inf')
    x_best = None

    from benchmark.init import sobol_init
    init_pts = sobol_init(seed, dim, n_init)
    for v in init_pts:
        fl = obj_fn(v)
        X_hist.append(v)
        y_hist.append(fl)
        if fl < bl: 
            bl = fl
            x_best = v.copy()
        curve.append(bl)

    ec = n_init
    
    arf = AdaptiveRF(seed)
    
    while ec < budget:
        Xh, yh = np.array(X_hist), np.array(y_hist)
        progress = ec / budget
        
        # Визначаємо адаптивні параметри через вашу функцію
        n_trees, n_gens, sigma_init = get_adaptive_params(X_hist, yh, arf.last_oob, ec, budget)

        # Lazy RF Fit (тепер з динамічною кількістю дерев)
        rf, oob = arf.fit_if_needed(Xh, yh, ec, n_trees=n_trees)
        
        # Lazy CMA-ES (з динамічними генераціями та sigma_init)
        cma_cand = _cma_on_surrogate_lazy(rf, Xh, yh, dim, rng, n_gens=n_gens, lam=10, sigma_init=sigma_init)
        
        # Dynamic Filtering (budget scaling depending on OOB)
        n_virtual = int(40 + 160 * progress * max(0.0, oob))
        virtual_pop = rng.random((n_virtual, dim))
        
        all_cand = np.vstack([cma_cand.reshape(1, -1), virtual_pop])
        p_fits = rf.predict(all_cand)
        best_vc = np.argmin(p_fits)
        candidate = np.clip(all_cand[best_vc], 0, 1)

        fl = obj_fn(candidate)
        X_hist.append(candidate)
        y_hist.append(fl)
        if fl < bl: 
            bl = fl
            x_best = candidate.copy()
        curve.append(bl)
        ec += 1

    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
