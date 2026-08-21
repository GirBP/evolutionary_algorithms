"""SA-CMA MAB — Метод від користувача з PopulationMAB та _AdaptiveRF"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def compute_complexity(v):
    # Адаптація для бенчмарку (щоб не падало на розмірностях < 8)
    try:
        return 0.4*v[1] + 0.3*v[2] + 0.2*v[3] + 0.1*v[7]
    except IndexError:
        return np.mean(v)

class PopulationMAB:
    ACTIONS = [-1, 0, +1]

    def __init__(self, epsilon=0.15):
        self.epsilon = epsilon
        self.q_values = np.zeros(3)
        self.counts = np.ones(3)
        self.last_action = 1

    def choose(self, rng):
        if rng.random() < self.epsilon:
            return rng.integers(3)
        return int(np.argmax(self.q_values))

    def update(self, action_idx, reward):
        self.counts[action_idx] += 1
        n = self.counts[action_idx]
        self.q_values[action_idx] += (reward - self.q_values[action_idx]) / n
        self.last_action = action_idx


def _cma_on_surrogate(surrogate, X_hist, y_hist, dim, rng,
                      n_gens=50, lam=10, parsimony_beta=0.0):
    """Еволюційне ядро CMA-ES — незмінне, крім lazy eigen-cache."""
    best_idx = np.argmin(y_hist)
    mean = X_hist[best_idx].copy()
    sigma = 0.15
    mu = max(2, lam // 2)
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= weights.sum()
    mueff = 1.0 / np.sum(weights**2)

    cc    = 4.0 / (dim + 4)
    cs    = (mueff + 2) / (dim + mueff + 5)
    c1    = 2.0 / ((dim + 1.3)**2 + mueff)
    cmu_val = min(1 - c1, 2*(mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
    damps = 1 + 2*max(0, np.sqrt((mueff - 1)/(dim + 1)) - 1) + cs

    pc = np.zeros(dim)
    ps = np.zeros(dim)
    C  = np.eye(dim)
    chiN = dim**0.5 * (1 - 1/(4*dim) + 1/(21*dim**2))

    best_x, best_f = mean.copy(), float('inf')

    # ── Lazy eigendecomposition cache (авторегуляція) ──────────────────
    _sqrtC    = np.eye(dim)
    _invsqrtC = np.eye(dim)
    _eigen_at = -1          # генерація останнього розкладання

    for gen in range(n_gens):
        # Адаптивний інтервал: частіше при великому σ, рідше при малому
        eigen_interval = max(1, int(dim / (10 * lam)))
        if gen - _eigen_at >= eigen_interval:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                _sqrtC    = eigvecs @ np.diag(np.sqrt(eigvals))    @ eigvecs.T
                _invsqrtC = eigvecs @ np.diag(1.0/np.sqrt(eigvals)) @ eigvecs.T
            except Exception:
                _sqrtC    = np.eye(dim)
                _invsqrtC = np.eye(dim)
            _eigen_at = gen

        sqrtC, invsqrtC = _sqrtC, _invsqrtC
        # ────────────────────────────────────────────────────────────────

        arz = rng.standard_normal((lam, dim))
        arx = np.array([np.clip(mean + sigma * sqrtC @ z, 0, 1) for z in arz])

        try:
            raw_fits = surrogate.predict(arx)
            if parsimony_beta > 0:
                complexities = np.array([compute_complexity(x) for x in arx])
                fits = raw_fits + parsimony_beta * complexities
            else:
                fits = raw_fits
        except Exception:
            fits = np.ones(lam) * 1e6

        idx = np.argsort(fits)
        raw_best_idx = np.argmin(fits)
        if fits[raw_best_idx] < best_f:
            best_f = fits[raw_best_idx]
            best_x = arx[raw_best_idx].copy()

        old_mean = mean.copy()
        mean = sum(weights[i] * arx[idx[i]] for i in range(mu))

        ps = ((1 - cs)*ps
              + np.sqrt(cs*(2-cs)*mueff) * invsqrtC @ (mean - old_mean) / sigma)
        hsig = (np.linalg.norm(ps)
                / np.sqrt(1 - (1-cs)**(2*(gen+1)))
                < (1.4 + 2/(dim+1)) * chiN)
        pc = ((1 - cc)*pc
              + hsig * np.sqrt(cc*(2-cc)*mueff) * (mean - old_mean) / sigma)

        artmp = (arx[idx[:mu]] - old_mean) / sigma
        C = ((1 - c1 - cmu_val)*C
             + c1*(np.outer(pc, pc) + (1-hsig)*cc*(2-cc)*C))
        for i in range(mu):
            C += cmu_val * weights[i] * np.outer(artmp[i], artmp[i])

        sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / chiN - 1))
        sigma = np.clip(sigma, 1e-8, 0.5)
        if sigma < 1e-6:
            break

    return np.clip(best_x, 0, 1)


class _AdaptiveRF:
    """
    Warm-start Random Forest:
    - зберігає попередній стан;
    - перенавчається лише якщо набір даних виріс ≥ refit_every зразків
      АБО oob_score впав нижче oob_decay_thr.
    """
    def __init__(self, seed):
        self.seed = seed
        self._rf     = None
        self._n_last = 0
        self.oob_    = 0.0
        self._refit_every = 3      # мінімальний крок між повними рефітами

    def fit_if_needed(self, Xh, yh, ec):
        n = len(Xh)
        needs_refit = (
            self._rf is None
            or (n - self._n_last) >= self._refit_every
            or self.oob_ < 0.15
        )
        if not needs_refit:
            return self._rf, self.oob_

        rf = RandomForestRegressor(
            n_estimators=30, max_depth=8,
            oob_score=True, random_state=self.seed + ec, n_jobs=1
        )
        try:
            rf.fit(Xh, yh)
            self.oob_    = max(0.0, getattr(rf, 'oob_score_', 0.0))
        except Exception:
            self.oob_ = 0.0
            
        self._rf     = rf
        self._n_last = n
        return self._rf, self.oob_


def run(seed, obj_fn, dim, budget):
    """
    SA-CMA MAB — версія, надана користувачем.
    Інтегровано під стандартний інтерфейс benchmark фреймворку.
    """
    rng = np.random.default_rng(seed)

    n_init = min(10, budget // 3)

    X_hist, y_hist = [], []
    curve = []
    bl    = float('inf')
    x_best = None

    # ── 1. Sobol init ────────────────────────────────────────────────────
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

    ec  = n_init
    mab = PopulationMAB(epsilon=0.15)
    arf = _AdaptiveRF(seed)

    lam    = 10
    n_gens = 50
    LAM_MIN, LAM_MAX = 4, 30
    GEN_MIN, GEN_MAX = 5, 120
    PARSIMONY_BASE   = 0.05

    # ── 2. Еволюційний цикл із сурогатом ─────────────────────────────────
    while ec < budget:
        Xh, yh   = np.array(X_hist), np.array(y_hist)
        progress = ec / budget

        # ── Adaptive RF: повне перенавчання лише при потребі ─────────────
        rf, oob = arf.fit_if_needed(Xh, yh, ec)

        if oob < 0.1:
            candidate = rng.random(dim)
        else:
            parsimony_beta = PARSIMONY_BASE * progress

            cma_best = _cma_on_surrogate(
                rf, Xh, yh, dim, rng,
                n_gens=n_gens, lam=lam,
                parsimony_beta=parsimony_beta
            )

            # ── Budget-scaled virtual sampling ───────────────────────────
            n_virtual = int(np.clip(40 + 160 * progress * oob, 40, 200))

            virtual_pop    = rng.random((n_virtual, dim))
            all_candidates = np.vstack([cma_best.reshape(1, -1), virtual_pop])

            try:
                predicted_fits = rf.predict(all_candidates)
                if parsimony_beta > 0:
                    complexities  = np.array([compute_complexity(x) for x in all_candidates])
                    adjusted_fits = predicted_fits + parsimony_beta * complexities
                else:
                    adjusted_fits = predicted_fits

                best_virtual_idx = np.argmin(adjusted_fits)
                candidate        = np.clip(all_candidates[best_virtual_idx], 0, 1)
            except Exception:
                candidate = cma_best

        # Реальна оцінка
        fl = obj_fn(candidate)
        X_hist.append(candidate.copy())
        y_hist.append(fl)
        
        # ── 3. MAB feedback ──────────────────────────────────────────────
        improved   = fl < bl
        cost_ratio = (lam * n_gens) / (LAM_MAX * GEN_MAX)

        if improved:
            reward = (bl - fl) / (bl + 1e-12) / max(cost_ratio, 0.01)
        else:
            reward = -cost_ratio

        mab.update(mab.last_action, reward)
        action_idx = mab.choose(rng)
        action     = PopulationMAB.ACTIONS[action_idx]
        mab.last_action = action_idx

        step_lam = max(1, int((LAM_MAX - LAM_MIN) * 0.2))
        step_gen = max(1, int((GEN_MAX - GEN_MIN) * 0.15))
        lam    = int(np.clip(lam    + action * step_lam, LAM_MIN, LAM_MAX))
        n_gens = int(np.clip(n_gens + action * step_gen, GEN_MIN, GEN_MAX))

        if fl < bl:
            bl = fl
            x_best = candidate.copy()
        curve.append(bl)
        ec += 1

    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
