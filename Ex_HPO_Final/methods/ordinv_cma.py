"""OrdInv-CMA — CMA-ES з ординальним сурогатним скринінгом.

Портовано з Ex19_HPO/ex19_code/methods/ordinv_cma.py для нового
бенчмарк-інтерфейсу run(seed, obj_fn, dim, budget).

Алгоритм:
  1. Sobol ініціалізація
  2. CMA-ES генерує популяцію
  3. Ординальний сурогат (Bradley-Terry через LogReg на попарних різницях)
     оцінює τ_b Кендала на hold-out
  4. Якщо τ_b > 0.6 і n_obs > 2*popsize — скринінг: оцінюємо лише top-k
  5. Неоцінені кандидати отримують worst fitness + penalty
  6. Адаптивне k_eval = max(2μ, popsize * (1 - τ_b/2))

Автор: портовано з OrdInvCMA class (Ex19)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


class _OrdinalSurrogate:
    """Bradley-Terry pairwise ordinal surrogate (lightweight)."""

    def __init__(self, seed=42):
        self.model = None
        self.kendall_tau = 0.0
        self.seed = seed
        self._obs_X = []
        self._obs_y = []

    def observe(self, x, fitness):
        self._obs_X.append(np.array(x, dtype=float))
        self._obs_y.append(float(fitness))

    @property
    def n_obs(self):
        return len(self._obs_X)

    def fit(self):
        n = self.n_obs
        if n < 4:
            self.model = None
            self.kendall_tau = 0.0
            return

        X_arr = np.array(self._obs_X)
        y_arr = np.array(self._obs_y)

        pair_X, pair_labels = [], []
        for i in range(n):
            for j in range(i + 1, n):
                diff = X_arr[i] - X_arr[j]
                label = 1 if y_arr[i] < y_arr[j] else 0
                pair_X.append(diff)
                pair_labels.append(label)
                pair_X.append(-diff)
                pair_labels.append(1 - label)

        pair_X = np.array(pair_X)
        pair_labels = np.array(pair_labels)

        if len(np.unique(pair_labels)) < 2:
            self.model = None
            self.kendall_tau = 0.0
            return

        self.model = LogisticRegression(
            max_iter=300, C=1.0, solver="lbfgs", random_state=self.seed
        )
        try:
            self.model.fit(pair_X, pair_labels)
        except Exception:
            self.model = None
            self.kendall_tau = 0.0
            return

        # Holdout τ_b: use last 30% of observations
        h_start = max(0, int(n * 0.7))
        if h_start < n - 1:
            hX = X_arr[h_start:]
            hy = y_arr[h_start:]
            concordant, discordant = 0, 0
            for i in range(len(hX)):
                for j in range(i + 1, len(hX)):
                    diff = (hX[i] - hX[j]).reshape(1, -1)
                    try:
                        pred_prob = self.model.predict_proba(diff)[:, 1][0]
                    except Exception:
                        pred_prob = 0.5
                    pred = 1 if pred_prob > 0.5 else 0
                    true = 1 if hy[i] < hy[j] else 0
                    if pred == true:
                        concordant += 1
                    else:
                        discordant += 1
            total = concordant + discordant
            self.kendall_tau = (concordant - discordant) / max(total, 1)
        else:
            self.kendall_tau = 0.0

    def predict_scores(self, candidates):
        """Copeland aggregation: average P(candidate > obs) across all obs."""
        if self.model is None or self.n_obs == 0:
            return np.full(len(candidates), 0.5)

        obs_X = np.array(self._obs_X)
        candidates = np.atleast_2d(candidates)
        scores = np.zeros(len(candidates))

        for obs_point in obs_X:
            diff = candidates - obs_point
            try:
                scores += self.model.predict_proba(diff)[:, 1]
            except Exception:
                scores += 0.5

        return scores / self.n_obs


def run(seed, obj_fn, dim, budget):
    """OrdInv-CMA: CMA-ES + ordinal screening.
    
    Interface: run(seed, obj_fn, dim, budget) -> dict
    Compatible with benchmark/run_method.py
    """
    import cma

    rng = np.random.default_rng(seed)
    surrogate = _OrdinalSurrogate(seed=seed)

    # --- Init phase (Sobol) ---
    from benchmark.init import sobol_init
    n_init = min(max(4, dim + 2), budget // 3)
    init_pts = sobol_init(seed, dim, n_init)

    X_hist, y_hist, curve = [], [], []
    bl = float('inf')
    x_best = None

    for v in init_pts:
        fl = obj_fn(v)
        X_hist.append(v.copy())
        y_hist.append(fl)
        surrogate.observe(v, fl)
        if fl < bl:
            bl = fl
            x_best = v.copy()
        curve.append(bl)

    surrogate.fit()
    eval_count = n_init

    # --- CMA-ES setup ---
    x0 = [0.5] * dim
    sigma0 = 0.3
    pop_size = 4 + int(3 * np.log(dim))
    mu_sel = pop_size // 2

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"popsize": pop_size, "seed": seed, "verbose": -9,
         "bounds": [[0.0] * dim, [1.0] * dim]}
    )

    k_eval = pop_size  # initially evaluate all

    # --- Main loop ---
    while eval_count < budget and not es.stop():
        offspring = list(es.ask())

        # Pre-screen with ordinal surrogate if confident
        if (surrogate.model is not None
                and surrogate.kendall_tau > 0.6
                and surrogate.n_obs > 2 * pop_size):
            scores = surrogate.predict_scores(np.array(offspring))
            eval_idx = np.argsort(-scores)[:k_eval]  # top k_eval by score
            to_eval = [offspring[i] for i in eval_idx]
        else:
            to_eval = list(offspring)

        # Evaluate selected candidates
        evaluated = []
        for x in to_eval:
            if eval_count >= budget:
                break
            x_arr = np.clip(np.array(x), 0.0, 1.0)
            fl = obj_fn(x_arr)
            eval_count += 1
            surrogate.observe(x_arr, fl)
            evaluated.append((x, fl))
            if fl < bl:
                bl = fl
                x_best = x_arr.copy()
            curve.append(bl)

        if not evaluated:
            break

        # CMA update: non-evaluated offspring get worst fitness + penalty
        max_f = max(f for _, f in evaluated) + 1.0
        full_fits = []
        for off in offspring:
            matched = False
            for ex, ef in evaluated:
                if np.allclose(off, ex):
                    full_fits.append(ef)
                    matched = True
                    break
            if not matched:
                full_fits.append(max_f)

        es.tell(offspring, full_fits)

        # Update surrogate and adaptive k_eval
        surrogate.fit()
        k_eval = max(2 * mu_sel, int(pop_size * (1 - surrogate.kendall_tau / 2)))

    return {
        'loss': bl,
        'curve': curve,
        'seed': seed,
        'x_best': x_best.tolist() if x_best is not None else None,
    }
