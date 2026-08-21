"""
Sigma-CMA — Surrogate Pre-screening CMA-ES with Rank-based Adaptive Filtering
====================================
Оригінальний алгоритм для пошуку гіперпараметрів нейронних мереж.

Ключові ідеї:
  1. Surrogate pre-screening: λ кандидатів оцінюються дешевою моделлю,
     до реального eval допускається лише top-⌈λ/ρ⌉
  2. Adaptive ρ: коефіцієнт відсіву коригується за точністю surrogate
  3. Warm buffer τ_warm: surrogate активується лише після накопичення
     достатньої кількості реальних точок
  4. Ranking-based surrogate: передбачаємо ранги, а не абсолютні значення
     (інваріантно до масштабу функції втрат)

Автор: оригінальна розробка, 2025
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple
from sklearn.ensemble import ExtraTreesRegressor
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Параметри алгоритму
# ---------------------------------------------------------------------------

@dataclass
class SigmaCMAParams:
    """Всі налаштування ΣCMA в одному місці."""

    # --- CMA-ES базові ---
    sigma0: float = 0.3          # початковий розмір кроку
    lam: Optional[int] = None    # розмір популяції (auto: 4+floor(3*ln(n)))
    mu: Optional[int] = None     # кількість батьків (auto: lam//2)

    # --- Surrogate ---
    tau_warm: int = 10           # мін. точок у буфері перед активацією surrogate
    rho_init: float = 4.0        # початковий коефіцієнт відсіву (лише 1/ρ оцінюється реально)
    rho_min: float = 1.5         # нижня межа ρ (при поганому surrogate ρ зменшується)
    rho_max: float = 8.0         # верхня межа ρ
    rho_adapt_rate: float = 0.2  # швидкість адаптації ρ
    surrogate_rank_thresh: float = 0.6   # мін. коефіцієнт Спірмена для довіри surrogate
    buffer_max: int = 200        # макс. розмір буфера (FIFO)

    # --- Зупинка ---
    max_evals: int = 500         # максимум реальних оцінок
    ftol: float = 1e-8           # допуск за значенням функції
    xtol: float = 1e-8           # допуск за кроком


# ---------------------------------------------------------------------------
# Surrogate модель (ранговий Extra Trees)
# ---------------------------------------------------------------------------

class RankSurrogate:
    """
    Surrogate модель на основі Extra Trees, що передбачає ранги.
    Передбачення рангів стійкіше до шуму і масштабу, ніж абсолютні значення.
    """

    def __init__(self, n_estimators: int = 50, min_samples: int = 5):
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self._model = None
        self._X: List[np.ndarray] = []
        self._y: List[float] = []

    # --- Буфер ---

    def add(self, x: np.ndarray, y: float) -> None:
        self._X.append(x.copy())
        self._y.append(float(y))

    def trim(self, max_size: int) -> None:
        if len(self._X) > max_size:
            self._X = self._X[-max_size:]
            self._y = self._y[-max_size:]

    @property
    def size(self) -> int:
        return len(self._X)

    # --- Навчання ---

    def fit(self) -> bool:
        if self.size < self.min_samples:
            return False
        X = np.array(self._X)
        y = np.array(self._y)
        ranks = np.argsort(np.argsort(y)).astype(float)  # перетворення → ранги
        self._model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=max(1, self.size // 20),
            random_state=42,
            n_jobs=1, # Обмежено для бенчмарку (RCU profiling)
        )
        self._model.fit(X, ranks)
        return True

    # --- Передбачення ---

    def predict_ranks(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(X))
        return self._model.predict(X)

    def rank_correlation(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """Коефіцієнт Спірмена між передбаченими та реальними рангами."""
        if self._model is None or len(X) < 3:
            return 0.0
        y_pred = self.predict_ranks(X)
        corr, _ = spearmanr(y_pred, y_true)
        return float(corr) if not np.isnan(corr) else 0.0


# ---------------------------------------------------------------------------
# Головний клас ΣCMA
# ---------------------------------------------------------------------------

class SigmaCMA:
    def __init__(
        self,
        dim: int,
        bounds: Optional[List[Tuple[float, float]]] = None,
        params: Optional[SigmaCMAParams] = None,
        seed: Optional[int] = None,
    ):
        self.n = dim
        self.bounds = bounds  # [(lb, ub), ...] або None
        self.p = params or SigmaCMAParams()
        self.rng = np.random.default_rng(seed)

        # --- Розміри популяції ---
        lam = self.p.lam or int(4 + np.floor(3 * np.log(dim)))
        self.lam = max(lam, 6)
        self.mu = self.p.mu or self.lam // 2

        # --- Ваги рекомбінації ---
        raw_w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_w / raw_w.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()

        # --- CMA параметри адаптації ---
        n = self.n
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff)
        )
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.chiN = n ** 0.5 * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # --- Стан CMA ---
        self.sigma = self.p.sigma0
        self.m = self.rng.uniform(0, 1, n)  # буде перевизначено при optimize()
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)

        # --- Surrogate ---
        self.surrogate = RankSurrogate()
        self.rho = self.p.rho_init

        # --- Лічильники ---
        self.eval_count = 0       # реальні eval
        self.gen = 0

    def _clip(self, x: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return x
        lb = np.array([b[0] for b in self.bounds])
        ub = np.array([b[1] for b in self.bounds])
        return np.clip(x, lb, ub)

    def _decompose(self) -> None:
        self.C = np.triu(self.C) + np.triu(self.C, 1).T  # симетрія
        try:
            eigvals, self.B = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-20)
            self.D = eigvals ** 0.5
        except Exception:
            self.B = np.eye(self.n)
            self.D = np.ones(self.n)

    def _sample_population(self) -> Tuple[np.ndarray, np.ndarray]:
        Z = self.rng.standard_normal((self.lam, self.n))
        D_mat = np.diag(self.D)
        X = self.m + self.sigma * (Z @ D_mat @ self.B.T)
        X = np.array([self._clip(x) for x in X])
        return X, Z

    def _update_evolution_paths(self, delta: np.ndarray) -> None:
        self.ps = (1 - self.cs) * self.ps + \
                  np.sqrt(self.cs * (2 - self.cs) * self.mueff) * \
                  (self.B @ (delta / self.D))

        hsig = (np.linalg.norm(self.ps) /
                np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) /
                self.chiN) < 1.4 + 2 / (self.n + 1)

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * delta

    def _update_covariance(self, D_weighted: np.ndarray, hsig: bool) -> None:
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = sum(
            self.weights[i] * np.outer(D_weighted[i], D_weighted[i])
            for i in range(self.mu)
        )
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (rank_one + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
            + self.cmu * rank_mu
        )

    def _update_sigma(self) -> None:
        self.sigma *= np.exp(
            (self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1)
        )

    def _adapt_rho(self, rank_corr: float) -> None:
        if rank_corr >= self.p.surrogate_rank_thresh:
            self.rho = min(self.p.rho_max, self.rho * (1 + self.p.rho_adapt_rate))
        else:
            self.rho = max(self.p.rho_min, self.rho * (1 - self.p.rho_adapt_rate))

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: Optional[np.ndarray] = None,
    ) -> "OptimizeResult":

        if x0 is not None:
            self.m = x0.copy().astype(float)
        elif self.bounds is not None:
            lb = np.array([b[0] for b in self.bounds])
            ub = np.array([b[1] for b in self.bounds])
            self.m = (lb + ub) / 2.0
        else:
            self.m = np.zeros(self.n)

        fbest = np.inf
        xbest = self.m.copy()
        history_f: List[float] = []
        history_evals: List[int] = []
        self.eval_count = 0
        self.gen = 0

        while self.eval_count < self.p.max_evals:
            self._decompose()
            X, Z = self._sample_population()
            n_candidates = len(X)

            use_surrogate = self.surrogate.size >= self.p.tau_warm
            real_indices = list(range(n_candidates))

            if use_surrogate:
                self.surrogate.fit()
                surrogate_ranks = self.surrogate.predict_ranks(X)
                n_real = max(self.mu, int(np.ceil(n_candidates / self.rho)))
                n_real = min(n_real, n_candidates)
                real_indices = np.argsort(surrogate_ranks)[:n_real].tolist()

            f_real = {}
            for idx in real_indices:
                if self.eval_count >= self.p.max_evals:
                    break
                fx = objective(X[idx])
                f_real[idx] = fx
                self.surrogate.add(X[idx], fx)
                self.eval_count += 1

                if fx < fbest:
                    fbest = fx
                    xbest = X[idx].copy()

            if not f_real:
                break

            if use_surrogate and len(f_real) >= 3:
                idxs = list(f_real.keys())
                y_true = np.array([f_real[i] for i in idxs])
                X_eval = X[idxs]
                rank_corr = self.surrogate.rank_correlation(X_eval, y_true)
                self._adapt_rho(rank_corr)

            f_all = {}
            for i in range(n_candidates):
                if i in f_real:
                    f_all[i] = f_real[i]
                elif use_surrogate:
                    f_all[i] = surrogate_ranks[i] + 1e6

            sorted_idx = sorted(f_all.keys(), key=lambda i: f_all[i])
            top_idx = sorted_idx[:self.mu]

            D_weighted = np.array([
                (X[i] - self.m) / self.sigma for i in top_idx
            ])
            delta = np.dot(self.weights, D_weighted)

            self.m = self.m + self.sigma * delta
            hsig = self._compute_hsig()
            self._update_evolution_paths(delta)
            self._update_covariance(D_weighted, hsig)
            self._update_sigma()
            self.surrogate.trim(self.p.buffer_max)

            history_f.append(fbest)
            history_evals.append(self.eval_count)
            self.gen += 1

            if self._check_stop(history_f):
                break

        return OptimizeResult(
            xbest=xbest,
            fbest=fbest,
            n_evals=self.eval_count,
            n_gens=self.gen,
            history_f=history_f,
            history_evals=history_evals,
            rho_final=self.rho,
        )

    def _compute_hsig(self) -> bool:
        norm_ps = np.linalg.norm(self.ps)
        denom_part = np.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1)))
        # Додав захист від ділення на 0, хоч і малоймовірно
        denom = max(denom_part * self.chiN, 1e-12)
        return (norm_ps / denom) < 1.4 + 2 / (self.n + 1)

    def _check_stop(self, history: List[float]) -> bool:
        if len(history) < 20:
            return False
        recent = history[-10:]
        if max(recent) - min(recent) < self.p.ftol:
            return True
        if self.sigma < self.p.xtol:
            return True
        return False


@dataclass
class OptimizeResult:
    xbest: np.ndarray
    fbest: float
    n_evals: int
    n_gens: int
    history_f: List[float]
    history_evals: List[int]
    rho_final: float

# ---------------------------------------------------------------------------
# Інтеграція у фреймворк HPO Benchmark
# ---------------------------------------------------------------------------
def run(seed, obj_fn, dim, budget):
    curve = []
    bl = float('inf')
    evals = 0
    
    # Створюємо обгортку для фіксації кожного кроку 
    # (оскільки ΣCMA оновлює history_f лише в кінці покоління)
    x_best = None
    def wrapped_obj(x):
        nonlocal bl, evals, x_best
        if evals >= budget:
            return bl
        fl = obj_fn(x)
        if fl < bl:
            bl = fl
            x_best = x.copy()
        curve.append(bl)
        evals += 1
        return fl

    # Ініціалізуємо параметри і викликаємо оптимізатор
    params = SigmaCMAParams(max_evals=budget)
    optimizer = SigmaCMA(dim=dim, bounds=[(0.0, 1.0)] * dim, params=params, seed=seed)
    
    # 1 попередній Sobol старт (уніфікована ініціалізація)
    from benchmark.init import sobol_init
    x0 = sobol_init(seed, dim, 1)[0]
    
    try:
        optimizer.optimize(wrapped_obj, x0=x0)
    except Exception as e:
        print(f"ΣCMA Error: {e}")
        
    # Дозаповнюємо curve, якщо алгоритм зупинився раніше (ftol)
    while len(curve) < budget:
        curve.append(bl)
        
    return {'loss': bl, 'curve': curve[:budget], 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
