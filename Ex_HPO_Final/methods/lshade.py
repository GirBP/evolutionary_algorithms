"""L-SHADE — SHADE with Linear Population Size Reduction (Tanabe & Fukunaga, CEC 2014)

Розширення SHADE з єдиною, але потужною модифікацією:
  - Розмір популяції NP лінійно зменшується від NP_init до NP_min
    протягом усього бюджету обчислень.
  - Це змушує алгоритм переходити від exploration до exploitation.

Стаття: R. Tanabe and A. Fukunaga, "Improving the Search Performance of
SHADE Using Linear Population Size Reduction," IEEE CEC 2014.

Залежності: numpy (вже встановлений)
"""
import numpy as np
from benchmark.init import sobol_init


def run(seed, obj_fn, dim, budget):
    """
    L-SHADE: SHADE + Linear Population Size Reduction.
    Args:
        seed: random seed
        obj_fn: callable, v ∈ [0,1]^dim → float (minimize)
        dim: dimensionality of search space
        budget: max number of real evaluations
    Returns:
        {'loss': float, 'curve': list, 'seed': int, 'x_best': list}
    """
    rng = np.random.default_rng(seed)

    # ── Параметри (за оригінальною статтею) ────────────────────────────
    NP_init = max(7, 2 * dim)
    NP_init = min(NP_init, budget // 2)
    NP_min = 4                          # мінімальний розмір популяції
    NP = NP_init
    H = NP_init                         # розмір історії (фіксований)
    p_best_rate = 0.1

    # ── Ініціалізація популяції (Sobol) ────────────────────────────────
    init_pts = sobol_init(seed, dim, NP)
    pop = [init_pts[i].copy() for i in range(NP)]
    fits = [obj_fn(np.clip(x, 0, 1)) for x in pop]

    bl = float('inf')
    curve = []
    ec = 0
    x_best = None

    for i in range(NP):
        ec += 1
        if fits[i] < bl:
            bl = fits[i]
            x_best = pop[i].copy()
        curve.append(bl)

    # ── Success-History пам'ять ─────────────────────────────────────────
    M_F = np.full(H, 0.5)
    M_CR = np.full(H, 0.5)
    k = 0

    # ── Архів ──────────────────────────────────────────────────────────
    archive = []
    archive_max = NP_init

    # ── Головний цикл ──────────────────────────────────────────────────
    while ec < budget:
        S_F = []
        S_CR = []
        S_delta = []

        for i in range(NP):
            if ec >= budget:
                break

            # ── Генерація F та CR із success-history ───────────────────
            r = rng.integers(H)

            # F ~ Cauchy(M_F[r], 0.1), обрізаний до (0, 1]
            while True:
                Fi = float(M_F[r] + 0.1 * rng.standard_cauchy())
                if Fi > 0:
                    break
            Fi = min(Fi, 1.0)

            # CR ~ N(M_CR[r], 0.1), обрізаний до [0, 1]
            CRi = float(rng.normal(M_CR[r], 0.1))
            CRi = np.clip(CRi, 0.0, 1.0)

            # ── Мутація: current-to-pbest/1 з архівом ──────────────────
            n_pbest = max(1, int(np.ceil(p_best_rate * NP)))
            sorted_idx = np.argsort(fits[:NP])
            pbest_pool = sorted_idx[:n_pbest]
            x_pbest = pop[rng.choice(pbest_pool)]

            candidates_r1 = [j for j in range(NP) if j != i]
            r1 = rng.choice(candidates_r1)

            union = list(range(NP)) + list(range(NP, NP + len(archive)))
            candidates_r2 = [j for j in union if j != i and j != r1]
            if not candidates_r2:
                candidates_r2 = [j for j in range(NP) if j != i]
            r2_idx = rng.choice(candidates_r2)

            if r2_idx < NP:
                x_r2 = pop[r2_idx]
            else:
                x_r2 = archive[r2_idx - NP]

            v = pop[i] + Fi * (x_pbest - pop[i]) + Fi * (pop[r1] - x_r2)
            v = np.clip(v, 0, 1)

            # ── Кросовер (binomial) ────────────────────────────────────
            u = pop[i].copy()
            j_rand = rng.integers(dim)
            for d in range(dim):
                if rng.random() < CRi or d == j_rand:
                    u[d] = v[d]

            # ── Селекція ───────────────────────────────────────────────
            fl = obj_fn(u)
            ec += 1

            if fl <= fits[i]:
                archive.append(pop[i].copy())
                if len(archive) > archive_max:
                    archive.pop(rng.integers(len(archive)))

                delta = fits[i] - fl
                S_F.append(Fi)
                S_CR.append(CRi)
                S_delta.append(delta)

                pop[i] = u
                fits[i] = fl

            if fl < bl:
                bl = fl
                x_best = u.copy()
            curve.append(bl)

        # ── Оновлення Success-History ──────────────────────────────────
        if S_F and S_CR:
            deltas = np.array(S_delta)
            weights = deltas / (deltas.sum() + 1e-30)

            f_arr = np.array(S_F)
            M_F[k] = float(np.sum(weights * f_arr**2) / (np.sum(weights * f_arr) + 1e-30))

            cr_arr = np.array(S_CR)
            M_CR[k] = float(np.sum(weights * cr_arr))

            k = (k + 1) % H

        # ══════════════════════════════════════════════════════════════
        # КЛЮЧОВА ВІДМІННІСТЬ L-SHADE: Linear Population Size Reduction
        # NP лінійно зменшується від NP_init до NP_min
        # ══════════════════════════════════════════════════════════════
        NP_new = max(NP_min, round(NP_init - (NP_init - NP_min) * ec / budget))

        if NP_new < NP:
            # Видаляємо найгірших індивідів
            sorted_idx = np.argsort(fits[:NP])
            survivors = sorted_idx[:NP_new]
            pop = [pop[j] for j in survivors]
            fits = [fits[j] for j in survivors]
            NP = NP_new

    return {
        'loss': bl,
        'curve': curve,
        'seed': seed,
        'x_best': x_best.tolist() if x_best is not None else None
    }
