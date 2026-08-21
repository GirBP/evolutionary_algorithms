"""SHADE — Success-History based Adaptive DE (Tanabe & Fukunaga, 2013)
Повна реалізація з адаптацією F та CR через success-history.

Відмінності від shade.py (DE_fixed):
  - F вибирається з Cauchy(M_F[r], 0.1) де M_F — ковзне середнє успішних F
  - CR вибирається з N(M_CR[r], 0.1) де M_CR — ковзне середнє успішних CR
  - Lehmer mean для оновлення M_F (зважений за покращенням)
  - Архів зовнішніх рішень для current-to-pbest/1
"""
import numpy as np
from benchmark.init import sobol_init


def run(seed, obj_fn, dim, budget):
    """
    SHADE з повною success-history адаптацією.
    Args:
        seed: random seed
        obj_fn: callable, v ∈ [0,1]^dim → float (minimize)
        dim: dimensionality of search space
        budget: max number of real evaluations
    Returns:
        {'loss': float, 'curve': list, 'seed': int}
    """
    rng = np.random.default_rng(seed)
    
    # ── Параметри ──────────────────────────────────────────────────────
    NP = max(7, 2 * dim)            # розмір популяції
    NP = min(NP, budget // 2)       # не більше половини бюджету
    H = NP                          # розмір історії параметрів
    p_best_rate = 0.1               # відсоток кращих для pbest
    
    # ── Ініціалізація популяції (Sobol) ───────────────────────────────
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
    
    # ── Success-History пам'ять ───────────────────────────────────────
    M_F = np.full(H, 0.5)          # ковзні середні успішних F
    M_CR = np.full(H, 0.5)         # ковзні середні успішних CR
    k = 0                           # індекс запису в історії
    
    # ── Архів ─────────────────────────────────────────────────────────
    archive = []
    archive_max = NP
    
    # ── Головний цикл ─────────────────────────────────────────────────
    while ec < budget:
        S_F = []                    # успішні F цього покоління
        S_CR = []                   # успішні CR цього покоління
        S_delta = []                # покращення fitness для зваження
        
        for i in range(NP):
            if ec >= budget:
                break
            
            # ── Генерація F та CR із success-history ──────────────────
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
            
            # ── Мутація: current-to-pbest/1 з архівом ────────────────
            # Вибір pbest
            n_pbest = max(1, int(np.ceil(p_best_rate * NP)))
            sorted_idx = np.argsort(fits)
            pbest_pool = sorted_idx[:n_pbest]
            x_pbest = pop[rng.choice(pbest_pool)]
            
            # Вибір r1 ≠ i з популяції
            candidates_r1 = [j for j in range(NP) if j != i]
            r1 = rng.choice(candidates_r1)
            
            # Вибір r2 ≠ i, r1 з популяції ∪ архіву
            union = list(range(NP)) + list(range(NP, NP + len(archive)))
            candidates_r2 = [j for j in union if j != i and j != r1]
            if not candidates_r2:
                candidates_r2 = [j for j in range(NP) if j != i]
            r2_idx = rng.choice(candidates_r2)
            
            if r2_idx < NP:
                x_r2 = pop[r2_idx]
            else:
                x_r2 = archive[r2_idx - NP]
            
            # Мутантний вектор
            v = pop[i] + Fi * (x_pbest - pop[i]) + Fi * (pop[r1] - x_r2)
            v = np.clip(v, 0, 1)
            
            # ── Кросовер (binomial) ──────────────────────────────────
            u = pop[i].copy()
            j_rand = rng.integers(dim)
            for d in range(dim):
                if rng.random() < CRi or d == j_rand:
                    u[d] = v[d]
            
            # ── Селекція ─────────────────────────────────────────────
            fl = obj_fn(u)
            ec += 1
            
            if fl <= fits[i]:
                # Зберігаємо старе рішення в архів
                archive.append(pop[i].copy())
                if len(archive) > archive_max:
                    archive.pop(rng.integers(len(archive)))
                
                # Записуємо успішні параметри
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
        
        # ── Оновлення Success-History ────────────────────────────────
        if S_F and S_CR:
            deltas = np.array(S_delta)
            weights = deltas / (deltas.sum() + 1e-30)  # зважуємо за покращенням
            
            # Lehmer mean для M_F (зважений)
            f_arr = np.array(S_F)
            M_F[k] = float(np.sum(weights * f_arr**2) / (np.sum(weights * f_arr) + 1e-30))
            
            # Зважене середнє для M_CR
            cr_arr = np.array(S_CR)
            M_CR[k] = float(np.sum(weights * cr_arr))
            
            k = (k + 1) % H
    
    return {'loss': bl, 'curve': curve, 'seed': seed, 'x_best': x_best.tolist() if x_best is not None else None}
