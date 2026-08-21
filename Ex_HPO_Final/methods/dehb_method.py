"""
DEHB-core — Differential Evolution with Successive Halving scheduling.

Based on: Awad et al., "DEHB: Evolutionary Hyperband for Scalable,
Robust and Efficient Hyperparameter Optimization", IJCAI 2021.

Core algorithm:
  - DE/rand/1/bin mutation + binomial crossover
  - Successive Halving bracket scheduler (Hyperband-style)
  - Since our benchmark is single-fidelity, only the DE component
    is active (no early-stopping benefit from multi-fidelity).

NOTE: The official DEHB library (v0.1.2) is incompatible with
ConfigSpace 0.6.1 (required by YAHPO Gym) due to internal API changes.
This is a faithful reimplementation of the DE core from the paper.
"""

import numpy as np
from benchmark.init import sobol_init


def run(seed, obj_fn, dim, budget):
    """
    DEHB-core: Differential Evolution for HPO benchmark.

    Implements DE/rand/1/bin from Algorithm 1 of Awad et al., IJCAI 2021,
    without the multi-fidelity Hyperband scheduling (since our tasks
    are single-fidelity).

    Args:
        seed: random seed
        obj_fn: callable, v ∈ [0,1]^dim → float (minimize)
        dim: dimensionality
        budget: max real evaluations
    Returns:
        {'loss': float, 'curve': list, 'seed': int, 'x_best': list}
    """
    rng = np.random.default_rng(seed)

    # ── DE hyperparameters (DEHB defaults) ─────────────────────────────
    F = 0.5       # mutation factor
    CR = 0.5      # crossover probability
    pop_size = max(2 * dim, 10)
    pop_size = min(pop_size, budget // 2)  # at least 2 generations
    pop_size = max(pop_size, 4)            # minimum viable population

    curve = []
    bl = float('inf')
    x_best = None

    def evaluate(v):
        nonlocal bl, x_best
        v_clipped = np.clip(v, 0.0, 1.0)
        fl = obj_fn(v_clipped)
        if fl < bl:
            bl = fl
            x_best = v_clipped.copy()
        curve.append(bl)
        return fl

    # ── Initialize population ──────────────────────────────────────────
    pop = sobol_init(seed, dim, pop_size)
    fitness = np.array([evaluate(pop[i]) for i in range(pop_size)])
    ec = pop_size

    # ── DE/rand/1/bin evolution loop ───────────────────────────────────
    while ec < budget:
        for i in range(pop_size):
            if ec >= budget:
                break

            # Mutation: rand/1
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = np.clip(mutant, 0.0, 1.0)

            # Binomial crossover
            cross_points = rng.random(dim) < CR
            j_rand = rng.integers(dim)
            cross_points[j_rand] = True
            trial = np.where(cross_points, mutant, pop[i])

            # Selection
            trial_fit = evaluate(trial)
            ec += 1
            if trial_fit <= fitness[i]:
                pop[i] = trial
                fitness[i] = trial_fit

    # Pad curve
    while len(curve) < budget:
        curve.append(bl)

    return {
        'loss': bl,
        'curve': curve[:budget],
        'seed': seed,
        'x_best': x_best.tolist() if x_best is not None else None,
    }
