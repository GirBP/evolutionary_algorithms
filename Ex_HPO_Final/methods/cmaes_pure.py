"""CMA-ES — Covariance Matrix Adaptation Evolution Strategy (Hansen, 2001/2016)

Використовує офіційну бібліотеку `cmaes` (Masashi Shibata, 2024).
Це pure-Python реалізація з 100% відповідністю до оригінального алгоритму.

CMA-ES — золотий стандарт derivative-free оптимізації.
Адаптує повну коваріаційну матрицю для направленого пошуку.

Залежності: cmaes (pip install cmaes), numpy
"""
import numpy as np
from cmaes import CMA


def run(seed, obj_fn, dim, budget):
    """
    CMA-ES через офіційну бібліотеку `cmaes`.
    Args:
        seed: random seed
        obj_fn: callable, v ∈ [0,1]^dim → float (minimize)
        dim: dimensionality of search space
        budget: max number of real evaluations
    Returns:
        {'loss': float, 'curve': list, 'seed': int, 'x_best': list}
    """
    # ── Параметри CMA-ES ───────────────────────────────────────────────
    # Стартова точка: центр простору [0.5, ..., 0.5]
    mean = np.full(dim, 0.5)
    sigma = 0.3  # Початковий крок (стандарт для [0,1] простору)

    # Розмір популяції: за замовчуванням CMA-ES
    population_size = 4 + int(3 * np.log(dim))

    optimizer = CMA(
        mean=mean,
        sigma=sigma,
        bounds=np.array([[0.0, 1.0]] * dim),
        seed=seed,
        population_size=population_size,
    )

    # ── Основний цикл ──────────────────────────────────────────────────
    curve = []
    bl = float('inf')
    x_best = None
    ec = 0

    while ec < budget:
        solutions = []

        for _ in range(population_size):
            if ec >= budget:
                break
            x = optimizer.ask()
            x_clipped = np.clip(x, 0.0, 1.0)
            y = obj_fn(x_clipped)
            ec += 1
            solutions.append((x, y))

            if y < bl:
                bl = y
                x_best = x_clipped.copy()
            curve.append(bl)

        # tell() вимагає рівно popsize рішень
        if len(solutions) == population_size:
            optimizer.tell(solutions)

        # CMA-ES може зупинитися раніше (збіжність)
        if optimizer.should_stop():
            break

    # Допадаємо криву якщо зупинились раніше
    while len(curve) < budget:
        curve.append(curve[-1] if curve else 1e9)

    return {
        'loss': bl,
        'curve': curve[:budget],
        'seed': seed,
        'x_best': x_best.tolist() if x_best is not None else None
    }
