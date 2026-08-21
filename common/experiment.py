# common/experiment.py — спільні компоненти для всіх експериментів

from __future__ import annotations

import sys
import time
from pathlib import Path

import os
from contextlib import contextmanager

import numpy as np

# Налаштування шляху для імпорту common модулів
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Локальні імпорти для уникнення циклічних залежностей
from common.style import set_thesis_style
from common.io import save_figure, save_table_latex, save_table_png, ensure_dir


@contextmanager
def suppress_stdout():
    """Тимчасово придушує stdout для прибирання зайвих повідомлень."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ==========================================
# ЕТАЛОН ЧАСУ (універсально для всього дослідження)
# ==========================================

# Комплексний еталон, який імітує реальні обчислення оптимізації
# Включає: матричні операції, тригонометричні функції, операції з пам'яттю
# Це забезпечує більш репрезентативну метрику для різних процесорів

# Параметри еталону (можна налаштувати для різних рівнів складності)
ETALON_DIM = 50  # Розмірність векторів/матриць для еталону
ETALON_ITERATIONS = 1300  # Кількість ітерацій комплексних операцій (налаштовано для ~0.3s)


def run_etalon() -> float:
    """
    Комплексний еталонний навантаження, який імітує реальні обчислення оптимізації.
    
    Включає:
    - Матричні операції (множення, обернення)
    - Векторні операції з тригонометричними функціями
    - Операції з пам'яттю (створення/копіювання масивів)
    - Комплексні обчислення, схожі на оцінку цільової функції
    
    Повертає CPU time (s).
    """
    t0 = time.process_time()
    
    # 1. Матричні операції (імітація обчислень коваріаційної матриці)
    A = np.random.randn(ETALON_DIM, ETALON_DIM).astype(np.float64)
    B = np.random.randn(ETALON_DIM, ETALON_DIM).astype(np.float64)
    for _ in range(ETALON_ITERATIONS):
        C = np.dot(A, B)  # Матричне множення
        A = 0.9 * A + 0.1 * C  # Оновлення з пам'яттю
    
    # 2. Векторні операції з тригонометричними функціями (імітація обчислення Rastrigin-подібних функцій)
    x = np.random.randn(ETALON_DIM * 10).astype(np.float64)
    for _ in range(ETALON_ITERATIONS):
        # Комплексні обчислення: квадрати, косинуси, суми
        y = np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        x = 0.95 * x + 0.05 * np.random.randn(len(x))
    
    # 3. Операції з пам'яттю та обчисленнями (імітація обробки популяції)
    vectors = [np.random.randn(ETALON_DIM).astype(np.float64) for _ in range(ETALON_DIM // 5)]
    for _ in range(ETALON_ITERATIONS):
        # Обчислення відстаней між векторами
        distances = np.zeros((len(vectors), len(vectors)))
        for i, v1 in enumerate(vectors):
            for j, v2 in enumerate(vectors):
                distances[i, j] = np.sqrt(np.sum((v1 - v2)**2))
        # Оновлення векторів
        vectors = [v + 0.1 * np.random.randn(ETALON_DIM) for v in vectors]
    
    # 4. Комплексні скалярні обчислення (імітація мета-обчислень)
    result = 0.0
    for _ in range(ETALON_ITERATIONS * 10):
        result = result * 1.0000001 + np.sin(result) * 0.1 + np.cos(result * 2) * 0.05
    
    t1 = time.process_time()
    return t1 - t0


def measure_etalon(n_runs: int = 20, seed: int | None = None) -> float:
    """
    Вимірює еталон n_runs разів і повертає середнє значення.
    
    Args:
        n_runs: Кількість прогонів для усереднення (за замовчуванням 20)
        seed: Опційно — seed для np.random перед вимірюванням (відтворюваність на тому ж комп'ютері)
    
    Returns:
        Середнє значення CPU time (s) за n_runs прогонів
    """
    if seed is not None:
        np.random.seed(seed)
    times = []
    for i in range(n_runs):
        t = run_etalon()
        times.append(t)
    return np.mean(times)


# ==========================================
# ІНІЦІАЛІЗАЦІЯ ЕКСПЕРИМЕНТУ
# ==========================================

def setup_experiment(experiment_dir: Path | str) -> Path:
    """
    Налаштування базових параметрів експерименту.
    
    Args:
        experiment_dir: Path до папки експерименту (напр., Path(__file__).resolve().parent).
    
    Returns:
        Path до папки results/.
    """
    set_thesis_style()
    output_dir = Path(experiment_dir) / "results"
    ensure_dir(output_dir)
    return output_dir
