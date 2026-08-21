# common/rcu_drift_calibration.py — стрес-тест дрейфу метрики RCU під CPU-навантаженням
#
# Незалежно вимірює еталонну задачу (common.experiment.run_etalon) N разів:
#   (1) у чистих умовах (без фонового навантаження);
#   (2) під фоновим CPU-навантаженням (background-потоки з матричним множенням,
#       звільняють GIL через BLAS — реально конкурують за процесорні ядра).
# Для кожного заміру фіксується як RCU-оцінка (сендвіч-профілювання через анкер,
# common.rcu.profile_rcu), так і "наївний" настінний час (wall-clock).
# Звітується відносний дрейф середнього значення кожної метрики між умовами, %.
#
# Дає незалежну емпіричну перевірку тези підрозділу 2.1.2 дисертації: нормалізація
# через локальний еталон (RCU) стійкіша до фонових збурень, ніж настінний час.
# Це спрощений одно-умовний (CPU-навантаження) відтворюваний стрес-тест, а НЕ
# повторення повного протоколу підрозділу 2.1.2 (4 типи навантаження × 5 умов).
# Числа з rcu_drift_report.json НЕ підставляються в текст дисертації автоматично.
#
# Використання:
#   python common/rcu_drift_calibration.py            # повний прогін (~ 15-30 с)
#   python common/rcu_drift_calibration.py --smoke    # швидка перевірка працездатності

from __future__ import annotations

import argparse
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy import stats

# --- Шляхи (відносно кореня репозиторію) ---
COMMON_DIR = Path(__file__).resolve().parent
ROOT = COMMON_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.rcu import profile_rcu, setup_rcu_worker
from common.experiment import run_etalon


# ==========================================
# Фонове CPU-навантаження
# ==========================================

def _cpu_load_worker(stop_event: threading.Event, dim: int = 256) -> None:
    """Один потік фонового навантаження: неперервне матричне множення.
    NumPy/BLAS-виклики звільняють GIL, тому потоки реально конкурують за ядра."""
    a = np.random.randn(dim, dim)
    b = np.random.randn(dim, dim)
    while not stop_event.is_set():
        c = a @ b
        a = 0.999 * a + 0.001 * c


class BackgroundLoad:
    """Контекст-менеджер: запускає/зупиняє N потоків фонового CPU-навантаження."""

    def __init__(self, n_threads: int) -> None:
        self.n_threads = n_threads
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> "BackgroundLoad":
        if self.n_threads > 0:
            self._stop.clear()
            self._threads = [
                threading.Thread(target=_cpu_load_worker, args=(self._stop,), daemon=True)
                for _ in range(self.n_threads)
            ]
            for t in self._threads:
                t.start()
            time.sleep(0.05)  # дати потокам стартувати перед вимірюванням
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)


# ==========================================
# Замір: RCU + настінний час за один прогін еталону
# ==========================================

def measure_once(seed: int | None = None) -> tuple[float, float]:
    """Один сендвіч-замір еталонної задачі. Повертає (rcu, wall_seconds)."""
    if seed is not None:
        np.random.seed(seed)
    w0 = time.perf_counter()
    _, rcu, _, _ = profile_rcu(run_etalon)
    wall = time.perf_counter() - w0
    return rcu, wall


def run_condition(n_reps: int, seed_base: int) -> dict:
    rcu_vals: list[float] = []
    wall_vals: list[float] = []
    for i in range(n_reps):
        rcu, wall = measure_once(seed=seed_base + i)
        rcu_vals.append(rcu)
        wall_vals.append(wall)
    return {
        "rcu": rcu_vals,
        "wall_s": wall_vals,
        "rcu_mean": float(np.mean(rcu_vals)),
        "rcu_std": float(np.std(rcu_vals)),
        "wall_mean_s": float(np.mean(wall_vals)),
        "wall_std_s": float(np.std(wall_vals)),
    }


def pct_drift(baseline: float, stressed: float) -> float:
    if baseline == 0:
        return float("nan")
    return (stressed - baseline) / baseline * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                         help="швидка перевірка (мало повторів, менше фонових потоків)")
    parser.add_argument("--reps", type=int, default=None,
                         help="кількість повторів на умову (за замовч. 20, у --smoke 3)")
    parser.add_argument("--bg-threads", type=int, default=None,
                         help="кількість потоків фонового навантаження (за замовч. 4, у --smoke 2)")
    parser.add_argument("--output", type=Path, default=None, help="шлях до вихідного JSON")
    args = parser.parse_args()

    smoke = args.smoke
    n_reps = args.reps if args.reps is not None else (3 if smoke else 20)
    n_bg = args.bg_threads if args.bg_threads is not None else (2 if smoke else 4)
    out_path = args.output or (COMMON_DIR / "results" / "rcu_drift_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    setup_rcu_worker()

    print("=" * 70)
    print("  RCU DRIFT CALIBRATION — дрейф RCU vs настінного часу під навантаженням")
    print("=" * 70)
    print(f"  Режим: {'SMOKE' if smoke else 'ПОВНИЙ'}; повторів на умову: {n_reps}; "
          f"фонових потоків: {n_bg}")

    print("\n[1/2] Чисті умови (без фонового навантаження)...")
    baseline = run_condition(n_reps, seed_base=1000)
    print(f"  RCU:  {baseline['rcu_mean']:.4f} ± {baseline['rcu_std']:.4f}")
    print(f"  Wall: {baseline['wall_mean_s']:.4f} с ± {baseline['wall_std_s']:.4f} с")

    print(f"\n[2/2] Під фоновим CPU-навантаженням ({n_bg} потоки)...")
    with BackgroundLoad(n_bg):
        stressed = run_condition(n_reps, seed_base=2000)
    print(f"  RCU:  {stressed['rcu_mean']:.4f} ± {stressed['rcu_std']:.4f}")
    print(f"  Wall: {stressed['wall_mean_s']:.4f} с ± {stressed['wall_std_s']:.4f} с")

    drift_rcu = pct_drift(baseline["rcu_mean"], stressed["rcu_mean"])
    drift_wall = pct_drift(baseline["wall_mean_s"], stressed["wall_mean_s"])

    # Mann-Whitney U — незалежні вибірки (умови не парні за реплікою)
    try:
        _, p_rcu = stats.mannwhitneyu(baseline["rcu"], stressed["rcu"], alternative="two-sided")
        _, p_wall = stats.mannwhitneyu(baseline["wall_s"], stressed["wall_s"], alternative="two-sided")
    except ValueError:
        p_rcu = p_wall = float("nan")

    report = {
        "smoke": smoke,
        "n_reps_per_condition": n_reps,
        "n_bg_threads": n_bg,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "baseline": baseline,
        "stressed": stressed,
        "drift_rcu_pct": drift_rcu,
        "drift_wall_pct": drift_wall,
        "mannwhitney_p_rcu": float(p_rcu),
        "mannwhitney_p_wall": float(p_wall),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("  РЕЗУЛЬТАТ")
    print("=" * 70)
    print(f"  Дрейф RCU під CPU-навантаженням:             {drift_rcu:+.2f}%  (p={p_rcu:.4f})")
    print(f"  Дрейф настінного часу під CPU-навантаженням: {drift_wall:+.2f}%  (p={p_wall:.4f})")
    print(f"\n  Звіт збережено: {out_path}")


if __name__ == "__main__":
    main()
