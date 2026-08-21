# Ex01: Виконання експерименту (збереження даних)
# Запускає експеримент, зберігає сирі дані у JSON. За замовчуванням після збереження запускається візуалізація
# (тестовий запуск теж перевіряє графіки/таблиці — щоб виявити невідповідність до повного прогону). Вимкнення: --no-viz.
# Конфіг: config_test.py (тест) / config_experiment.py (повний).

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = CODE_DIR.parent
# Спільний common/ — на корені публічного репозиторію: EXPERIMENT_DIR (Ex01/) ->
# Ex01-03_CMA_Boundary/ -> repo root. На один .parent більше, ніж в оригіналі.
ROOT = EXPERIMENT_DIR.parent.parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CODE_DIR))

from common import (
    setup_experiment,
    save_experiment_data,
    clean_output_dir,
    clean_pycache,
)
from common.rcu import ANCHOR_LOOPS
from ex01 import run_full_experiment
from config import (
    DATA_DIR,
    N_SENTINEL,
    VIZ_SCRIPT,
)
import config_test
import config_experiment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ex01: Виконання експерименту (збереження даних)")
    parser.add_argument("-q", "--quick", action="store_true", help="Швидкий запуск, config_test.py")
    parser.add_argument(
        "-n",
        "--trials",
        nargs="?",
        type=int,
        const=N_SENTINEL,
        default=None,
        help=f"Експериментальний запуск (за замовч. {config_experiment.N_TRIALS} прогонів) або -n N",
    )
    parser.add_argument("--no-viz", action="store_true", help="Не запускати візуалізацію після збереження")
    args = parser.parse_args()
    if args.quick and args.trials is not None:
        parser.error("Не можна вказати одночасно -q та -n")
    return args


def _run_visualization(data_file: Path) -> None:
    output_dir = setup_experiment(EXPERIMENT_DIR) / "raw"
    clean_output_dir(output_dir)
    print("Запуск візуалізації...")
    subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / VIZ_SCRIPT), "--data", str(data_file.resolve())],
        cwd=str(ROOT),
        check=False,
    )
    print("Готово. Щоб перебудувати лише графіки: python ex01_visualize.py --data <файл>")


def _resolve_run_config(args: argparse.Namespace) -> tuple[int, Path, str]:
    if args.quick or args.trials is None:
        n_trials = config_test.N_TRIALS
        data_file = DATA_DIR / config_test.DATA_FILE
        mode_label = config_test.MODE_LABEL
    elif args.trials == N_SENTINEL:
        n_trials = config_experiment.N_TRIALS
        data_file = DATA_DIR / config_experiment.DATA_FILE_PATTERN.format(n=n_trials)
        mode_label = config_experiment.MODE_LABEL
    else:
        n_trials = args.trials
        if n_trials < 1:
            raise ValueError("Кількість прогонів (-n) має бути >= 1")
        data_file = DATA_DIR / config_experiment.DATA_FILE_PATTERN.format(n=n_trials)
        mode_label = f"кастомний ({n_trials} прогонів)"
    return n_trials, data_file, mode_label


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args = _parse_args()
    try:
        n_trials, data_file, mode_label = _resolve_run_config(args)
    except ValueError as e:
        print(f"Помилка: {e}")
        sys.exit(1)

    print(f"Режим: {mode_label}. Прогонів: {n_trials}, файл: {data_file.name}")

    print("\nЗапуск експерименту (RCU anchor: {ANCHOR_LOOPS} loops)...")
    df_conv, df_final, L_TARGET, n_out = run_full_experiment(n_trials=n_trials)

    if df_conv.empty and df_final.empty:
        print("Помилка: експеримент не повернув жодного результату.")
        sys.exit(1)

    experiment_data = {
        "convergence": df_conv,
        "final": df_final,
        "metadata": {
            "ANCHOR_LOOPS": ANCHOR_LOOPS,
            "L_TARGET": float(L_TARGET),
            "N_TRIALS": int(n_out),
            "DIM": 10,
            "MAX_FE": 3000,
        },
    }
    save_experiment_data(experiment_data, data_file)
    print(f"\nЕксперимент завершено. Дані збережено: {data_file}")

    if not args.no_viz:
        _run_visualization(data_file)

    clean_pycache(EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
