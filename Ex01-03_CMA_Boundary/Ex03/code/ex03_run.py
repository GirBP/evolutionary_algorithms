# Ex03: Виконання експерименту (збереження даних)
# Запускає експеримент, зберігає дані у JSON. Візуалізація опційно після збереження.
# Щоб лише перебудувати графіки: python ex03_visualize.py [--data <файл>]

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Корінь проєкту для common (файл у Ex01-03_CMA_Boundary/Ex03/code/; спільний
# common/ — на корені публічного репозиторію, тому на один .parent більше,
# ніж в оригіналі Ex03/code/)
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from common import clean_output_dir, setup_experiment, save_experiment_data, clean_pycache

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ex03 import CONFIG_EXPERIMENT, CONFIG_TEST, run_full_experiment
from config import (
    DATA_DIR,
    DATA_FILE_QUICK,
    DATA_FILE_TRIALS_PATTERN,
    EXPERIMENT_DIR,
    N_SENTINEL,
    VIZ_SCRIPT,
)





def _parse_args() -> argparse.Namespace:
    """Парсить аргументи CLI. Перевіряє взаємовиключні опції (-q та -n)."""
    parser = argparse.ArgumentParser(
        description="Ex03: Запуск експерименту (Adam vs EA, класифікація)"
    )
    parser.add_argument(
        "-q",
        "--quick",
        action="store_true",
        help="Швидкий (тестовий) запуск, файл ex03_data_quick.json",
    )
    parser.add_argument(
        "-n",
        "--trials",
        nargs="?",
        type=int,
        const=N_SENTINEL,
        default=None,
        help="Експериментальний запуск (за замовч. n_runs з CONFIG_EXPERIMENT) або -n N для N прогонів",
    )
    parser.add_argument(
        "--datasets",
        "-d",
        nargs="*",
        default=None,
        help="Датасети: moons, classification20, digits. За замовчуванням: з режиму",
    )
    parser.add_argument(
        "--methods",
        "-m",
        nargs="*",
        default=None,
        help="Методи: Adam, CMA-ES, L-SHADE, CLPSO, RTS, RS. За замовчуванням: з режиму",
    )
    parser.add_argument(
        "--rcu-budget",
        "-b",
        type=int,
        default=None,
        help="Перевизначити бюджет RCU на run",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Не запускати візуалізацію після збереження",
    )
    args = parser.parse_args()
    if args.quick and args.trials is not None:
        parser.error("Не можна вказати одночасно -q та -n")
    return args


def _resolve_run_config(
    args: argparse.Namespace,
) -> tuple[int, Path, str, dict]:
    """Повертає (n_runs, data_file, mode_label, config). config — словник для run_full_experiment."""
    if args.quick or args.trials is None:
        config = dict(CONFIG_TEST)
        n_runs = config["n_runs"]
        data_file = DATA_DIR / DATA_FILE_QUICK
        mode_label = "швидкий (тестовий)"
    elif args.trials == N_SENTINEL:
        config = dict(CONFIG_EXPERIMENT)
        n_runs = config["n_runs"]
        data_file = DATA_DIR / DATA_FILE_TRIALS_PATTERN.format(n=n_runs)
        mode_label = "експериментальний"
    else:
        n_runs = args.trials
        if n_runs < 1:
            raise ValueError("Кількість прогонів (-n) має бути >= 1")
        config = dict(CONFIG_EXPERIMENT)
        config["n_runs"] = n_runs
        data_file = DATA_DIR / DATA_FILE_TRIALS_PATTERN.format(n=n_runs)
        mode_label = f"кастомний ({n_runs} прогонів)"

    if args.rcu_budget is not None:
        config["rcu_budget"] = args.rcu_budget
    return n_runs, data_file, mode_label, config


def _run_visualization(data_file: Path) -> None:
    """Очищає results/, запускає ex03_visualize.py з --data, cwd=ROOT."""
    output_dir = setup_experiment(EXPERIMENT_DIR) / "raw"
    clean_output_dir(output_dir)
    print("Запуск візуалізації...")
    subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / VIZ_SCRIPT), "--data", str(data_file.resolve())],
        cwd=str(ROOT),
        check=False,
    )
    print("Готово. Щоб перебудувати лише графіки: python ex03_visualize.py --data <файл>")


def main() -> None:
    """Точка входу: парсинг, експеримент, збереження, опційно візуалізація."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    args = _parse_args()
    try:
        n_runs, data_file, mode_label, config = _resolve_run_config(args)
    except ValueError as e:
        print(f"Помилка: {e}")
        sys.exit(1)

    datasets = args.datasets or config["datasets"]
    methods = args.methods or config["methods"]
    total_jobs = len(datasets) * len(methods) * n_runs
    W = min(os.cpu_count() or 4, total_jobs)
    print(
        f"Режим: {mode_label}. n_runs={n_runs}, файл: {data_file.name}. "
        f"Датасети: {datasets}, методи: {methods}, rcu_budget={config['rcu_budget']}. "
        f"Паралелізм: {W} процесів, {total_jobs} завдань."
    )

    df_final, df_convergence, metadata = run_full_experiment(
        n_runs=n_runs, datasets=datasets, methods=methods, config=config
    )
    metadata["config_mode"] = "test" if (args.quick or args.trials is None) else "experiment"

    if df_final.empty and df_convergence.empty:
        print("Помилка: експеримент не повернув жодного результату.")
        sys.exit(1)

    experiment_data = {
        "convergence": df_convergence,
        "final": df_final,
        "metadata": metadata,
    }
    save_experiment_data(experiment_data, data_file)
    print(f"Дані збережено: {data_file}")

    if not args.no_viz:
        _run_visualization(data_file)

    clean_pycache(EXPERIMENT_DIR)


if __name__ == "__main__":
    main()
