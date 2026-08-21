# Ex01: Запуск повного (експериментального) прогону
# Викликає code/ex01_run.py з опцією -n. Додатковий аргумент N (число) — кількість прогонів (напр. ex01_run_experiment.py 10).
# Решта аргументів (напр. --no-viz) передаються далі.

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    EXPERIMENT_DIR = Path(__file__).resolve().parent
    CODE_DIR = EXPERIMENT_DIR / "code"
    sys.path.insert(0, str(CODE_DIR))
    rest = [a for a in sys.argv[1:] if a not in ("-q", "--quick")]
    if rest and rest[0].isdigit():
        sys.argv = [sys.argv[0], "-n", rest[0]] + rest[1:]
    else:
        sys.argv = [sys.argv[0], "-n"] + rest
    import ex01_run  # noqa: E402  # з code/
    ex01_run.main()
