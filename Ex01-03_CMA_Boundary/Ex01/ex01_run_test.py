# Ex01: Запуск тестового (швидкого) експерименту
# Викликає code/ex01_run.py з опцією -q. За замовчуванням після збереження даних запускається візуалізація —
# щоб одразу перевірити, чи графіки/таблиці підходять під задачу, без очікування повного прогону.
# Щоб не запускати візуалізацію: ex01_run_test.py --no-viz

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    EXPERIMENT_DIR = Path(__file__).resolve().parent
    CODE_DIR = EXPERIMENT_DIR / "code"
    sys.path.insert(0, str(CODE_DIR))
    # Пропускаємо -n/--trials; додаємо -q
    rest = [a for a in sys.argv[1:] if a not in ("-n", "--trials")]
    if rest and rest[0].isdigit():
        rest = rest[1:]
    sys.argv = [sys.argv[0], "-q"] + rest
    import ex01_run  # noqa: E402  # з code/
    ex01_run.main()
