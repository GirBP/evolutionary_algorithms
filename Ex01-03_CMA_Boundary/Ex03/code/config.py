# Ex03: Конфігурація експерименту (шляхи та імена файлів запуску)
# Параметри експерименту (n_runs, time_limit, datasets, methods) — у ex03.py (CONFIG_TEST, CONFIG_EXPERIMENT).

from pathlib import Path

# Корінь експерименту (папка Ex03; конфіг у Ex03/code/)
EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"

N_SENTINEL = -1

DATA_FILE_QUICK = "ex03_data_quick.json"
DATA_FILE_TRIALS_PATTERN = "ex03_data_n{n}.json"
VIZ_SCRIPT = "ex03_visualize.py"
