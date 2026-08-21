# Ex01: Шляхи та спільні константи (не параметри тест/експеримент)
# Параметри тесту та експерименту — у config_test.py та config_experiment.py.

from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = EXPERIMENT_DIR / "data"

N_SENTINEL = -1

MEASURE_ETALON_RUNS = 20
MEASURE_ETALON_SEED = 0

VIZ_SCRIPT = "ex01_visualize.py"
