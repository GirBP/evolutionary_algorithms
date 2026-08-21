# Ex08-spirals: Same as experiment config but with spirals dataset
from config_experiment import CONFIG as _BASE

CONFIG = {**_BASE, 'dataset': 'spirals', 'sparsities': [0.10, 0.20, 0.30]}
MODE_LABEL = "Full Profiling (SimpleMLP spirals, s=0.10..0.30, 3 seeds, 70/20/10 split)"
