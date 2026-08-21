# Ex08-circles: Same as experiment config but with circles dataset
from config_experiment import CONFIG as _BASE

CONFIG = {**_BASE, 'dataset': 'circles'}
MODE_LABEL = "Full Profiling (SimpleMLP circles, s=0.10..0.97 (15 levels), 3 seeds, 70/20/10 split)"
