# Ex08-blobs: Same as experiment config but with blobs dataset
from config_experiment import CONFIG as _BASE

CONFIG = {**_BASE, 'dataset': 'blobs'}
MODE_LABEL = "Full Profiling (SimpleMLP blobs, s=0.10..0.97 (15 levels), 3 seeds, 70/20/10 split)"
