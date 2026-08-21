# config_ex08_1.py — Ex08.1: Порівняння TESA-26 vs DSA/LAMP/ERK
# Ті самі умови, що й Ex08 (moons, SimpleMLP), підмножина методів

CONFIG = {
    'model': 'SimpleMLP',
    'dataset': 'moons',
    'seeds': [42, 123, 999],
    'sparsities': [0.50, 0.70, 0.80, 0.85,
                   0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97],
    'n_runs_base': 3,
    'batch_size': 64,
    'epochs_pretrain': 100,
    'device': 'cpu',
    'max_workers': 6,

    # Методи для Ex08.1
    'methods': ['tesa26', 'lamp', 'erk', 'dsa', 'magnitude'],

    # Файнтюнінг
    'finetune_batches': 25,
    'finetune_batches_evo': 20,

    # CMA-ES (для tesa26)
    'pop_size': 8,
    'max_evals': 40,

    # DSA specific
    'dsa_steps': 20,
}

MODE_LABEL = "Ex08.1 — TESA-26 vs DSA/LAMP/ERK (moons, SimpleMLP, 12 sparsities, 3 seeds)"
