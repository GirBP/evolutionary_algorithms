# Ex08: Preview конфіг — швидкі попередні результати для ВСІХ методів
# 1 seed × 3 sparsities × 13 methods = 39 задач → ~10 хвилин

CONFIG = {
    'model': 'CompactCNN',
    'seeds': [42],
    'sparsities': [0.80, 0.95, 0.98],
    'n_runs_base': 1,
    'batch_size': 64,
    'epochs_pretrain': 15,
    'finetune_batches': 50,
    'device': 'cpu',
    'max_workers': 8,
    # --- CMA-ES (швидкий, але достатній) ---
    'pop_size': 8,
    'max_evals': 20,
    # --- SET ---
    'set_epochs': 2,
    'set_zeta': 0.2,
    # --- SoftMask ---
    'softmask_iters': 100,
    'softmask_lr': 5e-3,
    'softmask_weight_decay': 0.05,
    'softmask_lambda_reg': 1.0,
    # --- EvoStruct ---
    'micro_lr': 0.01,
    'micro_steps': 10,
    'temperature_init': 1.0,
    'temperature_final': 0.1,
    'merge_threshold': 0.85,
    'spectral_n_clusters_ratio': 0.5,
    'f1_min_threshold': 0.01,
    'f1_recovery_finetune': 20,
}

MODE_LABEL = "PREVIEW (всі методи, 1 seed, 3 sparsities)"
