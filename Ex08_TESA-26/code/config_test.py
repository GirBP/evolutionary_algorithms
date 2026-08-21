# Ex08: Smoke test конфіг (~5 секунд на задачу, ~2,500 RCU)
# Мета: перевірити що всі 16 методів запускаються без помилок

CONFIG = {
    'model': 'CompactResNet',
    'seeds': [42],
    'sparsities': [0.50],
    'n_runs_base': 1,
    'batch_size': 64,
    'epochs_pretrain': 50,          # uses cached teacher — not retrained
    'device': 'cpu',
    'max_workers': 9,

    # --- ~2,500 RCU budget (5 seconds) ---
    'finetune_batches': 10,         # Simple methods: ~1500 + 1000 base ≈ 2,500

    # --- CMA-ES ---
    'pop_size': 4,
    'max_evals': 4,                 # minimal: 1 generation
    'finetune_batches_evo': 5,

    # --- SET ---
    'set_epochs': 1,
    'set_zeta': 0.2,
    'finetune_batches_set': 5,

    # --- SoftMask ---
    'softmask_iters': 10,
    'softmask_lr': 5e-3,
    'softmask_weight_decay': 0.05,
    'softmask_lambda_reg': 1.0,
    'finetune_batches_softmask': 5,

    # --- EvoStruct ---
    'micro_lr': 0.01,
    'micro_steps': 3,
    'temperature_init': 1.0,
    'temperature_final': 0.05,
    'merge_threshold': 0.85,
    'spectral_n_clusters_ratio': 0.5,
    'f1_min_threshold': 0.01,
    'f1_recovery_finetune': 3,
}

MODE_LABEL = "SMOKE TEST (~5s/task)"
