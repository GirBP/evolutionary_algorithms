# Ex08: Baseline конфіг (Magnitude + Evo-SynFlow, wide sparsity range)

CONFIG = {
    'model': 'CompactResNet',       # 9-layer residual CNN (~914K params)
    'seeds': [42, 123, 999],
    'sparsities': [0.50, 0.60, 0.80, 0.90, 0.95, 0.98],
    'n_runs_base': 3,
    'batch_size': 64,
    'epochs_pretrain': 15,
    'finetune_batches': 100,
    'device': 'cpu',
    'max_workers': 9,
    # --- CMA-ES (for Evo-SynFlow) ---
    'pop_size': 12,
    'max_evals': 30,
    # --- SET (not used in baseline, but required by registry) ---
    'set_epochs': 3,
    'set_zeta': 0.2,
    # --- SoftMask (not used in baseline) ---
    'softmask_iters': 500,
    'softmask_lr': 5e-3,
    'softmask_weight_decay': 0.05,
    'softmask_lambda_reg': 1.0,
    # --- EvoStruct (not used in baseline) ---
    'micro_lr': 0.01,
    'micro_steps': 20,
    'temperature_init': 1.0,
    'temperature_final': 0.05,
    'merge_threshold': 0.85,
    'spectral_n_clusters_ratio': 0.5,
    'f1_min_threshold': 0.70,
    'f1_recovery_finetune': 50,
}

MODE_LABEL = "baseline (Magnitude + Evo-SynFlow, 50–98%)"
