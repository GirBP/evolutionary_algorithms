# Ex08-cnn: CompactCNN on FashionMNIST
# Tests pruning methods on a different architecture (CNN vs MLP)

CONFIG = {
    'model': 'CompactCNN',
    'dataset': 'fashionmnist',
    'seeds': [42, 123, 999],
    'sparsities': [0.50, 0.70, 0.80, 0.90, 0.93, 0.94, 0.95, 0.96],
    'n_runs_base': 3,
    'batch_size': 64,
    'epochs_pretrain': 50,
    'device': 'cpu',
    'max_workers': 9,

    # No early stopping for CNN (robust architecture)
    # 'early_stop_consecutive_fails': 2,
    # 'early_stop_f1_threshold': 0.15,

    # --- Micro-finetuning budgets ---
    'finetune_batches': 25,
    'finetune_batches_evo': 20,
    'finetune_batches_softmask': 20,
    'finetune_batches_set': 30,

    # --- CMA-ES ---
    'pop_size': 8,
    'max_evals': 40,

    # --- SET ---
    'set_epochs': 5,
    'set_zeta': 0.35,
    'set_update_every_k': 5,

    # --- SoftMask ---
    'softmask_iters': 30,
    'softmask_lr': 5e-3,
    'softmask_weight_decay': 0.05,
    'softmask_lambda_reg': 1.0,

    # --- EvoStruct ---
    'micro_lr': 0.01,
    'micro_steps': 10,
    'temperature_init': 1.0,
    'temperature_final': 0.05,
    'merge_threshold': 0.85,
    'spectral_n_clusters_ratio': 0.5,
    'f1_min_threshold': 0.70,
    'f1_recovery_finetune': 20,
}

MODE_LABEL = "Full Profiling (CompactCNN FashionMNIST, s=0.50..0.96, 3 seeds)"
