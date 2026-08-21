# Ex08: Multi-Fidelity Benchmarking config
# Stress-test at s=0.95, micro-finetuning, static proxy

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
    'max_workers': 9,

    # --- Micro-finetuning budgets (OneCycleLR) ---
    'finetune_batches': 25,             # one-shot methods (was 200)
    'finetune_batches_evo': 20,         # evo methods (was 100)
    'finetune_batches_softmask': 20,    # softmask (was 50)
    'finetune_batches_set': 30,         # SET high-freq (was 50)

    # --- CMA-ES ---
    'pop_size': 8,
    'max_evals': 40,

    # --- SET (High-Frequency) ---
    'set_epochs': 5,
    'set_zeta': 0.35,                   # aggressive exploration (was 0.2)
    'set_update_every_k': 5,            # topology update every K batches

    # --- SoftMask (reduced iters) ---
    'softmask_iters': 30,              # was 100
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

MODE_LABEL = "Full Profiling (SimpleMLP moons, s=0.50..0.97 (12 levels), 3 seeds, 70/20/10 split)"
