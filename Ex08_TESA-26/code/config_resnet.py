# config_resnet.py — CompactResNet (FashionMNIST) top-5 methods
CONFIG = {
    'model': 'CompactResNet',
    'dataset': 'fashionmnist',
    'seeds': [42, 123, 999],
    'sparsities': [0.50, 0.70, 0.80, 0.90, 0.93, 0.94, 0.95, 0.96],
    'n_runs_base': 3,
    'batch_size': 64,
    'epochs_pretrain': 10,
    'max_workers': 9,

    # No early stopping
    # 'early_stop_consecutive_fails': 2,
    # 'early_stop_f1_threshold': 0.15,

    # --- Micro-finetuning budgets ---
    'finetune_batches': 25,
    'finetune_batches_evo': 20,

    # --- CMA-ES / DE ---
    'pop_size': 12,
    'fes_generations': 10,
    'max_evals': 40,

    # --- Top-5 methods only ---
    'methods': ['ehta-snr'],
}

MODE_LABEL = 'CompactResNet FashionMNIST, s=0.50..0.96, 3 seeds'
