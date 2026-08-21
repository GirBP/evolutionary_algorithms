# Ex09: Sparse-to-Dense Network Conversion
# Мета: перетворити розріджену мережу у компактну dense мережу
# з мінімальною втратою якості
#
# Pre-protocol config (Knuth protocol §1)

CONFIG = {
    'model': 'SimpleMLP',
    'datasets': ['moons', 'circles', 'spirals', 'blobs',
                 'gaussian_quantiles', 'classification',
                 'highdim', 'sequence_cls'],
    'seeds': [42, 123],
    'target_sparsity': 0.90,
    'min_f1_weighted': 0.80,  # sparse model must retain ≥80% F1
    'batch_size': 64,
    'epochs_pretrain': 100,
    'device': 'cpu',

    # Finetuning for compact model
    'finetune_epochs': 10,
    'finetune_lr': 0.01,

    # Conversion methods
    'conversion_methods': [
        'neuron_removal',
        'svd_compression',
        'knowledge_distill',
        'weight_redistribution',
        'evomerge',
    ],
}
