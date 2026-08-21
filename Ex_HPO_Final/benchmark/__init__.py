"""HPO Benchmark Framework — Configuration & Tier Definitions"""

ACTIVE_METHODS = [
    "sacma_v3",
    "sacma_base",
    "sacma_mab",
    "sacma_lazy",
    "antivanila",
    "tpe",
    "bo_gp",
    "shade",
    "lshade",
    "cmaes_pure",
    "whales_cma",
    "random_search",
    "smac_method",
    "dehb_method",
    "ordinv_cma",
]

# Методи для wall-clock дослідження (5 цільових)
WCT_METHODS = [
    "sacma_v3",
    "whales_cma",
    "tpe",
    "shade",
    "bo_gp",
]


TIERS = {
    'L0': {
        'datasets': ['synth_regression', 'synth_classification', 'synth_friedman'],
        'models': ['hgb', 'rf', 'mlp', 'svm', 'gb'],
        'default_seeds': 2,
        'budget': 30,
    },

    'L2': {
        'datasets': [
            'yahpo__lcbench__3945',         # AutoPyTorch MLP
            'yahpo__lcbench__7593',         # AutoPyTorch MLP
            'yahpo__lcbench__34539',        # AutoPyTorch MLP
            'yahpo__lcbench__126025',       # AutoPyTorch MLP
            'yahpo__lcbench__126026',       # AutoPyTorch MLP
            'yahpo__lcbench__167104',       # AutoPyTorch MLP
            'yahpo__lcbench__167149',       # AutoPyTorch MLP
            'yahpo__lcbench__167152',       # AutoPyTorch MLP
            'yahpo__lcbench__167168',       # AutoPyTorch MLP
            'yahpo__lcbench__168868',       # AutoPyTorch MLP
        ],
        'models': ['yahpo'],         # YAHPO surrogate
        'default_seeds': 10,
        'budget': 50,
    },

    # === Wall-Clock Time Analysis Tier ===
    # Ті самі LCBench датасети, але результати зберігаються окремо.
    # В JSON-записах будуть поля train_time_curve і wall_clock_curve.
    # Не конфліктує з results/L2/ — зберігається в results/L2_WCT/
    'L2_WCT': {
        'datasets': [
            'yahpo__lcbench__3945',
            'yahpo__lcbench__7593',
            'yahpo__lcbench__34539',
            'yahpo__lcbench__126025',
            'yahpo__lcbench__126026',
            'yahpo__lcbench__167104',
            'yahpo__lcbench__167149',
            'yahpo__lcbench__167152',
            'yahpo__lcbench__167168',
            'yahpo__lcbench__168868',
        ],
        'models': ['yahpo'],
        'default_seeds': 10,
        'budget': 50,
    },

    'L2_MLP_PD1': {
        'datasets': [
            # CNN / WideResNet — 3 датасети
            'cifar10_wresnet',              # WideResNet, CIFAR-10 (10 класів)
            'cifar100_wresnet',             # WideResNet, CIFAR-100 (100 класів)
            'svhn_wresnet',                 # WideResNet, SVHN (цифри вулиць)
            # CNN простіша архітектура — 2 датасети
            'mnist_cnn',                    # Max-Pooling CNN, MNIST
            'fashion_cnn',                  # Max-Pooling CNN, Fashion-MNIST
            # ResNet — 1 датасет
            'imagenet_resnet',              # ResNet, ImageNet (1000 класів)
            # Transformer / NLP — 3 датасети
            'lm1b_transformer',             # Transformer, LM1B (мовне моделювання)
            'translate_transformer',        # Transformer, WMT (переклад)
            'uniref50_transformer',         # Transformer, UniRef50 (білкові послідовності)
        ],
        'models': ['pd1'],         # сурогат PD1, D=4 (чистий HPO)
        'default_seeds': 10,
        'budget': 50,
    },
    'L3_NAS_SUPER': {
        'datasets': [
            'yahpo__nb301__None',
            'yahpo__iaml_super__40981'
        ],
        'models': ['yahpo'],
        'default_seeds': 2,
        'budget': 50,
    },
    'L4': {
        'datasets': [
            'sequential__digits',
            'residual__fashion_mini',
            'dense__digits',
        ],
        'models': ['l4'],          # real PyTorch training
        'default_seeds': 10,       # Increased to 10 for statistical significance
        'budget': 20,              
    },

    'L_ABLATION': {
        'datasets': [
            'yahpo__lcbench__3945',
            'yahpo__lcbench__7593',
            'yahpo__lcbench__34539',
        ],
        'models': ['yahpo'],
        'default_seeds': 10,
        'budget': 50,
    },

    # === FCNet Tabular Benchmark (Klein & Hutter, 2019) ===
    # 4 медичні датасети, 9 HP, табличний сурогат із реальним runtime.
    # Ідеальний для wall-clock аналізу: кожна конфігурація повертає
    # (valid_mse, runtime_seconds_for_full_training).
    'L5_FCNET': {
        'datasets': [
            'fcnet_protein',        # Protein Structure Prediction
            'fcnet_slice',          # CT Slice Localization
            'fcnet_naval',          # Naval Propulsion Maintenance
            'fcnet_parkinsons',     # Parkinsons Telemonitoring
        ],
        'models': ['fcnet'],
        'default_seeds': 10,
        'budget': 50,
    },
}
