#!/usr/bin/env python3
"""
FCNet Benchmark Adapter — Tabular HPO benchmark (Klein & Hutter, 2019).
======================================================================
4 датасети (Protein, Slice, Naval, Parkinsons), 9 HP, ~60K табличних точок кожен.
Кожна конфігурація повертає (valid_mse, runtime_seconds).

Вимагає: fcnet HDF5 файли у data/fcnet/
"""
import os
import sys
import json
import numpy as np

# Monkey-patch щоб обійти `from nasbench import api` при імпорті пакету
import types
_fake_nb = types.ModuleType('nasbench')
_fake_nb.api = types.ModuleType('nasbench.api')
_fake_nb.lib = types.ModuleType('nasbench.lib')
_fake_nb.lib.graph_util = types.ModuleType('nasbench.lib.graph_util')
sys.modules['nasbench'] = _fake_nb
sys.modules['nasbench.api'] = _fake_nb.api
sys.modules['nasbench.lib'] = _fake_nb.lib
sys.modules['nasbench.lib.graph_util'] = _fake_nb.lib.graph_util

from tabular_benchmarks.fcnet_benchmark import (
    FCNetProteinStructureBenchmark,
    FCNetSliceLocalizationBenchmark,
    FCNetNavalPropulsionBenchmark,
    FCNetParkinsonsTelemonitoringBenchmark,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fcnet')

FCNET_TASKS = {
    'fcnet_protein':     FCNetProteinStructureBenchmark,
    'fcnet_slice':       FCNetSliceLocalizationBenchmark,
    'fcnet_naval':       FCNetNavalPropulsionBenchmark,
    'fcnet_parkinsons':  FCNetParkinsonsTelemonitoringBenchmark,
}

# 9 гіперпараметрів FCNet у порядку ConfigSpace
HP_SPEC = [
    ('activation_fn_1', ['relu', 'tanh']),
    ('activation_fn_2', ['relu', 'tanh']),
    ('batch_size',      [8, 16, 32, 64]),
    ('dropout_1',       [0.0, 0.3, 0.6]),
    ('dropout_2',       [0.0, 0.3, 0.6]),
    ('init_lr',         [5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]),
    ('lr_schedule',     ['const', 'cosine']),
    ('n_units_1',       [16, 32, 64, 128, 256, 512]),
    ('n_units_2',       [16, 32, 64, 128, 256, 512]),
]


def get_fcnet_objective(dataset_name):
    """
    Повертає:
        (dim, make_objective)
    де make_objective() → obj_fn(v) → float (valid_mse)
    а obj_fn.last_train_time містить runtime останнього виклику.
    """
    if dataset_name not in FCNET_TASKS:
        raise ValueError(f"Unknown FCNet task: {dataset_name}. Available: {list(FCNET_TASKS.keys())}")
    
    bench_cls = FCNET_TASKS[dataset_name]
    dim = len(HP_SPEC)  # 9
    
    def make_objective(Xt=None, yt=None, Xv=None, yv=None):
        import time
        import ConfigSpace
        
        bench = bench_cls(data_dir=DATA_DIR)
        cs = bench.get_configuration_space()
        
        def decode(v):
            v_clipped = np.clip(v, 0.0, 1.0)
            raw_config = {}
            for i, (hp_name, choices) in enumerate(HP_SPEC):
                idx = int(np.floor(v_clipped[i] * len(choices)))
                idx = min(max(idx, 0), len(choices) - 1)
                raw_config[hp_name] = choices[idx]
            
            cfg = ConfigSpace.Configuration(cs, values=raw_config)
            return cfg.get_dictionary()

        def get_test_metrics(v):
            try:
                config = decode(v)
                test_err, rtime = bench.objective_function_test(config)
                return {'final_test_error': test_err, 'runtime': rtime}
            except Exception:
                return {}

        def objective(v):
            st = time.thread_time_ns()
            try:
                config = decode(v)
                
                # FCNet повертає (valid_mse, runtime_in_seconds)
                valid_mse, runtime = bench.objective_function_deterministic(
                    config, budget=100, index=0
                )
                
                loss = float(valid_mse)
                objective.last_train_time = float(runtime)
                objective.time_curve.append(float(runtime))
                
            except Exception as e:
                loss = 1e6
                objective.last_train_time = None
                objective.time_curve.append(None)
                
            objective.total_time_ns += (time.thread_time_ns() - st)
            return loss
            
        objective.total_time_ns = 0
        objective.last_train_time = None
        objective.time_curve = []
        objective.decode = decode
        objective.get_test_metrics = get_test_metrics
        return objective
    
    return dim, make_objective

