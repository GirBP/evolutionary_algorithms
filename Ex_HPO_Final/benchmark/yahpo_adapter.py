"""
YAHPO Gym Adapter — Surrogate-based objective for various ML pipelines.
========================================================================
Перетворює YAHPO Gym об'єкти (ONNX сурогати) у формат об'єктивної функції
obj_fn(v) → loss, з яким працює наш фреймворк.

Підтримує SVM, XGBoost, Random Forest, ElasticNet, DARTS тощо.
"""

import os
import numpy as np

# Patch np.NaN for ConfigSpace compatibility with numpy 2.0
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
    np.float = float

from ConfigSpace import Configuration

# Ініціалізація YAHPO Gym
import yahpo_gym
from yahpo_gym import local_config

local_config.init_config()
target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'yahpo_data')
local_config.set_data_path(target_dir)

from yahpo_gym import benchmark_set

def get_yahpo_objective(dataset_name):
    """
    dataset_name у форматі 'yahpo__<scenario>__<instance>'
    Приклад: 'yahpo__rbv2_svm__1049'
    Повертає: (dim, make_objective)
    """
    parts = dataset_name.split('__')
    if len(parts) != 3 or parts[0] != 'yahpo':
        raise ValueError(f"Invalid YAHPO task name: {dataset_name}")
    
    scenario = parts[1]
    instance = parts[2]
    
    bench = benchmark_set.BenchmarkSet(scenario)
    if instance != 'None':
        bench.set_instance(instance)
    elif scenario == 'nb301' and bench.instances:
        bench.set_instance(bench.instances[0])
    
    cs = bench.get_opt_space()
    dim = len(cs.get_hyperparameters())
    
    def make_objective(Xt=None, yt=None, Xv=None, yv=None):
        import time
        
        def decode(v):
            v_clipped = np.clip(v, 0.0, 1.0)
            v_dict = {}
            for i, hp in enumerate(cs.get_hyperparameters()):
                if getattr(bench.config, 'instance_names', None) and hp.name in bench.config.instance_names:
                    if instance != 'None':
                        v_dict[hp.name] = str(instance)
                        continue
                    
                type_str = type(hp).__name__
                if 'Constant' in type_str:
                    v_dict[hp.name] = hp.value
                elif 'Categorical' in type_str:
                    num_choices = len(hp.choices)
                    idx = int(np.floor(v_clipped[i] * num_choices))
                    idx = min(max(idx, 0), num_choices - 1)
                    v_dict[hp.name] = hp.choices[idx]
                elif 'Integer' in type_str:
                    val = int(np.floor(hp.lower + v_clipped[i] * (hp.upper - hp.lower + 1)))
                    v_dict[hp.name] = int(min(max(val, hp.lower), hp.upper))
                elif hasattr(hp, 'log') and hp.log:
                    val = np.exp(np.log(hp.lower) + v_clipped[i] * (np.log(hp.upper) - np.log(hp.lower)))
                    v_dict[hp.name] = float(np.clip(val, hp.lower, hp.upper))
                else:
                    val = hp.lower + v_clipped[i] * (hp.upper - hp.lower)
                    v_dict[hp.name] = float(np.clip(val, hp.lower, hp.upper))

            # Динамічно видаляємо неактивні параметри
            active_keys = set(v_dict.keys())
            while True:
                try:
                    cfg = Configuration(cs, values={k: v_dict[k] for k in active_keys})
                    config = cfg.get_dictionary()
                    break
                except ValueError as e:
                    msg = str(e)
                    if "Inactive hyperparameter" in msg and "must not be specified" in msg:
                        hp_name = msg.split("'")[1]
                        active_keys.remove(hp_name)
                    else:
                        raise e
            
            # Додаємо максимальний fidelity
            fid_space = bench.get_fidelity_space()
            for fid_hp in fid_space.get_hyperparameters():
                config[fid_hp.name] = fid_hp.upper
                
            return config

        def get_test_metrics(v):
            try:
                config = decode(v)
                result = bench.objective_function([config])
                return result[0]
            except Exception:
                return {}

        def objective(v):
            st = time.thread_time_ns()
            try:
                config = decode(v)
                
                result = bench.objective_function([config])
                
                targets = bench.config.y_names
                
                # Вибір метрики для мінімізації
                if 'mmce' in targets:
                    loss = float(result[0]['mmce'])
                elif 'rmse' in targets:
                    loss = float(result[0]['rmse'])
                elif 'logloss' in targets:
                    loss = float(result[0]['logloss'])
                elif 'val_cross_entropy' in targets:
                    loss = float(result[0]['val_cross_entropy'])
                elif 'val_accuracy' in targets:
                    acc = float(result[0]['val_accuracy'])
                    # nb301 returns accuracy in %, i.e. ~91.7
                    # Normalize to [0,1] error rate: 1 - acc/100
                    if acc > 1.0:
                        loss = 1.0 - acc / 100.0
                    else:
                        loss = 1.0 - acc
                else:
                    loss = float(result[0][targets[0]])
                    
                # Зберігаємо час навчання (RCU_train) з метаданих YAHPO
                train_time = result[0].get('time', None)
                if train_time is not None:
                    objective.last_train_time = float(train_time)
                    objective.time_curve.append(float(train_time))
                    
            except Exception as e:
                loss = 1e6
                objective.last_train_time = None
                objective.time_curve.append(None)  # failed eval
                
            objective.total_time_ns += (time.thread_time_ns() - st)
            return loss
            
        objective.total_time_ns = 0
        objective.last_train_time = None   # час навчання останньої конфіг.
        objective.time_curve = []          # час навчання кожної конфіг. (wall-clock)
        objective.decode = decode
        objective.get_test_metrics = get_test_metrics
        return objective

    return dim, make_objective
