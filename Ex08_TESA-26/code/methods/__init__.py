# Ex08: Method registry
# Кожен метод — окремий модуль з функцією run(teacher_state, sp, seed, config, train_dl, test_dl) -> dict

from __future__ import annotations
from typing import Callable, Dict, List

_REGISTRY: Dict[str, dict] = {}


def register(key: str, display_name: str, color: str):
    """Декоратор для реєстрації методу."""
    def wrapper(func: Callable):
        _REGISTRY[key] = {
            'func': func,
            'display_name': display_name,
            'color': color,
            'key': key,
        }
        return func
    return wrapper


def get_method(key: str) -> dict:
    if key not in _REGISTRY:
        raise KeyError(f"Method '{key}' not registered. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[key]


def list_methods() -> List[str]:
    return list(_REGISTRY.keys())


def get_all_colors() -> Dict[str, str]:
    return {v['display_name']: v['color'] for v in _REGISTRY.values()}


def get_display_name(key: str) -> str:
    return _REGISTRY[key]['display_name']


# --- Import all method modules to trigger registration ---
from methods import (
    # magnitude,  — registered separately as magnitude + magnitude_v2 (ERK)
    magnitude,
    set_method,
    softmask_method,
    softmask_grad,
    vpam,
    # --- Removed Evo-SynFlow variants (underperforming in Ex08) ---
    # evo_synflow,           # baseline — AUSCa=0.575, #29
    # evo_synflow_adaptive,  # Adaptive — AUSCa=0.598, #27
    # evo_synflow_symwanda,  # SymWanda — AUSCa=0.641, #17
    evostruct,
    evo_hmt,
    tesa26,
    magnitude_v2,
    softmask_grad_v2,
    set_v2,
    # evo_synflow_v2,        # Taylor — AUSCa=0.630, #20
    fes_nsde,
    acde,
    eacde,
    evo_syn_flow_mgf,
    evo_synflow_ex07,
    sparsegpt,
    wanda_sota,
    ria,
    lamp,
    erk,
    dsa,
    earl,
    esmd,
    eeta,
    epqm,
    ehta,
    ehta_snr,
    tspe,
    obse,
    psmr,
    # atse_cma, atse_cma_v2,  — не в дисертації (§2.3 порівнює лише методи
    # таблиці 2.9/results_by_stage.md); модулі відсутні в публічній копії коду.
)
