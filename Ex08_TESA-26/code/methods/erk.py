# methods/erk.py — ERK: Erdős–Rényi–Kernel Sparsity Distribution
#
# Джерело: Evci et al. "Rigging the Lottery: Making All Tickets Winners"
#           (ICML 2020). https://arxiv.org/abs/1911.11134
#
# Алгоритм:
#   Для кожного шару l із розмірністю [n_out × n_in] keep-ratio визначається як:
#
#       keep_ratio_l ∝ (n_out + n_in) / (n_out × n_in)   [ERK formula]
#
#   Після нормалізації до глобального бюджету (1 - sp) застосовується magnitude
#   pruning із цими per-layer ratios.
#
#   Інтуїція: менші (shallow) шари мають більш "щільну" репрезентацію →
#   отримують вищий keep-ratio; великі шари можна проріджувати агресивніше.
#
# Для Conv-шарів: n_in = C_in * kH * kW, n_out = C_out
# Це відповідає оригінальній ERK-формулі з Evci et al.

import torch
import numpy as np
from methods import register
from or08_01 import (create_model, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity)


@register('erk', 'ERK', '#2ca02c')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()

    layers = model.get_prunable_layers()

    with torch.no_grad():
        # ── Крок 1: ERK-score для кожного шару ──
        erk_scores = []
        n_params = []
        for name, layer, mask_name in layers:
            W = layer.weight.data
            if W.dim() == 4:  # Conv2d: [out, in, kH, kW]
                n_out = W.shape[0]
                n_in = W.shape[1] * W.shape[2] * W.shape[3]
            else:  # Linear: [out, in]
                n_out, n_in = W.shape[0], W.shape[1]

            # ERK-score: (n_out + n_in) / (n_out * n_in)
            erk = (n_out + n_in) / (n_out * n_in)
            erk_scores.append(erk)
            n_params.append(W.numel())

        # ── Крок 2: нормалізація до глобального бюджету ──
        # keep_l = erk_l * C, де C = scaling constant for global budget
        # Σ_l (keep_l * n_params_l) = (1 - sp) * Σ_l n_params_l
        total_params = sum(n_params)
        target_params = (1.0 - sp) * total_params

        erk_arr = np.array(erk_scores, dtype=np.float64)
        n_arr = np.array(n_params, dtype=np.float64)

        # Binary search for scaling constant C > 0
        # Σ_l min(1.0, erk_l * C) * n_l = target_params
        lo, hi = 0.0, 1e6
        for _ in range(64):
            mid = (lo + hi) / 2.0
            kept = np.sum(np.minimum(1.0, erk_arr * mid) * n_arr)
            if kept < target_params:
                lo = mid
            else:
                hi = mid
        C = (lo + hi) / 2.0

        keep_ratios = np.minimum(1.0, erk_arr * C)

        # ── Крок 3: per-layer magnitude pruning із ERK ratios ──
        for (name, layer, mask_name), kr in zip(layers, keep_ratios):
            W = layer.weight.data
            if kr >= 1.0:
                mask = torch.ones_like(W)
            elif kr <= 0.0:
                mask = torch.zeros_like(W)
            else:
                w_abs = W.abs().view(-1)
                n_keep = max(1, int(round(kr * w_abs.numel())))
                thresh = torch.topk(w_abs, n_keep).values[-1]
                mask = (W.abs() >= thresh).float()
            getattr(model, mask_name).copy_(mask)

    # Топологічна перевірка
    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    # Мікро-файнтюнінг
    train_finetune_micro(model, train_dl, config.get('finetune_batches', 25))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
