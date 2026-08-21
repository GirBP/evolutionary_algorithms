# methods/dsa.py — DSA: Differentiable Sparsity Allocation
#
# Джерело: Ning et al. "DSA: More Efficient Budgeted Pruning via
#           Differentiable Sparsity Allocation" (ECCV 2020).
#           https://doi.org/10.1007/978-3-030-58580-8_35
#
# Реалізована ідея:
#   Оригінальний DSA оптимізує неперервні prune-ratios через ADMM.
#   У цьому фреймворку реалізовано диференційовний підхід через
#   soft-thresholding + projected gradient descent на симплексі keep-ratios.
#
# Алгоритм:
#   1. Ініціалізувати keep-ratios r_l рівномірно (1 - sp) для всіх шарів.
#   2. На кожній ітерації:
#      a) Обчислити Taylor saliency S_l = Σ |w * ∂L/∂w| для кожного шару.
#      b) Апроксимувати градієнт: прошарки з вищою sensitivity мають
#         вищий ∂loss/∂r_l → зменшуємо їх ratio (видаляємо обережніше).
#         Конкретно: ∂obj/∂r_l ≈ -S_l (sensitivity = цінність шару).
#      c) Gradient step: r_l ← r_l + α * S_l (більше зберігаємо sensitive шари).
#      d) Projection на Δ: Σ_l r_l * n_l = (1-sp) * Σ_l n_l, r_l ∈ [0, 1].
#   3. Застосувати per-layer magnitude pruning із фінальними r_l.
#
# Примітка: це faithful спрощення (без ADMM), що зберігає ключову ідею DSA —
# диференційовне визначення розподілу через градієнти чутливості шарів.

import torch
import torch.nn as nn
import numpy as np
from methods import register
from or08_01 import (create_model, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity)


def _compute_taylor_saliency(model, train_dl, n_batches=3):
    """Taylor saliency Σ |w * ∂L/∂w| per layer, averaged over batches."""
    model.eval()
    layers = model.get_prunable_layers()
    L = len(layers)
    accum = [torch.zeros(1) for _ in range(L)]

    dl_iter = iter(train_dl)
    for b in range(n_batches):
        try:
            X, y = next(dl_iter)
        except StopIteration:
            dl_iter = iter(train_dl)
            X, y = next(dl_iter)

        for _, layer, _ in layers:
            layer.weight.requires_grad_(True)
        model.zero_grad()

        out = model(X)
        loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()

        with torch.no_grad():
            for i, (_, layer, _) in enumerate(layers):
                G = layer.weight.grad if layer.weight.grad is not None \
                    else torch.zeros_like(layer.weight)
                accum[i] += (layer.weight * G).abs().sum()
        model.zero_grad()

    # Повертаємо scalar sensitivity per layer (sum |w*g|), нормовану між ітераціями
    return [a.item() / n_batches for a in accum]


def _project_to_budget(ratios, n_params, target_params, n_iters=200):
    """
    Проєкція на Δ = {r ≥ 0 : Σ r_l * n_l = target_params, r_l ≤ 1}.
    Метод: лагранжевий λ-бісекції (Duchi et al. 2008).
    """
    r = np.array(ratios, dtype=np.float64)
    n = np.array(n_params, dtype=np.float64)
    total = target_params

    lo, hi = -10.0, 10.0
    for _ in range(n_iters):
        mid = (lo + hi) / 2.0
        proj = np.clip(r + mid, 0.0, 1.0)
        if np.dot(proj, n) < total:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    return np.clip(r + lam, 0.0, 1.0).tolist()


@register('dsa', 'DSA', '#9467bd')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()

    layers = model.get_prunable_layers()
    L = len(layers)
    n_params = [l.weight.data.numel() for _, l, _ in layers]
    total_params = sum(n_params)
    target_params = (1.0 - sp) * total_params

    # ── Крок 1: Taylor sensitivity per layer ──
    sensitivity = _compute_taylor_saliency(model, train_dl, n_batches=3)

    # ── Крок 2: Projected Gradient Ascent на keep-ratios ──
    # Ініціалізація: рівномірний розподіл
    ratios = [(1.0 - sp)] * L

    # Нормалізуємо sensitivity для стабільності градієнтного кроку
    s_arr = np.array(sensitivity, dtype=np.float64)
    s_max = s_arr.max() + 1e-12
    s_norm = s_arr / s_max  # ∈ [0, 1]

    alpha = 0.05  # крок
    n_steps = config.get('dsa_steps', 20)

    for step in range(n_steps):
        # Gradient ascent: зберігаємо більше там, де вища sensitivity
        r_arr = np.array(ratios, dtype=np.float64)
        r_arr = r_arr + alpha * s_norm
        # Проєкція на бюджетний симплекс
        ratios = _project_to_budget(r_arr.tolist(), n_params, target_params)
        # Адаптивний крок (cosine decay)
        alpha = 0.05 * (1.0 + np.cos(np.pi * step / n_steps)) / 2.0

    # ── Крок 3: Per-layer magnitude pruning із фінальними ratios ──
    with torch.no_grad():
        for (name, layer, mask_name), kr in zip(layers, ratios):
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
