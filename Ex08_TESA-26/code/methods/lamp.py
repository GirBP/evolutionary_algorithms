# methods/lamp.py — LAMP: Layer-Adaptive Magnitude-based Pruning
#
# Джерело: Lee et al. "Layer-adaptive sparsity for the Magnitude-based
#           Pruning" (ICLR 2021). https://arxiv.org/abs/2010.07611
#
# Алгоритм:
#   Для кожної ваги w_i в шарі l обчислюється LAMP-score:
#       s_i = w_i^2 / Σ_{j ∈ layer l} w_j^2
#   після чого застосовується глобальний поріг по всіх шарах.
#   Це природньо дає адаптивний розподіл розрідженості між шарами:
#   шари з меншою сумарною "потужністю" ваг отримують нижчий поріг.
#
# Відмінність від Magnitude: LAMP нормує кожен вектор ваги per-layer,
# що захищає малі шари від надмірного проріджування.

import torch
from methods import register
from or08_01 import (create_model, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity)


@register('lamp', 'LAMP', '#ff7f0e')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()

    layers = model.get_prunable_layers()

    with torch.no_grad():
        # ── Крок 1: обчислити LAMP-score per weight ──
        all_scores = []
        all_meta = []

        for name, layer, mask_name in layers:
            W = layer.weight.data  # shape: [out, in] або [out, in, kH, kW]

            # Нормуємо квадрат ваги на суму квадратів у шарі (LAMP-score)
            w_sq = W ** 2
            layer_sum = w_sq.sum() + 1e-12
            lamp_score = w_sq / layer_sum  # того ж shape, що й W

            all_scores.append(lamp_score.view(-1))
            all_meta.append((layer, mask_name, lamp_score))

        # ── Крок 2: глобальний поріг по LAMP-score ──
        all_flat = torch.cat(all_scores)
        n_keep = max(1, int(all_flat.numel() * (1.0 - sp)))
        threshold = torch.topk(all_flat, n_keep).values[-1]

        # ── Крок 3: застосувати маски ──
        for layer, mask_name, lamp_score in all_meta:
            mask = (lamp_score >= threshold).float()
            getattr(model, mask_name).copy_(mask)

    # Топологічна перевірка
    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    # Мікро-файнтюнінг
    train_finetune_micro(model, train_dl, config.get('finetune_batches', 25))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
