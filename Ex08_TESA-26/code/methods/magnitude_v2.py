# methods/magnitude_v2.py — Magnitude + ERK Layer Allocation
from methods import register
from or08_01 import (create_model, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity)
import torch, math


def _erk_densities(model, sparsity):
    """ERK: density_l = alpha * (fan_in + fan_out) / (fan_in * fan_out)."""
    layers = model.get_prunable_layers()
    raw = []
    counts = []
    for _, layer, _ in layers:
        w = layer.weight
        fan_in, fan_out = w.shape[1], w.shape[0]
        if w.dim() == 4:
            fan_in *= w.shape[2] * w.shape[3]
        raw.append((fan_in + fan_out) / (fan_in * fan_out))
        counts.append(w.numel())
    # Binary search for alpha
    lo, hi = 0.0, 1e6
    total = sum(counts)
    target = total * (1.0 - sparsity)
    for _ in range(64):
        alpha = (lo + hi) / 2
        active = sum(min(1.0, alpha * r) * n for r, n in zip(raw, counts))
        if active < target:
            lo = alpha
        else:
            hi = alpha
    alpha = (lo + hi) / 2
    return [min(1.0, alpha * r) for r in raw]


@register('magnitude-v2', 'Magnitude-ERK', '#aec7e8')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    densities = _erk_densities(model, sp)
    with torch.no_grad():
        for (_, layer, mask_name), d in zip(model.get_prunable_layers(), densities):
            w = layer.weight.abs()
            k = max(1, int(w.numel() * d))
            thresh = w.view(-1).topk(k).values[-1]
            getattr(model, mask_name).copy_((w >= thresh).float())
    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    train_finetune_micro(model, train_dl, config['finetune_batches'])
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
