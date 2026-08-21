# methods/set_v2.py — SET with TESA-26 init instead of magnitude init
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from methods import register
from methods.tesa26 import _compute_saliency, _masks_from_k, _apply_masks, _evaluate_fitness_static
from or08_01 import (create_model, evaluate_full, set_seed, check_mask_connectivity)


def _tesa_init(model, train_dl, sp, pop_size=15):
    """TESA-26 init: Taylor saliency + Micro-ES per-layer allocation."""
    saliency = _compute_saliency(model, train_dl)
    layer_counts = [s.numel() for s in saliency]
    total_params = sum(layer_counts)
    target_params = total_params * (1.0 - sp)
    L = len(layer_counts)
    mu = np.full(L, 1.0 - sp)
    sigma = 0.05
    for _ in range(10):
        pop = np.clip(np.random.normal(mu, sigma, (pop_size, L)), 0.01, 1.0)
        fits = np.array([_evaluate_fitness_static(ind, saliency, layer_counts, target_params) for ind in pop])
        elite_idx = np.argsort(fits)[-3:]
        mu = np.mean(pop[elite_idx], axis=0)
        sigma = max(0.01, sigma * 0.9)
    masks = _masks_from_k(saliency, mu)
    _apply_masks(model, masks)


@register('set-v2', 'SET-v2 (TESA init)', '#c49c94')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    # TESA-26 init instead of magnitude
    _tesa_init(model, train_dl, sp)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    # OneCycleLR finetune (no topology swap — just quality init + train)
    total_batches = 100
    opt_m = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    sched = optim.lr_scheduler.OneCycleLR(opt_m, max_lr=0.05, total_steps=total_batches)
    crit = nn.CrossEntropyLoss()
    model.train()
    it = iter(train_dl)
    for step in range(total_batches):
        try:
            X, y = next(it)
        except StopIteration:
            it = iter(train_dl); X, y = next(it)
        opt_m.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        with torch.no_grad():
            for _, layer, mn in model.get_prunable_layers():
                if layer.weight.grad is not None:
                    layer.weight.grad.mul_(getattr(model, mn))
        opt_m.step()
        sched.step()

    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
