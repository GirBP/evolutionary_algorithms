# methods/set_method.py — SET: Sparse Evolutionary Training (Adaptive Budget)
#
# Adaptive batch budget formula:
#   B_total = B_warmup + n_swaps × B_recovery + B_cooldown
#   B_recovery = ceil(ζ · N_active / batch_size) × k_safety
#   n_swaps = min(max_swaps, N_active / (ζ · N_active + 1))
#
import math
import torch
import torch.nn as nn
import torch.optim as optim
from methods import register
from or08_01 import (create_model, apply_global, evaluate_full,
                     set_seed, check_mask_connectivity)


def compute_set_budget(n_params, sparsity, batch_size, zeta_base=0.10,
                       k_safety=3, max_swaps=5, warmup_epochs=0.5, cooldown_epochs=0.5,
                       n_data=5000):
    """
    Compute adaptive SET budget based on network capacity and sparsity.

    Returns: (total_batches, warmup_batches, update_every_k, zeta, n_swaps)

    Math:
      N_active = N × (1 - s)
      ζ = ζ_base × min(1, s/0.8)                     # adaptive zeta
      B_recovery = ceil(ζ × N_active / batch_size) × k # batches to recover from 1 swap
      B_warmup = ceil(n_data / batch_size × warmup_epochs)
      B_cooldown = ceil(n_data / batch_size × cooldown_epochs)
      n_swaps = min(max_swaps, floor(1 / (2ζ)))       # stability: can't swap > 50% total
      B_total = B_warmup + n_swaps × B_recovery + B_cooldown
    """
    n_active = int(n_params * (1 - sparsity))
    zeta = zeta_base * min(1.0, sparsity / 0.80)

    batches_per_epoch = math.ceil(n_data / batch_size)
    b_warmup = math.ceil(batches_per_epoch * warmup_epochs)
    b_cooldown = math.ceil(batches_per_epoch * cooldown_epochs)

    # Recovery batches: enough SGD steps to absorb ζ × N_active perturbation
    perturbed_params = max(1, int(zeta * n_active))
    b_recovery = math.ceil(perturbed_params / batch_size) * k_safety
    b_recovery = max(b_recovery, 5)  # minimum 5 batches between swaps

    # Limit swaps to prevent cumulative disruption
    n_swaps = min(max_swaps, max(1, int(1.0 / (2 * zeta + 1e-8))))

    total = b_warmup + n_swaps * b_recovery + b_cooldown
    update_every_k = b_recovery

    return total, b_warmup, update_every_k, zeta, n_swaps


def _train_set_adaptive(model, train_dl, total_batches, warmup_batches, update_every_k, zeta):
    """SET training with warmup → topology swaps → cooldown."""
    opt = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.05, total_steps=total_batches)
    crit = nn.CrossEntropyLoss()
    model.train()
    iterator = iter(train_dl)

    for step in range(total_batches):
        try:
            X, y = next(iterator)
        except StopIteration:
            iterator = iter(train_dl)
            X, y = next(iterator)

        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()

        # Mask gradients
        with torch.no_grad():
            for _, layer, m_name in model.get_prunable_layers():
                if layer.weight.grad is not None:
                    layer.weight.grad.mul_(getattr(model, m_name))
        opt.step()
        scheduler.step()

        # Topology swap only during middle phase (after warmup, before cooldown)
        in_swap_phase = warmup_batches <= step < (total_batches - warmup_batches)
        if in_swap_phase and (step - warmup_batches + 1) % update_every_k == 0:
            with torch.no_grad():
                for _, layer, m_name in model.get_prunable_layers():
                    mask = getattr(model, m_name)
                    flat_m = mask.view(-1)
                    w_flat = layer.weight.abs().view(-1)
                    active_idx = (flat_m > 0).nonzero(as_tuple=True)[0]
                    n_active = len(active_idx)
                    n_drop = max(1, int(n_active * zeta))

                    if n_active <= n_drop + 1:
                        continue

                    # Drop smallest-magnitude
                    drop_local = w_flat[active_idx].argsort()[:n_drop]
                    drop_idx = active_idx[drop_local]
                    flat_m[drop_idx] = 0.0
                    layer.weight.view(-1)[drop_idx] = 0.0

                    # Regrow: gradient-informed
                    inactive_idx = (flat_m == 0).nonzero(as_tuple=True)[0]
                    n_grow = min(n_drop, len(inactive_idx))
                    if n_grow > 0 and layer.weight.grad is not None:
                        g_flat = layer.weight.grad.abs().view(-1)
                        top_k = g_flat[inactive_idx].topk(min(n_grow, len(inactive_idx))).indices
                        grow_idx = inactive_idx[top_k]
                        flat_m[grow_idx] = 1.0
                        layer.weight.view(-1)[grow_idx] = 0.001 * torch.sign(
                            layer.weight.grad.view(-1)[grow_idx])


@register('set', 'SET', '#9467bd')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_global(model, sp)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    n_params = sum(l.weight.numel() for _, l, _ in model.get_prunable_layers())
    batch_size = config.get('batch_size', 64)

    total, warmup, update_k, zeta, n_swaps = compute_set_budget(
        n_params, sp, batch_size)

    _train_set_adaptive(model, train_dl, total, warmup, update_k, zeta)
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
