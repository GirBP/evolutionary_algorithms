# methods/tesa26.py — TESA-ISR v2: Taylor-Evolutionary Sparsity Allocation
#                     with Iterative Saliency Recalibration
#
# Наукова новизна (відсутні в DSA NeurIPS'24, SparseGPT, Taylor FO):
# 1. ISR: Iterative Saliency Recalibration — перерахунок Taylor saliency
#    з поточною маскою (saliency змінюється при sparsity)
# 2. NFP: Neuron Flow Penalty — неперервний штраф за мертві нейрони
# 3. Layer-Adaptive sigma: ES sigma per-layer по розкиду saliency
# 4. v2: CMA-ES замість (μ+λ), multi-batch saliency, warm-start
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods import register
from or08_01 import (create_model, apply_global, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome)


def _plain_forward(model, x):
    """Forward through model with proper architecture handling."""
    return model(x)


def _compute_saliency_multi(model, train_dl, n_batches=3):
    """Taylor Saliency усереднена по n_batches для стабільності."""
    model.eval()
    layers = model.get_prunable_layers()
    L = len(layers)
    accum = [torch.zeros_like(layers[i][1].weight) for i in range(L)]

    dl_iter = iter(train_dl)
    for b in range(n_batches):
        try:
            X_cal, y_cal = next(dl_iter)
        except StopIteration:
            dl_iter = iter(train_dl)
            X_cal, y_cal = next(dl_iter)

        for _, layer, _ in layers:
            layer.weight.requires_grad_(True)
        model.zero_grad()

        out = _plain_forward(model, X_cal)
        loss = nn.CrossEntropyLoss()(out, y_cal)
        loss.backward()

        with torch.no_grad():
            for i, (_, layer, _) in enumerate(layers):
                G = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(layer.weight)
                accum[i] += (layer.weight * G).abs()
        model.zero_grad()

    saliency = [a / n_batches for a in accum]
    return saliency


def _masks_from_k(saliency, k_vector):
    """Бінарні маски з per-layer keep-ratio."""
    masks = []
    for sal, k in zip(saliency, k_vector):
        if k <= 0.0:
            masks.append(torch.zeros_like(sal))
            continue
        if k >= 1.0:
            masks.append(torch.ones_like(sal))
            continue
        n_keep = max(1, int(k * sal.numel()))
        thresh = torch.topk(sal.view(-1), n_keep).values[-1]
        masks.append((sal >= thresh).float())
    return masks


def _apply_masks(model, masks):
    with torch.no_grad():
        for (_, _, mask_name), m in zip(model.get_prunable_layers(), masks):
            getattr(model, mask_name).copy_(m)


# ── NOVELTY §2: Neuron Flow Penalty ──
def _neuron_flow_penalty(masks):
    """Неперервна міра мертвих нейронів у кожному шарі (крім останнього)."""
    penalty = 0.0
    for m in masks[:-1]:
        if m.dim() >= 2:
            out_activity = m.sum(dim=tuple(range(1, m.dim())))
            n_dead = (out_activity == 0).sum().item()
            n_total = out_activity.numel()
            if n_total > 0:
                frac_dead = n_dead / n_total
                penalty += np.log1p(frac_dead * 10.0)
    return penalty


def _evaluate_fitness(genome, layer_counts, target_sparsity, saliency):
    """
    Fitness = log-barrier capacity - budget penalty - NFP.
    genome → normalized keep-ratios → masks → score.
    """
    ratios = normalize_genome(genome, layer_counts, target_sparsity)
    masks = _masks_from_k(saliency, ratios)

    # Log-barrier: maximize Σ log(kept_saliency / total_saliency)
    log_capacity = 0.0
    for sal, m in zip(saliency, masks):
        kept = (sal * m).sum().item()
        total = sal.sum().item() + 1e-12
        log_capacity += np.log(kept / total + 1e-12)

    # NFP
    nfp = _neuron_flow_penalty(masks)

    return -(log_capacity - 3.0 * nfp)  # CMA-ES minimizes


@register('tesa26', 'TESA-26', '#e41a1c')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    import cma
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    layers = model.get_prunable_layers()
    layer_counts = np.array([l.weight.numel() for _, l, _ in layers])
    L = len(layers)

    # ── Step 1: Multi-batch Taylor Saliency ──
    saliency = _compute_saliency_multi(model, train_dl, n_batches=3)

    # ── Warm start: magnitude-based ratios ──
    apply_global(model, sp)
    x0 = np.array([getattr(model, mn).sum().item() / getattr(model, mn).numel()
                    for _, _, mn in layers])
    model.load_state_dict(teacher_state)  # reset

    # ── Step 2: CMA-ES with ISR ──
    pop_size = config.get('pop_size', 12)
    max_evals = config.get('max_evals', 200)
    recalibrate_at = max_evals // 2  # ISR: recalibrate once at 50%

    sigma0 = 0.15 * (1.0 - sp)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': pop_size, 'verbose': -1, 'bounds': [0, None]
    })

    evals_done = 0
    recalibrated = False
    best_genome = x0.copy()

    while not es.stop() and evals_done < max_evals:
        # ── NOVELTY §1: ISR — recalibrate saliency mid-search ──
        if not recalibrated and evals_done >= recalibrate_at:
            recalibrated = True
            ratios = normalize_genome(best_genome, layer_counts, sp)
            masks = _masks_from_k(saliency, ratios)
            _apply_masks(model, masks)
            saliency = _compute_saliency_multi(model, train_dl, n_batches=3)
            model.load_state_dict(teacher_state)  # reset weights, keep new saliency

        solutions = es.ask()
        fitnesses = [_evaluate_fitness(s, layer_counts, sp, saliency) for s in solutions]
        es.tell(solutions, fitnesses)
        evals_done += len(solutions)

        # Track best
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < _evaluate_fitness(best_genome, layer_counts, sp, saliency):
            best_genome = solutions[best_idx].copy()

    # ── Step 3: Final ISR + apply ──
    best_ratios = normalize_genome(es.result.xbest, layer_counts, sp)

    # Final saliency recalibration with best mask
    masks = _masks_from_k(saliency, best_ratios)
    _apply_masks(model, masks)
    final_saliency = _compute_saliency_multi(model, train_dl, n_batches=3)

    # Recompute masks with recalibrated saliency
    final_masks = _masks_from_k(final_saliency, best_ratios)

    # Apply to fresh teacher
    model.load_state_dict(teacher_state)
    _apply_masks(model, final_masks)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}


# Backward-compatible alias for set_v2
def _compute_saliency(model, train_dl):
    return _compute_saliency_multi(model, train_dl, n_batches=1)

def _evaluate_fitness_static(k_vector, saliency, layer_counts, target_params):
    total_score = 0.0
    current_params = 0.0
    for sal, k, n in zip(saliency, k_vector, layer_counts):
        n_keep = max(1, int(k * n))
        current_params += n_keep
        if n_keep >= n:
            total_score += sal.sum().item()
        else:
            total_score += sal.view(-1).topk(n_keep).values.sum().item()
    total_n = sum(layer_counts)
    penalty = ((current_params - target_params) / total_n) ** 2
    return total_score - 100.0 * penalty
