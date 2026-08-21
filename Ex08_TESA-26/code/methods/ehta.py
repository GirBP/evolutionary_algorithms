# methods/ehta.py — E-HTA wrapper for Ex08 SimpleMLP
# E-HTA: Evolutionary Hessian-Trace Approximation
# Core: e_hta.py (global pruning via CMA-ES optimized λ for score = |g*w| + λ*w²)
# Related: GraSP (Wang et al., ICLR 2020) — Hessian-gradient product, fixed formula
# Novelty: E-HTA evolves λ balance via CMA-ES instead of fixed scoring
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)


def _plain_forward(model, x):
    """Forward through model with proper architecture handling (skip connections, BN, etc)."""
    return model(x)


@register('ehta', 'E-HTA (Hessian)', '#8c564b')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    """
    Evolutionary Hessian-Trace Approximation (E-HTA).
    CMA-ES optimizes per-layer λ_l in: score = |g*w| + λ_l * w²
    Global pruning: no per-layer quotas — emergent layer allocation.
    """
    import cma
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    layers = model.get_prunable_layers()
    L = len(layers)
    criterion = nn.CrossEntropyLoss()

    # ── Step 1: Compute Taylor bases (one forward+backward) ──
    model.eval()
    for _, layer, _ in layers:
        layer.weight.requires_grad_(True)
    model.zero_grad()

    X_cal, y_cal = next(iter(train_dl))
    out = _plain_forward(model, X_cal)
    criterion(out, y_cal).backward()

    t1_list, t2_list = [], []
    layer_offsets = []
    offset = 0
    with torch.no_grad():
        for _, layer, _ in layers:
            W = layer.weight.data.view(-1)
            G = layer.weight.grad.view(-1) if layer.weight.grad is not None else torch.zeros_like(W)
            t1 = torch.abs(G * W)       # 1st order: |g*w| (Taylor 2019)
            t2 = W ** 2                  # 2nd order: w² (magnitude squared)
            t1_list.append(t1)
            t2_list.append(t2)
            layer_offsets.append((offset, offset + W.numel()))
            offset += W.numel()
    model.zero_grad()

    total_params = offset
    target_keep = max(1, int(total_params * (1.0 - sp)))

    # Save original weights for rollback
    orig_weights = {}
    with torch.no_grad():
        for _, layer, _ in layers:
            orig_weights[id(layer)] = layer.weight.data.clone()

    # ── Step 2: CMA-ES over λ vector (L dimensions) ──
    x0 = np.zeros(L)  # start from pure Taylor (λ=0)
    sigma0 = 0.5
    pop_size = config.get('pop_size', 8)
    max_evals = config.get('max_evals', 40)

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': pop_size,
        'verbose': -1,
        'bounds': [0.0, None],  # λ ≥ 0 (curvature is non-negative)
    })

    while not es.stop() and es.countevals < max_evals:
        solutions = es.ask()
        fitnesses = []

        for lam_vec in solutions:
            with torch.no_grad():
                # Build global score: Ω_i = |g_i * w_i| + λ_l * w_i²
                global_scores = []
                for i in range(L):
                    score = t1_list[i] + float(max(0, lam_vec[i])) * t2_list[i]
                    global_scores.append(score)

                all_scores = torch.cat(global_scores)
                threshold = torch.kthvalue(all_scores, total_params - target_keep).values

                # Apply mask and evaluate loss
                for i, (_, layer, mask_name) in enumerate(layers):
                    s, e = layer_offsets[i]
                    mask = (global_scores[i] >= threshold).float().view_as(layer.weight)
                    getattr(model, mask_name).copy_(mask)

                out = _plain_forward(model, X_cal)
                loss = criterion(out, y_cal).item()
                fitnesses.append(loss)

                # Restore masks
                for _, _, mn in layers:
                    getattr(model, mn).fill_(1.0)

        es.tell(solutions, fitnesses)

    # ── Step 3: Apply best λ ──
    best_lam = es.result.xbest
    with torch.no_grad():
        global_scores = []
        for i in range(L):
            global_scores.append(t1_list[i] + float(max(0, best_lam[i])) * t2_list[i])

        all_scores = torch.cat(global_scores)
        threshold = torch.kthvalue(all_scores, total_params - target_keep).values

        for i, (_, layer, mask_name) in enumerate(layers):
            mask = (global_scores[i] >= threshold).float().view_as(layer.weight)
            getattr(model, mask_name).copy_(mask)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    # Finetuning
    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
