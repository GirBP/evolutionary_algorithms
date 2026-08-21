# methods/ehta_snr.py — E-HTA-SNR: Score Normalization + Residual Boost
#
# Extension of E-HTA for residual architectures.
#
# Three fixes over naive Block Flow Penalty:
#   1. Per-layer Z-normalization of scores → prevents FC1 budget starvation
#   2. Residual skip boost γ in scores → preventive, not reactive
#   3. Channel-level hard rescue → guarantees no dead channels
#
# CMA-ES optimizes [λ_1,...,λ_L, γ_1,...,γ_B]
#   λ_l : Hessian balance per layer (same as E-HTA)
#   γ_b : skip path boost per residual block with projection
#
import torch
import torch.nn as nn
import numpy as np
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)


def _detect_residual_blocks(model):
    """
    Auto-detect residual block structure.
    Returns [{main: [idx], skip: [idx], type: str}].
    """
    blocks = []
    layers = model.get_prunable_layers()
    names = [n for n, _, _ in layers]

    # ResBlock 1: conv2+conv3, identity skip (no prunable skip layer)
    if 'c2' in names and 'c3' in names:
        blocks.append({
            'main': [names.index('c2'), names.index('c3')],
            'skip': [],
            'type': 'identity'
        })

    # ResBlock 2: conv4+conv5, projection skip (conv_skip)
    if 'c4' in names and 'c5' in names and 'cs' in names:
        blocks.append({
            'main': [names.index('c4'), names.index('c5')],
            'skip': [names.index('cs')],
            'type': 'projection'
        })

    return blocks


def _channel_rescue(masks, scores_raw, layers, blocks):
    """
    Hard constraint: ensure no output channel is fully dead in residual layers.
    For each dead channel, restore the weight with highest raw Ω.
    Returns number of rescued weights.
    """
    rescued = 0
    # Collect all layer indices in residual blocks
    res_indices = set()
    for b in blocks:
        res_indices.update(b['main'])
        res_indices.update(b['skip'])

    for idx in res_indices:
        _, layer, _ = layers[idx]
        m = masks[idx]
        W = layer.weight.data

        if W.dim() < 2:
            continue

        C_out = W.shape[0]
        for j in range(C_out):
            channel_mask = m[j]  # shape: [C_in, ...] or scalar
            if channel_mask.sum() == 0:
                # Channel j is dead — find highest-scoring weight to restore
                channel_scores = scores_raw[idx].view(W.shape)[j]
                flat_idx = channel_scores.argmax()
                # Unflatten index
                channel_mask.view(-1)[flat_idx] = 1.0
                rescued += 1

    return rescued


@register('ehta-snr', 'E-HTA-SNR', '#2ca02c')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    """
    E-HTA with Score Normalization + Residual Boost.

    Algorithm:
    1. Compute Taylor bases: t1 = |g·w|, t2 = w²
    2. CMA-ES optimizes [λ_1,...,λ_L, γ_1,...,γ_B]:
       a. Raw scores: Ω_i = t1_i + λ_l · t2_i
       b. Z-normalize per layer: Ω̃_i = (Ω_i - μ_l) / (σ_l + ε)
       c. Skip boost: Ω̃_i += γ_b for weights in skip paths
       d. Global threshold → masks
       e. Channel rescue (hard)
       f. Fitness = CE_loss(masked model)
    3. Apply best masks, finetune, evaluate
    """
    import cma
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    layers = model.get_prunable_layers()
    L = len(layers)
    criterion = nn.CrossEntropyLoss()

    # ── Detect residual structure ──
    blocks = _detect_residual_blocks(model)
    proj_blocks = [b for b in blocks if b['type'] == 'projection']
    B = len(proj_blocks)

    # Build skip-layer → block mapping
    skip_layer_to_block = {}
    for bi, b in enumerate(proj_blocks):
        for idx in b['skip']:
            skip_layer_to_block[idx] = bi

    # ── Step 1: Taylor bases ──
    model.eval()
    for _, layer, _ in layers:
        layer.weight.requires_grad_(True)
    model.zero_grad()

    X_cal, y_cal = next(iter(train_dl))
    out = model(X_cal)
    criterion(out, y_cal).backward()

    t1_list, t2_list = [], []
    layer_sizes = []
    with torch.no_grad():
        for _, layer, _ in layers:
            W = layer.weight.data.view(-1)
            G = layer.weight.grad.view(-1) if layer.weight.grad is not None else torch.zeros_like(W)
            t1_list.append(torch.abs(G * W))
            t2_list.append(W ** 2)
            layer_sizes.append(W.numel())
    model.zero_grad()

    total_params = sum(layer_sizes)
    target_keep = max(1, int(total_params * (1.0 - sp)))

    # ── Step 2: CMA-ES over [λ_1,...,λ_L, γ_1,...,γ_B] ──
    n_dims = L + B
    x0 = np.zeros(n_dims)
    if B > 0:
        x0[L:] = 1.0  # initial γ = 1.0 (moderate skip boost)

    sigma0 = 0.5
    pop_size = config.get('pop_size', 8)
    max_evals = config.get('max_evals', 40)

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': pop_size,
        'verbose': -1,
        'bounds': [0.0, None],
    })

    def _evaluate(sol):
        lam_vec = sol[:L]
        gamma_vec = sol[L:L + B] if B > 0 else []

        with torch.no_grad():
            # (a) Raw scores
            raw_scores = []
            for i in range(L):
                score = t1_list[i] + float(max(0, lam_vec[i])) * t2_list[i]
                raw_scores.append(score)

            # (b) Z-normalize per layer
            norm_scores = []
            for i in range(L):
                s = raw_scores[i]
                mu = s.mean()
                sigma = s.std() + 1e-8
                norm_scores.append((s - mu) / sigma)

            # (c) Skip boost
            for layer_idx, block_idx in skip_layer_to_block.items():
                gamma = float(max(0, gamma_vec[block_idx])) if block_idx < len(gamma_vec) else 0
                norm_scores[layer_idx] = norm_scores[layer_idx] + gamma

            # (d) Global threshold
            all_scores = torch.cat(norm_scores)
            if total_params - target_keep >= all_scores.numel():
                threshold = all_scores.max() + 1
            else:
                threshold = torch.kthvalue(all_scores, total_params - target_keep).values

            offset = 0
            mask_list = []
            for i, (_, layer, mask_name) in enumerate(layers):
                n = layer_sizes[i]
                layer_norm_scores = norm_scores[i]
                mask = (layer_norm_scores >= threshold).float().view_as(layer.weight)
                mask_list.append(mask)
                getattr(model, mask_name).copy_(mask)
                offset += n

            # (e) Channel rescue
            _channel_rescue(mask_list, raw_scores, layers, blocks)
            # Re-apply rescued masks
            for i, (_, _, mask_name) in enumerate(layers):
                getattr(model, mask_name).copy_(mask_list[i])

            # (f) Fitness = CE loss
            out = model(X_cal)
            loss = criterion(out, y_cal).item()

            # Restore masks
            for _, _, mn in layers:
                getattr(model, mn).fill_(1.0)

        return loss

    while not es.stop() and es.countevals < max_evals:
        solutions = es.ask()
        fitnesses = [_evaluate(s) for s in solutions]
        es.tell(solutions, fitnesses)

    # ── Step 3: Apply best ──
    best_sol = es.result.xbest
    best_lam = best_sol[:L]
    best_gamma = best_sol[L:L + B] if B > 0 else []

    with torch.no_grad():
        raw_scores = []
        for i in range(L):
            raw_scores.append(t1_list[i] + float(max(0, best_lam[i])) * t2_list[i])

        norm_scores = []
        for i in range(L):
            s = raw_scores[i]
            norm_scores.append((s - s.mean()) / (s.std() + 1e-8))

        for layer_idx, block_idx in skip_layer_to_block.items():
            gamma = float(max(0, best_gamma[block_idx])) if block_idx < len(best_gamma) else 0
            norm_scores[layer_idx] = norm_scores[layer_idx] + gamma

        all_scores = torch.cat(norm_scores)
        if total_params - target_keep >= all_scores.numel():
            threshold = all_scores.max() + 1
        else:
            threshold = torch.kthvalue(all_scores, total_params - target_keep).values

        mask_list = []
        for i, (_, layer, mask_name) in enumerate(layers):
            mask = (norm_scores[i] >= threshold).float().view_as(layer.weight)
            mask_list.append(mask)
            getattr(model, mask_name).copy_(mask)

        rescued = _channel_rescue(mask_list, raw_scores, layers, blocks)
        for i, (_, _, mask_name) in enumerate(layers):
            getattr(model, mask_name).copy_(mask_list[i])

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
