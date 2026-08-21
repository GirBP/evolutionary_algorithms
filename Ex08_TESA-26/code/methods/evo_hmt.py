# methods/evo_hmt.py — Evo-HMT v2 with Deferred BN Recalibration (Multi-Fidelity)
# §5: BN recalibration moved OUT of MicroES grid search, applied only to final topology
import torch
import torch.nn as nn
import numpy as np
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity, apply_global)


def _compute_importance(model, train_dl):
    """Compute M_mag and M_grad importance matrices (once per seed)."""
    model.eval()
    model.zero_grad()
    layers = model.get_prunable_layers()
    imp = {}

    # M_mag = minmax(|W|)
    with torch.no_grad():
        for name, layer, _ in layers:
            w = layer.weight.abs()
            w_max = w.max()
            imp[name] = {'M_mag': (w / (w_max + 1e-8)).detach()}

    # M_grad = minmax(|W * G|) using REAL calibration data
    X_cal, y_cal = next(iter(train_dl))
    model.train()
    model.zero_grad()
    try:
        out = model(X_cal)
        nn.CrossEntropyLoss()(out, y_cal).backward()
    except RuntimeError:
        for name, _, _ in layers:
            imp[name]['M_grad'] = imp[name]['M_mag'].clone()
        return imp

    with torch.no_grad():
        for name, layer, _ in layers:
            if layer.weight.grad is not None:
                g = (layer.weight * layer.weight.grad).abs()
                g_max = g.max()
                imp[name]['M_grad'] = (g / (g_max + 1e-8)).detach()
            else:
                imp[name]['M_grad'] = imp[name]['M_mag'].clone()
    return imp


def _erk_ratios(model, sparsity, erk_power):
    """ERK layer allocation with binary search for scaling factor."""
    layers = model.get_prunable_layers()
    layer_counts = [l.weight.numel() for _, l, _ in layers]
    total = sum(layer_counts)
    target_active = total * (1.0 - sparsity)

    densities = []
    for _, layer, _ in layers:
        w = layer.weight
        if w.dim() == 4:
            d = (w.shape[0] + w.shape[1] + w.shape[2] + w.shape[3]) / w.numel()
        else:
            d = (w.shape[0] + w.shape[1]) / w.numel()
        densities.append(d)

    lo, hi = 0.0, 1e6
    for _ in range(64):
        alpha = (lo + hi) / 2
        ratios = [min(1.0, alpha * (d ** erk_power)) for d in densities]
        active = sum(r * n for r, n in zip(ratios, layer_counts))
        if active < target_active:
            lo = alpha
        else:
            hi = alpha

    alpha = (lo + hi) / 2
    return [min(1.0, alpha * (d ** erk_power)) for d in densities]


# MicroES grid (12 points)
GRID = [
    (1.0, 0.0, 1.0), (1.0, 0.0, 2.0), (0.5, 0.5, 1.0), (0.5, 0.5, 2.0),
    (0.0, 1.0, 1.0), (0.0, 1.0, 2.0), (0.8, 0.2, 1.5), (0.2, 0.8, 1.5),
    (1.0, 0.0, 0.5), (0.5, 0.5, 0.5), (0.8, 0.2, 0.8), (0.6, 0.4, 1.2),
    # Extended grid for high sparsity
    (0.7, 0.3, 1.0), (0.3, 0.7, 1.0), (0.9, 0.1, 1.0), (0.1, 0.9, 1.0),
    (0.7, 0.3, 2.0), (0.3, 0.7, 2.0), (0.5, 0.5, 3.0), (1.0, 0.0, 3.0),
    (0.4, 0.6, 0.8), (0.6, 0.4, 0.5), (0.8, 0.2, 2.5), (0.5, 0.5, 1.5),
]


def _evaluate_genome_static(w_mag, w_grad, p_erk, model_template, imp, sparsity, layer_counts):
    """§5 Deferred BN: evaluate genome WITHOUT BN recalibration (pure scoring)."""
    ratios = _erk_ratios(model_template, sparsity, p_erk) if p_erk > 0 else [1.0 - sparsity] * len(layer_counts)
    total_score = 0.0
    for i, (name, layer, _) in enumerate(model_template.get_prunable_layers()):
        S = w_mag * imp[name]['M_mag'] + w_grad * imp[name]['M_grad']
        k = max(1, int(layer_counts[i] * ratios[i]))
        if k >= layer_counts[i]:
            total_score += S.sum().item()
        else:
            total_score += S.view(-1).topk(k).values.sum().item()
    return total_score


def _apply_best_mask(model, imp, w_mag, w_grad, ratios):
    """Apply composite scoring mask to model."""
    with torch.no_grad():
        for i, (name, layer, mask_name) in enumerate(model.get_prunable_layers()):
            S = w_mag * imp[name]['M_mag'] + w_grad * imp[name]['M_grad']
            k = max(1, int(layer.weight.numel() * ratios[i]))
            score_flat = S.view(-1)
            if k >= score_flat.numel():
                mask = torch.ones_like(score_flat)
            else:
                thresh = score_flat.topk(k).values[-1]
                mask = (score_flat >= thresh).float()
                # Tie-breaking: ensure exactly k
                if mask.sum().item() > k:
                    excess = int(mask.sum().item()) - k
                    tie_mask = (score_flat == thresh)
                    tie_indices = tie_mask.nonzero(as_tuple=True)[0]
                    for idx in tie_indices[:excess]:
                        mask[idx] = 0.0
            getattr(model, mask_name).copy_(mask.view_as(layer.weight))


def _run_hmt(teacher_state, sp, seed, config, train_dl, test_dl,
             use_erk=True, use_bn=True, use_grad=True):
    set_seed(seed)
    model_template = create_model()
    model_template.load_state_dict(teacher_state)
    imp = _compute_importance(model_template, train_dl)
    layer_counts = [l.weight.numel() for _, l, _ in model_template.get_prunable_layers()]

    # MicroES grid search + local perturbation
    scored = []
    for w_mag, w_grad, p_erk in GRID:
        if not use_grad:
            w_grad = 0.0
            w_mag = 1.0
        if not use_erk:
            p_erk = 0.0
        score = _evaluate_genome_static(w_mag, w_grad, p_erk, model_template, imp, sp, layer_counts)
        scored.append((score, (w_mag, w_grad, p_erk)))

    # Local perturbation of top-3
    scored.sort(key=lambda x: x[0], reverse=True)
    import random
    random.seed(seed)
    for s_val, (wm, wg, pe) in scored[:3]:
        for _ in range(4):
            wm2 = max(0, min(1, wm + random.uniform(-0.1, 0.1)))
            wg2 = max(0, min(1, wg + random.uniform(-0.1, 0.1)))
            pe2 = max(0.1, pe + random.uniform(-0.3, 0.3))
            # Renormalize
            ws = wm2 + wg2 + 1e-8
            wm2, wg2 = wm2/ws, wg2/ws
            if not use_grad:
                wg2 = 0.0; wm2 = 1.0
            if not use_erk:
                pe2 = 0.0
            sc = _evaluate_genome_static(wm2, wg2, pe2, model_template, imp, sp, layer_counts)
            scored.append((sc, (wm2, wg2, pe2)))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_genome = scored[0]

    # Apply best genome to fresh model
    w_mag, w_grad, p_erk = best_genome
    if not use_grad:
        w_grad = 0.0; w_mag = 1.0
    ratios = _erk_ratios(model_template, sp, p_erk) if use_erk else [1.0 - sp] * len(layer_counts)

    model = create_model()
    model.load_state_dict(teacher_state)
    _apply_best_mask(model, imp, w_mag, w_grad, ratios)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    # §5 Deferred BN Recalibration: only on final topology
    if use_bn:
        bn_layers = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
        if bn_layers:
            model.train()
            with torch.no_grad():
                for X, _ in train_dl:
                    model(X)
                    break  # 1 batch calibration

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}


@register('evo-hmt-no-erk', 'Evo-HMT (−ERK)', '#aec7e8')
def run_no_erk(teacher_state, sp, seed, config, train_dl, test_dl):
    return _run_hmt(teacher_state, sp, seed, config, train_dl, test_dl, use_erk=False)

@register('evo-hmt-no-bn', 'Evo-HMT (−BN)', '#ffbb78')
def run_no_bn(teacher_state, sp, seed, config, train_dl, test_dl):
    return _run_hmt(teacher_state, sp, seed, config, train_dl, test_dl, use_bn=False)
