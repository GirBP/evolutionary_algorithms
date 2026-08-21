# methods/evo_synflow_ex07.py — Evo-SynFlow adapted directly from Ex07 (iterative SynFlow)
# Key difference: multi-iteration SynFlow scoring with importance-weighted aggregation
import torch
import numpy as np
from methods import register
from or08_01 import (create_model, apply_ratios, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome, apply_global)


def _get_synflow_score_iterative(model, n_iterations=3):
    """Ex07-style iterative SynFlow: |w * grad(w)| summed over multiple passes."""
    model.eval()
    eps = 1e-8
    all_layer_scores = []

    for _ in range(n_iterations):
        model.zero_grad()
        layers = model.get_prunable_layers()
        first_l = layers[0][1]
        if first_l.weight.dim() == 2 and first_l.weight.shape[1] <= 10:
            x = torch.ones(1, first_l.weight.shape[1])
        else:
            x = torch.ones(1, 1, 28, 28)
        try:
            out = model(x)
            out.sum().backward()
        except RuntimeError:
            break

        layer_scores = []
        with torch.no_grad():
            for i, (_, layer, m_name) in enumerate(layers):
                if layer.weight.grad is not None:
                    mask = getattr(model, m_name)
                    n_active = mask.sum().item()
                    if n_active == 0:
                        continue
                    term = (layer.weight * layer.weight.grad * mask).abs()
                    ls = term.sum().item()
                    li = ls / (n_active + eps)
                    layer_scores.append((i, li, ls))
        if layer_scores:
            all_layer_scores.append(layer_scores)

    if not all_layer_scores:
        return eps

    agg = {}
    for it_scores in all_layer_scores:
        for idx, imp, sc in it_scores:
            if idx not in agg:
                agg[idx] = {'imp': [], 'sc': []}
            agg[idx]['imp'].append(imp)
            agg[idx]['sc'].append(sc)

    total = 0.0
    for data in agg.values():
        avg_sc = np.mean(data['sc'])
        avg_imp = np.mean(data['imp'])
        total += avg_sc * (1.0 + avg_imp * 0.1)
    return max(total, eps)


def _synflow_worker(genome, teacher_state, layer_counts, target_sparsity):
    """Evaluate SynFlow score for one genome."""
    ratios = normalize_genome(genome, layer_counts, target_sparsity)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_ratios(model, ratios)
    n_iter = 5 if target_sparsity >= 0.90 else 3
    score = _get_synflow_score_iterative(model, n_iterations=n_iter)
    return -score


@register('evo-synflow-ex07', 'Evo-SynFlow (Ex07)', '#ff7f0e')
def run(teacher_state, sp, seed, config, train_dl, test_dl, *, _state=None):
    import cma
    set_seed(seed)

    temp = create_model()
    layer_counts = np.array([l.weight.numel() for _, l, _ in temp.get_prunable_layers()])

    # Warm start from magnitude ratios
    prv = _state.get('prev_ratios') if _state else None
    if prv is not None:
        x0 = prv.copy()
        scale = (1.0 - sp) / (1.0 - (1.0 - np.sum(x0 * layer_counts) / np.sum(layer_counts)))
        x0 = np.clip(x0 * scale, 0.01, 1.0)
    else:
        temp.load_state_dict(teacher_state)
        apply_global(temp, sp)
        x0 = np.array([getattr(temp, mn).sum().item() / getattr(temp, mn).numel()
                        for _, _, mn in temp.get_prunable_layers()])

    # Ex07-style sigma and pop settings
    pop_size = config.get('pop_size', 12)
    max_evals = config.get('max_evals', 200)
    sigma0 = 0.15 * (1.0 - sp)
    if sp >= 0.90:
        pop_size = int(pop_size * 1.5)
        max_evals = int(max_evals * 1.5)
        sigma0 = min(sigma0, 0.12 * (1.0 - sp))

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': pop_size, 'verbose': -1, 'bounds': [0, None]})

    while not es.stop() and es.countevals < max_evals:
        sols = es.ask()
        es.tell(sols, [_synflow_worker(s, teacher_state, layer_counts, sp) for s in sols])

    best_ratios = normalize_genome(es.result.xbest, layer_counts, sp)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_ratios(model, best_ratios)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)

    if _state is not None:
        _state['prev_ratios'] = best_ratios
    return {'F1': f1}
