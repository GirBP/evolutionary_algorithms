# methods/evo_synflow_adaptive.py — Evo-SynFlow (Adaptive) with static proxy
import torch
import numpy as np
from methods import register
from or08_01 import (create_model, apply_ratios, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome, apply_global)


def _precompute_static_proxy_pz(teacher_state):
    """Pruner-Zero static proxy: |W|^2 * σ(|G|)."""
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()
    model.zero_grad()
    layers = model.get_prunable_layers()
    first_layer = layers[0][1]
    if first_layer.weight.dim() == 2 and first_layer.weight.shape[1] <= 10:
        x = torch.ones(1, first_layer.weight.shape[1])
    else:
        x = torch.ones(1, 1, 28, 28)
    out = model(x)
    out.sum().backward()
    scores = {}
    layer_counts = []
    with torch.no_grad():
        for i, (_, layer, _) in enumerate(layers):
            W = layer.weight
            G = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(W)
            scores[i] = ((W.abs() ** 2) * torch.sigmoid(G.abs())).view(-1).numpy()
            layer_counts.append(W.numel())
    return scores, np.array(layer_counts)


def _static_eval(genome, layer_counts, target_sparsity, static_scores):
    ratios = normalize_genome(genome, layer_counts, target_sparsity)
    total = 0.0
    for i, (ratio, n) in enumerate(zip(ratios, layer_counts)):
        k = max(1, int(n * ratio))
        sc = static_scores[i]
        total += np.partition(sc, -min(k, len(sc)))[-min(k, len(sc)):].sum() if k < len(sc) else sc.sum()
    return -total


@register('evo-synflow-adaptive', 'Evo-SynFlow (Adaptive)', '#8c564b')
def run(teacher_state, sp, seed, config, train_dl, test_dl, *, _state=None):
    import cma
    set_seed(seed)
    static_scores, layer_counts = _precompute_static_proxy_pz(teacher_state)

    prv = _state.get('prev_ratios') if _state else None
    if prv is not None:
        x0 = prv.copy()
        scale = (1.0 - sp) / (1.0 - (1.0 - np.sum(x0 * layer_counts) / np.sum(layer_counts)))
        x0 = np.clip(x0 * scale, 0.01, 1.0)
    else:
        temp = create_model()
        temp.load_state_dict(teacher_state)
        apply_global(temp, sp)
        x0 = np.array([getattr(temp, mn).sum().item() / getattr(temp, mn).numel()
                        for _, _, mn in temp.get_prunable_layers()])

    es = cma.CMAEvolutionStrategy(x0, 0.1, {
        'popsize': config['pop_size'], 'verbose': -1, 'bounds': [0, None]})
    while not es.stop() and es.countevals < config['max_evals']:
        sols = es.ask()
        es.tell(sols, [_static_eval(s, layer_counts, sp, static_scores) for s in sols])

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
