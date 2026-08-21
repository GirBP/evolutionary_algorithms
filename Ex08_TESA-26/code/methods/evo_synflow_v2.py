# methods/evo_synflow_v2.py — Evo-SynFlow with Taylor proxy instead of SynFlow
import torch, numpy as np
from methods import register
from methods.tesa26 import _compute_saliency
from or08_01 import (create_model, apply_ratios, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome, apply_global)


def _taylor_static_proxy(teacher_state, train_dl):
    """Taylor-based static proxy: abs(W * dL/dW) from calibration data."""
    model = create_model()
    model.load_state_dict(teacher_state)
    saliency = _compute_saliency(model, train_dl)
    scores = {}
    layer_counts = []
    for i, sal in enumerate(saliency):
        scores[i] = sal.view(-1).numpy()
        layer_counts.append(sal.numel())
    return scores, np.array(layer_counts)


def _taylor_eval(genome, layer_counts, target_sparsity, static_scores):
    ratios = normalize_genome(genome, layer_counts, target_sparsity)
    total = 0.0
    for i, (ratio, n) in enumerate(zip(ratios, layer_counts)):
        k = max(1, int(n * ratio))
        sc = static_scores[i]
        total += np.partition(sc, -min(k, len(sc)))[-min(k, len(sc)):].sum() if k < len(sc) else sc.sum()
    return -total


@register('evo-synflow-v2', 'Evo-SynFlow-Taylor', '#98df8a')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    import cma
    set_seed(seed)
    scores, layer_counts = _taylor_static_proxy(teacher_state, train_dl)

    temp = create_model(); temp.load_state_dict(teacher_state); apply_global(temp, sp)
    x0 = np.array([getattr(temp, mn).sum().item() / getattr(temp, mn).numel()
                    for _, _, mn in temp.get_prunable_layers()])

    es = cma.CMAEvolutionStrategy(x0, 0.15 * (1 - sp), {
        'popsize': config['pop_size'], 'verbose': -1, 'bounds': [0, None]})
    while not es.stop() and es.countevals < config['max_evals']:
        sols = es.ask()
        es.tell(sols, [_taylor_eval(s, layer_counts, sp, scores) for s in sols])

    best_ratios = normalize_genome(es.result.xbest, layer_counts, sp)
    model = create_model(); model.load_state_dict(teacher_state); apply_ratios(model, best_ratios)
    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
