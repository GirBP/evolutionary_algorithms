# methods/evo_synflow.py — Evo-SynFlow baseline with Graph-Detached Static Proxy (Multi-Fidelity)
import torch
import numpy as np
from methods import register
from or08_01 import (create_model, apply_ratios, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome, apply_global)


def _precompute_static_proxy(teacher_state):
    """§4 Graph-Detached: compute G0 = ∇θ L(θ; D_calib) once."""
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()
    model.zero_grad()
    layers = model.get_prunable_layers()
    first_layer = layers[0][1]
    if hasattr(first_layer, 'weight') and first_layer.weight.dim() == 2 and first_layer.weight.shape[1] <= 10:
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
            scores[i] = (W * G).abs().view(-1).numpy()
            layer_counts.append(W.numel())
    return scores, np.array(layer_counts)


def _static_eval(genome, layer_counts, target_sparsity, static_scores):
    """§4 Vectorized RAM operation: S(m) = m^T (|W|^2 ⊙ σ(|G0|)). No autograd."""
    ratios = normalize_genome(genome, layer_counts, target_sparsity)
    total = 0.0
    for i, (ratio, n) in enumerate(zip(ratios, layer_counts)):
        k = max(1, int(n * ratio))
        sc = static_scores[i]
        if k >= len(sc):
            total += sc.sum()
        else:
            total += np.partition(sc, -k)[-k:].sum()
    return -total


@register('evo-synflow', 'Evo-SynFlow', '#2ca02c')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    import cma
    set_seed(seed)
    static_scores, layer_counts = _precompute_static_proxy(teacher_state)
    dim = len(layer_counts)

    # §4 Warm start from magnitude ratios
    temp = create_model()
    temp.load_state_dict(teacher_state)
    apply_global(temp, sp)
    x0 = []
    with torch.no_grad():
        for _, layer, mask_name in temp.get_prunable_layers():
            mask = getattr(temp, mask_name)
            x0.append(mask.sum().item() / mask.numel())
    x0 = np.array(x0)

    sigma0 = 0.15 * (1.0 - sp)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': config['pop_size'],
        'verbose': -1,
        'bounds': [0, None],
    })
    while not es.stop() and es.countevals < config['max_evals']:
        solutions = es.ask()
        scores = [_static_eval(s, layer_counts, sp, static_scores) for s in solutions]
        es.tell(solutions, scores)

    best_ratios = normalize_genome(es.result.xbest, layer_counts, sp)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_ratios(model, best_ratios)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
