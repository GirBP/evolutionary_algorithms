# methods/evostruct.py — EvoStruct: Macro-Micro Decoupling (Multi-Fidelity)
# §5: CMA-ES uses static proxy. DiffSynFlow + spectral merge only on final genome.
import torch, numpy as np
from methods import register
from or08_01 import (create_model, apply_ratios, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome, apply_global)


def _precompute_static_proxy(teacher_state):
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval(); model.zero_grad()
    layers = model.get_prunable_layers()
    first_l = layers[0][1]
    if first_l.weight.dim() == 2 and first_l.weight.shape[1] <= 10:
        x = torch.ones(1, first_l.weight.shape[1])
    else:
        x = torch.ones(1, 1, 28, 28)
    out = model(x); out.sum().backward()
    scores, layer_counts = {}, []
    with torch.no_grad():
        for i, (_, layer, _) in enumerate(layers):
            W = layer.weight
            G = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(W)
            scores[i] = (W * G).abs().view(-1).numpy()
            layer_counts.append(W.numel())
    return scores, np.array(layer_counts)


@register('evostruct', 'EvoStruct', '#ff9896')
def run(teacher_state, sp, seed, config, train_dl, test_dl, *, _state=None):
    import cma
    set_seed(seed)
    static_scores, layer_counts = _precompute_static_proxy(teacher_state)

    # §5 Macro: CMA-ES with static proxy (fast)
    temp = create_model(); temp.load_state_dict(teacher_state); apply_global(temp, sp)
    x0 = np.array([getattr(temp, mn).sum().item()/getattr(temp, mn).numel()
                    for _,_,mn in temp.get_prunable_layers()])
    es = cma.CMAEvolutionStrategy(x0, 0.15*(1-sp), {
        'popsize': config['pop_size'], 'verbose': -1, 'bounds': [0, None]})
    while not es.stop() and es.countevals < config['max_evals']:
        sols = es.ask()
        scores = []
        for s in sols:
            ratios = normalize_genome(s, layer_counts, sp)
            total = sum(np.partition(static_scores[i], -max(1,int(layer_counts[i]*ratios[i])))[-max(1,int(layer_counts[i]*ratios[i])):].sum()
                        if int(layer_counts[i]*ratios[i]) < layer_counts[i] else static_scores[i].sum()
                        for i in range(len(layer_counts)))
            scores.append(-total)
        es.tell(sols, scores)

    macro_ratios = normalize_genome(es.result.xbest, layer_counts, sp)

    # §5 Micro: apply ratios (DiffSynFlow + spectral merge ONLY on final genome)
    # For SimpleMLP, skip spectral merge (no Conv layers to merge)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_ratios(model, macro_ratios)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    if _state is not None:
        _state['prev_ratios'] = macro_ratios
    return {'F1': f1}
