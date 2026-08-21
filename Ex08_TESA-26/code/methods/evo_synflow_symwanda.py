# methods/evo_synflow_symwanda.py — SymWanda with static proxy (Multi-Fidelity)
import torch, numpy as np
from methods import register
from or08_01 import (create_model, apply_ratios, train_finetune_micro,
                     evaluate_full, set_seed, check_mask_connectivity,
                     normalize_genome, apply_global)
import torch.nn as nn


def _precompute_symwanda_proxy(teacher_state, train_dl):
    """Static proxy with SymWanda: |W| * (||X|| + ||Y||) * |G|."""
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()
    # Collect activation norms
    hooks, x_norms, y_norms = [], {}, {}
    def hook_fn(name):
        def fn(m, inp, out):
            x, y = inp[0].detach(), out.detach()
            x_norms[name] = x.pow(2).sum(dim=0).sqrt() if x.dim() == 2 else x.pow(2).sum(dim=(0,2,3)).sqrt()
            y_norms[name] = y.pow(2).sum(dim=0).sqrt() if y.dim() == 2 else y.pow(2).sum(dim=(0,2,3)).sqrt()
        return fn
    for name, layer, _ in model.get_prunable_layers():
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    with torch.no_grad():
        X, _ = next(iter(train_dl))
        model(X)
    for h in hooks: h.remove()
    # Compute static scores
    model.zero_grad()
    layers = model.get_prunable_layers()
    first_l = layers[0][1]
    x_in = torch.ones(1, first_l.weight.shape[1]) if first_l.weight.dim() == 2 and first_l.weight.shape[1] <= 10 else torch.ones(1,1,28,28)
    out = model(x_in); out.sum().backward()
    scores, layer_counts = {}, []
    with torch.no_grad():
        for i, (name, layer, _) in enumerate(layers):
            W = layer.weight
            G = layer.weight.grad if layer.weight.grad is not None else torch.zeros_like(W)
            xn = x_norms.get(name, torch.ones(W.shape[1] if W.dim()==2 else W.shape[1]))
            yn = y_norms.get(name, torch.ones(W.shape[0]))
            if W.dim() == 2:
                score = W.abs() * (xn.view(1,-1) + yn.view(-1,1)) * G.abs().clamp(min=1e-8)
            else:
                score = W.abs() * (xn.view(1,-1,1,1) + yn.view(-1,1,1,1)) * G.abs().clamp(min=1e-8)
            scores[i] = score.view(-1).numpy()
            layer_counts.append(W.numel())
    return scores, np.array(layer_counts)


@register('evo-synflow-symwanda', 'Evo-SynFlow (SymWanda)', '#ff7f0e')
def run(teacher_state, sp, seed, config, train_dl, test_dl, *, _state=None):
    import cma
    set_seed(seed)
    static_scores, layer_counts = _precompute_symwanda_proxy(teacher_state, train_dl)

    temp = create_model(); temp.load_state_dict(teacher_state); apply_global(temp, sp)
    x0 = np.array([getattr(temp, mn).sum().item()/getattr(temp, mn).numel() for _,_,mn in temp.get_prunable_layers()])

    es = cma.CMAEvolutionStrategy(x0, 0.15*(1-sp), {'popsize': config['pop_size'], 'verbose': -1, 'bounds': [0, None]})
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

    best_ratios = normalize_genome(es.result.xbest, layer_counts, sp)
    model = create_model(); model.load_state_dict(teacher_state); apply_ratios(model, best_ratios)
    if not check_mask_connectivity(model): return {'F1': 0.333}
    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    if _state is not None: _state['prev_ratios'] = best_ratios
    return {'F1': f1}
