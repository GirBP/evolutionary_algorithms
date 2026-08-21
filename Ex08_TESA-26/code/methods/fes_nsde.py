# methods/fes_nsde.py — FES-NSDE wrapper for Ex08 benchmark
# Core algorithm in methods/_fes_nsde_core.py
import sys
import torch
import torch.nn as nn
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)

from methods._fes_nsde_core import Pruner_FES_NSDE
from methods._forward_wrapper import PlainForwardWrapper


def _convert_fes_masks_to_buffers(model, pruner, saliency, best_k):
    """Extract masks from FES-NSDE result into our buffer system."""
    with torch.no_grad():
        prunable = list(model.get_prunable_layers())
        for i, (name, layer, mask_name) in enumerate(prunable):
            sal = saliency[i]
            k = best_k[i]
            num_keep = max(1, int(k * sal.numel()))
            threshold = torch.topk(sal.view(-1), num_keep).values[-1]
            mask = (sal >= threshold).float()
            getattr(model, mask_name).copy_(mask)
            layer.weight.data.mul_(mask)


@register('fes-nsde', 'FES-NSDE', '#e6550d')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    wrapper = PlainForwardWrapper(model)

    pruner = Pruner_FES_NSDE(
        wrapper,
        target_sparsity=sp,
        pop_size=config.get('pop_size', 15),
        generations=config.get('fes_generations', 10),
    )

    criterion = nn.CrossEntropyLoss()

    # Run evolution — this applies parametrize to wrapper's layers
    # We intercept before finalization: compute saliency + best genome manually
    saliency = pruner._compute_fes_saliency(train_dl, criterion)
    L = len(pruner.prunable_modules)
    import numpy as np, random as rnd
    rnd.seed(seed); np.random.seed(seed)
    base_k = np.full(L, 1.0 - sp)
    population = [pruner._evolutionary_repair(base_k + np.random.normal(0, 0.05, L))
                  for _ in range(pruner.pop_size)]
    fitnesses = [pruner._evaluate_fitness(ind, saliency, train_dl, criterion)
                 for ind in population]

    for gen in range(pruner.generations):
        for i in range(pruner.pop_size):
            idxs = list(range(pruner.pop_size)); idxs.remove(i)
            r1, r2, r3 = rnd.sample(idxs, 3)
            mutant = population[r1] + pruner.F * (population[r2] - population[r3])
            cross_points = np.random.rand(L) < pruner.CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, L)] = True
            trial = np.where(cross_points, mutant, population[i])
            trial_repaired = pruner._evolutionary_repair(trial)
            f_trial = pruner._evaluate_fitness(trial_repaired, saliency, train_dl, criterion)
            if f_trial > fitnesses[i]:
                population[i] = trial_repaired
                fitnesses[i] = f_trial

    best_k = population[np.argmax(fitnesses)]

    # Apply masks to original model buffers
    _convert_fes_masks_to_buffers(model, pruner, saliency, best_k)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
