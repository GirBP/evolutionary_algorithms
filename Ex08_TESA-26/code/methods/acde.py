# methods/acde.py — ACDE (Aitchison Compositional DE) wrapper for Ex08 benchmark
# Core algorithm in methods/_acde_core.py
import torch
import torch.nn as nn
import numpy as np
import random as rnd
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)

from methods._acde_core import ACDE_Pruner
from methods._forward_wrapper import PlainForwardWrapper


@register('acde', 'ACDE', '#756bb1')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    wrapper = PlainForwardWrapper(model)

    pruner = ACDE_Pruner(
        wrapper,
        target_sparsity=sp,
        pop_size=config.get('pop_size', 15),
        generations=config.get('fes_generations', 10),
    )

    criterion = nn.CrossEntropyLoss()

    # Run core algorithm manually (same as prune() but without parametrize finalization)
    saliency = pruner._compute_fes_saliency(train_dl, criterion)
    L = len(pruner.prunable_modules)

    np.random.seed(seed)
    rnd.seed(seed)

    base_p = pruner._aitchison_closure(pruner.layer_sizes)
    population = [pruner._aitchison_closure(base_p * np.exp(np.random.normal(0, 0.1, L)))
                  for _ in range(pruner.pop_size)]

    def get_fitness(p_vec):
        k_vec = pruner._decode_simplex_to_quotas(p_vec)
        orig_weights = []
        with torch.no_grad():
            for i, (m, sal, k) in enumerate(zip(pruner.prunable_modules, saliency, k_vec)):
                orig_weights.append(m.weight.data.clone())
                num_keep = max(1, int(k * sal.numel()))
                threshold = torch.topk(sal.view(-1), num_keep).values[-1]
                m.weight.data.mul_((sal >= threshold).float())
            inputs, targets = next(iter(train_dl))
            loss = criterion(wrapper(inputs), targets).item()
            for i, m in enumerate(pruner.prunable_modules):
                m.weight.data.copy_(orig_weights[i])
        return -loss

    fitnesses = [get_fitness(ind) for ind in population]

    for gen in range(pruner.generations):
        for i in range(pruner.pop_size):
            idxs = [idx for idx in range(pruner.pop_size) if idx != i]
            r1, r2, r3 = rnd.sample(idxs, 3)
            mutant = pruner._compositional_mutation(population[r1], population[r2], population[r3])
            cross = np.random.rand(L) < pruner.CR
            if not np.any(cross):
                cross[np.random.randint(L)] = True
            trial = np.where(cross, mutant, population[i])
            trial = pruner._aitchison_closure(trial)
            f_trial = get_fitness(trial)
            if f_trial > fitnesses[i]:
                population[i], fitnesses[i] = trial, f_trial

    best_p = population[np.argmax(fitnesses)]
    best_k = pruner._decode_simplex_to_quotas(best_p)

    # Apply masks to original model buffers
    with torch.no_grad():
        for i, (name, layer, mask_name) in enumerate(model.get_prunable_layers()):
            sal = saliency[i]
            k = best_k[i]
            num_keep = max(1, int(k * sal.numel()))
            threshold = torch.topk(sal.view(-1), num_keep).values[-1]
            mask = (sal >= threshold).float()
            getattr(model, mask_name).copy_(mask)
            layer.weight.data.mul_(mask)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
