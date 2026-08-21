# methods/eacde.py — E-ACDE (Elastic ACDE) wrapper for Ex08 benchmark
# Core algorithm in methods/_eacde_core.py
import torch
import torch.nn as nn
import numpy as np
import random as rnd
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)

from methods._eacde_core import EACDE_Pruner
from methods._forward_wrapper import PlainForwardWrapper


@register('eacde', 'E-ACDE', '#e7298a')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    wrapper = PlainForwardWrapper(model)

    pruner = EACDE_Pruner(
        wrapper,
        min_sparsity=sp,
        tolerance=1.15,
        pop_size=config.get('pop_size', 15),
        generations=config.get('fes_generations', 10),
    )

    criterion = nn.CrossEntropyLoss()

    # Run core: compute saliency, base loss, evolve
    saliency = pruner._compute_fes_saliency(train_dl, criterion)

    base_c = pruner._aitchison_closure(np.append(pruner.layer_sizes, pruner.B_max * 0.001))
    k_base, _ = pruner._decode_simplex(base_c)

    inputs, targets = next(iter(train_dl))

    orig_weights = [m.weight.data.clone() for m in pruner.prunable_modules]
    with torch.no_grad():
        for i, m in enumerate(pruner.prunable_modules):
            num_keep = max(1, int(k_base[i] * saliency[i].numel()))
            thresh = torch.topk(saliency[i].view(-1), num_keep).values[-1]
            m.weight.data.mul_((saliency[i] >= thresh).float())
        base_loss = criterion(wrapper(inputs), targets).item()
        for i, m in enumerate(pruner.prunable_modules):
            m.weight.data.copy_(orig_weights[i])

    L_max = base_loss * pruner.tolerance

    np.random.seed(seed)
    rnd.seed(seed)

    population = [pruner._aitchison_closure(base_c * np.exp(np.random.normal(0, 0.1, pruner.dim)))
                  for _ in range(pruner.pop_size)]

    def evaluate(c_vec):
        k_vec, actual_sp = pruner._decode_simplex(c_vec)
        with torch.no_grad():
            for i, m in enumerate(pruner.prunable_modules):
                num_keep = max(1, int(k_vec[i] * saliency[i].numel()))
                thresh = torch.topk(saliency[i].view(-1), num_keep).values[-1]
                m.weight.data.mul_((saliency[i] >= thresh).float())
            loss = criterion(wrapper(inputs), targets).item()
            for i, m in enumerate(pruner.prunable_modules):
                m.weight.data.copy_(orig_weights[i])
        return loss, actual_sp

    eval_data = [evaluate(ind) for ind in population]

    for gen in range(pruner.generations):
        for i in range(pruner.pop_size):
            idxs = [idx for idx in range(pruner.pop_size) if idx != i]
            r1, r2, r3 = rnd.sample(idxs, 3)
            mutant = pruner._compositional_mutation(population[r1], population[r2], population[r3])
            cross = np.random.rand(pruner.dim) < pruner.CR
            if not np.any(cross):
                cross[np.random.randint(pruner.dim)] = True
            trial = pruner._aitchison_closure(np.where(cross, mutant, population[i]))
            t_loss, t_sp = evaluate(trial)
            p_loss, p_sp = eval_data[i]

            # LATS selection
            t_in = t_loss <= L_max
            p_in = p_loss <= L_max
            replace = False
            if t_in and p_in:
                replace = t_sp > p_sp
            elif t_in and not p_in:
                replace = True
            elif not t_in and not p_in:
                replace = t_loss < p_loss

            if replace:
                population[i] = trial
                eval_data[i] = (t_loss, t_sp)

    # Select champion
    valid_pop = [(population[i], eval_data[i]) for i in range(pruner.pop_size) if eval_data[i][0] <= L_max]
    if valid_pop:
        champion_p, champion_eval = max(valid_pop, key=lambda x: x[1][1])
    else:
        champion_p, champion_eval = min(zip(population, eval_data), key=lambda x: x[1][0])

    best_k, _ = pruner._decode_simplex(champion_p)

    # Apply masks to model buffers
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
