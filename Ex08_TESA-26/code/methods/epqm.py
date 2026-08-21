# methods/epqm.py — E-PQM wrapper for Ex08 SimpleMLP
# E-PQM: Evolutionary Phase-Space Quantization Mapping
# Core: e_pqm.py (2D phase space |W|×|G| → M×M density matrix evolved by CMA-ES)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)


def _plain_forward(model, x):
    """Forward through model with proper architecture handling."""
    return model(x)


@register('epqm', 'E-PQM (Phase)', '#17becf')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    import cma
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    layers = model.get_prunable_layers()
    L = len(layers)
    M = 8
    num_bins = M * M

    # ── Step 1: Compute phase space (|W| × |G|) ──
    model.eval()
    for _, layer, _ in layers:
        layer.weight.requires_grad_(True)
    model.zero_grad()

    X_cal, y_cal = next(iter(train_dl))
    out = _plain_forward(model, X_cal)
    nn.CrossEntropyLoss()(out, y_cal).backward()

    W_list, G_list, layer_offsets = [], [], []
    offset = 0
    with torch.no_grad():
        for _, layer, _ in layers:
            W = layer.weight.data.view(-1).abs()
            G = layer.weight.grad.view(-1).abs() if layer.weight.grad is not None else torch.zeros_like(W)
            W_list.append(W); G_list.append(G)
            layer_offsets.append((offset, offset + W.numel()))
            offset += W.numel()
        W_all = torch.cat(W_list); G_all = torch.cat(G_list)
    model.zero_grad()

    total_params = W_all.numel()
    target_keep = int(total_params * (1.0 - sp))

    # Quantize into M×M grid
    W_log = torch.log(W_all + 1e-12); G_log = torch.log(G_all + 1e-12)
    W_norm = (W_log - W_log.min()) / (W_log.max() - W_log.min() + 1e-12)
    G_norm = (G_log - G_log.min()) / (G_log.max() - G_log.min() + 1e-12)
    w_bin = torch.clamp((W_norm * M).long(), 0, M - 1)
    g_bin = torch.clamp((G_norm * M).long(), 0, M - 1)
    bin_ids = w_bin * M + g_bin
    local_scores = W_all * G_all

    bin_indices = []
    bin_caps = np.zeros(num_bins)
    for b in range(num_bins):
        idx = torch.where(bin_ids == b)[0]
        if len(idx) > 0:
            sorted_idx = idx[torch.argsort(local_scores[idx], descending=True)]
            bin_indices.append(sorted_idx)
            bin_caps[b] = len(sorted_idx)
        else:
            bin_indices.append(torch.tensor([], dtype=torch.long))

    def project(p_vec):
        def calc_k(nu): return np.clip(p_vec - nu, 0.0, 1.0)
        lo, hi = -2.0, 2.0
        for _ in range(35):
            mid = (lo + hi) / 2.0
            if np.sum(calc_k(mid) * bin_caps) > target_keep: lo = mid
            else: hi = mid
        return calc_k((lo + hi) / 2.0)

    # ── Step 2: CMA-ES over density matrix ──
    x0 = np.full(num_bins, 1.0 - sp)
    sigma0 = 0.1
    pop_size = config.get('pop_size', 12)
    max_evals = config.get('max_evals', 200)
    criterion = nn.CrossEntropyLoss()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': pop_size, 'verbose': -1
    })

    while not es.stop() and es.countevals < max_evals:
        solutions = es.ask()
        fitnesses = []
        for p_vec in solutions:
            p_valid = project(p_vec)
            global_mask = torch.zeros(total_params)
            for b in range(num_bins):
                k = int(p_valid[b] * bin_caps[b])
                if k > 0:
                    global_mask[bin_indices[b][:k]] = 1.0

            with torch.no_grad():
                for i, (_, layer, mask_name) in enumerate(layers):
                    s, e = layer_offsets[i]
                    getattr(model, mask_name).copy_(global_mask[s:e].view_as(layer.weight))
                out = _plain_forward(model, X_cal)
                loss = criterion(out, y_cal).item()
                fitnesses.append(loss)
                for _, _, mn in layers:
                    getattr(model, mn).fill_(1.0)
        es.tell(solutions, fitnesses)

    # ── Step 3: Apply best ──
    best_p = project(es.result.xbest)
    global_mask = torch.zeros(total_params)
    for b in range(num_bins):
        k = int(best_p[b] * bin_caps[b])
        if k > 0:
            global_mask[bin_indices[b][:k]] = 1.0

    with torch.no_grad():
        for i, (_, layer, mask_name) in enumerate(layers):
            s, e = layer_offsets[i]
            getattr(model, mask_name).copy_(global_mask[s:e].view_as(layer.weight))

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
