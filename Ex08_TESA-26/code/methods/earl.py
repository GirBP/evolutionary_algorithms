# methods/earl.py — EARL wrapper for Ex08 SimpleMLP
# EARL: Evolutionary Anisotropic Ricci Landscape
# Core: tesa_rev.py (Forman-Ricci curvature + CMA-ES over alpha/beta)
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


@register('earl', 'EARL (Ricci)', '#9467bd')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    import cma
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()

    layers = model.get_prunable_layers()
    L = len(layers)
    layer_sizes = np.array([l.weight.numel() for _, l, _ in layers])
    total_params = layer_sizes.sum()
    target_keep = int(total_params * (1.0 - sp))

    # ── Step 1: Compute differential geometry (gamma + Forman-Ricci) ──
    activation_norms = {}
    hooks = []

    def make_hook(idx):
        def hook(m, inp, out):
            x = inp[0].detach()
            if x.dim() > 2:
                x = x.view(-1, x.size(-1))
            norm = torch.norm(x, p=1, dim=0)
            activation_norms[idx] = norm
        return hook

    for i, (_, layer, _) in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        X_cal, y_cal = next(iter(train_dl))
        _plain_forward(model, X_cal)

    for h in hooks:
        h.remove()

    gamma_tensors, frc_tensors = [], []
    with torch.no_grad():
        for i, (_, layer, _) in enumerate(layers):
            W = layer.weight.data
            orig_shape = W.shape
            W2d = W.view(W.size(0), -1) if W.dim() > 2 else W

            x_norm = activation_norms.get(i, torch.ones(W2d.shape[1]))
            if W2d.shape[1] != x_norm.shape[0]:
                x_norm = torch.ones(W2d.shape[1])

            # Kinetic mass γ = |W| · ||X||₁
            gamma = torch.abs(W2d) * x_norm.unsqueeze(0)

            # Forman-Ricci curvature: R = 3γ - Σᵢγ - Σⱼγ
            R_i = gamma.sum(dim=1, keepdim=True)
            C_j = gamma.sum(dim=0, keepdim=True)
            frc = 3 * gamma - R_i - C_j

            gamma_tensors.append((gamma / (gamma.mean() + 1e-9)).view(orig_shape))
            frc_tensors.append((frc / (frc.abs().mean() + 1e-9)).view(orig_shape))

    # ── Step 2: CMA-ES over alpha/beta ──
    x0 = np.ones(2 * L)
    sigma0 = 0.5
    pop_size = config.get('pop_size', 12)
    max_evals = config.get('max_evals', 200)
    criterion = nn.CrossEntropyLoss()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': pop_size, 'verbose': -1
    })

    while not es.stop() and es.countevals < max_evals:
        solutions = es.ask()
        fitnesses = []

        for theta in solutions:
            alphas, betas = theta[:L], theta[L:]

            # Topological energy per layer
            energies = []
            for i in range(L):
                E = alphas[i] * gamma_tensors[i] - betas[i] * frc_tensors[i]
                energies.append(E.view(-1))

            # Global threshold (emergent layer allocation)
            global_E = torch.cat(energies)
            k_remove = max(1, total_params - target_keep)
            tau = torch.kthvalue(global_E, k_remove).values

            # Apply masks + evaluate loss
            with torch.no_grad():
                for i, (_, layer, mask_name) in enumerate(layers):
                    E_layer = alphas[i] * gamma_tensors[i] - betas[i] * frc_tensors[i]
                    mask = (E_layer >= tau).float()
                    getattr(model, mask_name).copy_(mask)

                out = _plain_forward(model, X_cal)
                loss = criterion(out, y_cal).item()
                fitnesses.append(loss)  # CMA-ES minimizes

            # Restore masks to ones
            with torch.no_grad():
                for _, _, mn in layers:
                    getattr(model, mn).fill_(1.0)

        es.tell(solutions, fitnesses)

    # ── Step 3: Apply best solution ──
    best = es.result.xbest
    alphas, betas = best[:L], best[L:]
    energies = [(alphas[i] * gamma_tensors[i] - betas[i] * frc_tensors[i]).view(-1) for i in range(L)]
    global_E = torch.cat(energies)
    k_remove = max(1, total_params - target_keep)
    tau = torch.kthvalue(global_E, k_remove).values

    with torch.no_grad():
        for i, (_, layer, mask_name) in enumerate(layers):
            E_layer = alphas[i] * gamma_tensors[i] - betas[i] * frc_tensors[i]
            mask = (E_layer >= tau).float()
            getattr(model, mask_name).copy_(mask)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
