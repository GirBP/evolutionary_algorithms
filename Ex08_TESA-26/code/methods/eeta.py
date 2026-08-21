# methods/eeta.py — E-ETA wrapper for Ex08 SimpleMLP
# E-ETA: Evolutionary Elastic Topology Adaptation
# Core: e_eta.py (meta-matrix Theta[2x3] maps landscape features → per-layer decisions)
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


def _norm_global(t):
    t_min, t_max = t.min(), t.max()
    return (t - t_min) / (t_max - t_min + 1e-12)


def _norm_row(t):
    t2 = t.view(t.size(0), -1) if t.dim() > 2 else t
    r_min = t2.min(dim=1, keepdim=True).values
    r_max = t2.max(dim=1, keepdim=True).values
    return ((t2 - r_min) / (r_max - r_min + 1e-12)).view_as(t)


@register('eeta', 'E-ETA (Elastic)', '#8c564b')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    import cma
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()

    layers = model.get_prunable_layers()
    L = len(layers)

    # ── Step 1: Collect activations + compute WANDA/RIA bases ──
    X_cal, y_cal = next(iter(train_dl))
    X_abs_list = []

    with torch.no_grad():
        x = X_cal
        if x.dim() > 2 and x.dim() != 4:
            x = x.view(x.size(0), -1)
        if x.dim() == 4:
            for i, (_, layer, mask_name) in enumerate(layers):
                w = layer.weight * getattr(model, mask_name)
                if isinstance(layer, nn.Conv2d):
                    X_abs_list.append(None)  # skip conv for now
                    x = F.conv2d(x, w, layer.bias, padding=layer.padding, stride=layer.stride)
                    if hasattr(model, 'pool') and i < 2:
                        x = model.pool(F.relu(x))
                    else:
                        x = F.relu(x)
                elif isinstance(layer, nn.Linear):
                    if x.dim() > 2: x = x.view(x.size(0), -1)
                    X_abs_list.append(x.abs())
                    x = F.linear(x, w, layer.bias)
                    if i < L - 1: x = F.relu(x)
        else:
            if x.dim() > 2: x = x.view(x.size(0), -1)
            for i, (_, layer, mask_name) in enumerate(layers):
                w = layer.weight * getattr(model, mask_name)
                X_abs_list.append(x.abs())
                x = F.linear(x, w, layer.bias)
                if i < L - 1: x = F.relu(x)

    # Compute WANDA/RIA scores + landscape features
    S_Wanda, S_RIA, V_features = [], [], []

    with torch.no_grad():
        for i, (_, layer, _) in enumerate(layers):
            W = layer.weight.data
            W2d = W.view(W.size(0), -1) if W.dim() > 2 else W
            X = X_abs_list[i]

            if X is not None and W2d.shape[1] == X.shape[1]:
                norm_x2 = torch.linalg.vector_norm(X, ord=2, dim=0)
                norm_x1 = torch.linalg.vector_norm(X, ord=1, dim=0)
            else:
                norm_x2 = torch.ones(W2d.shape[1])
                norm_x1 = torch.ones(W2d.shape[1])

            s_w = _norm_global(W2d.abs() * norm_x2.unsqueeze(0))
            row_sum = W2d.abs().sum(dim=1, keepdim=True) + 1e-9
            s_r = _norm_global((W2d.abs() / row_sum) * norm_x1.unsqueeze(0))

            S_Wanda.append(s_w.view_as(W))
            S_RIA.append(s_r.view_as(W))

            # Landscape features
            w_var = W2d.abs().var() / (W2d.abs().mean() ** 2 + 1e-9)
            if X is not None and X.shape[0] > 1:
                mu_x = X.mean(dim=0)
                std_x = X.std(dim=0) + 1e-9
                kurtosis = torch.mean(((X - mu_x) / std_x) ** 4)
            else:
                kurtosis = torch.tensor(3.0)
            shape_ratio = W2d.shape[0] / W2d.shape[1]
            V_features.append([w_var.item(), kurtosis.item(), shape_ratio])

    V = torch.tensor(V_features, dtype=torch.float32)
    V = (V - V.mean(dim=0)) / (V.std(dim=0) + 1e-5)

    # ── Step 2: CMA-ES over meta-matrix Theta [2×3] = 6 params ──
    x0 = np.zeros(6)
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

        for theta_flat in solutions:
            Theta = torch.tensor(theta_flat, dtype=torch.float32).view(2, 3)
            Z = V @ Theta.T  # [L, 2]
            Alphas = torch.sigmoid(Z[:, 0])  # Wanda vs RIA blend
            Rhos = torch.sigmoid(Z[:, 1])     # Global vs Row-wise

            with torch.no_grad():
                for i, (_, layer, mask_name) in enumerate(layers):
                    alpha, rho = Alphas[i].item(), Rhos[i].item()
                    S_base = alpha * S_Wanda[i] + (1.0 - alpha) * S_RIA[i]
                    S_glob = _norm_global(S_base)
                    S_row = _norm_row(S_base)
                    S_elastic = (1.0 - rho) * S_glob + rho * S_row

                    k = max(1, int(S_elastic.numel() * (1.0 - sp)))
                    thresh = torch.topk(S_elastic.view(-1), k).values[-1]
                    mask = (S_elastic >= thresh).float()
                    if layer.weight.dim() > 2:
                        mask = mask.view_as(layer.weight)
                    getattr(model, mask_name).copy_(mask)

                out = _plain_forward(model, X_cal)
                loss = criterion(out, y_cal).item()
                fitnesses.append(loss)

            with torch.no_grad():
                for _, _, mn in layers:
                    getattr(model, mn).fill_(1.0)

        es.tell(solutions, fitnesses)

    # ── Step 3: Apply best ──
    best = es.result.xbest
    Theta = torch.tensor(best, dtype=torch.float32).view(2, 3)
    Z = V @ Theta.T
    Alphas = torch.sigmoid(Z[:, 0])
    Rhos = torch.sigmoid(Z[:, 1])

    with torch.no_grad():
        for i, (_, layer, mask_name) in enumerate(layers):
            alpha, rho = Alphas[i].item(), Rhos[i].item()
            S_base = alpha * S_Wanda[i] + (1.0 - alpha) * S_RIA[i]
            S_elastic = (1.0 - rho) * _norm_global(S_base) + rho * _norm_row(S_base)

            k = max(1, int(S_elastic.numel() * (1.0 - sp)))
            thresh = torch.topk(S_elastic.view(-1), k).values[-1]
            mask = (S_elastic >= thresh).float()
            if layer.weight.dim() > 2:
                mask = mask.view_as(layer.weight)
            getattr(model, mask_name).copy_(mask)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
