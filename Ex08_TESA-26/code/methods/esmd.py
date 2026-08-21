# methods/esmd.py — E-SMD wrapper for Ex08 SimpleMLP
# E-SMD: Evolutionary Synaptic Metric Discovery
# Core: e_smd.py (evolves per-layer metric alpha/beta/gamma/p via CMA-ES)
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


@register('esmd', 'E-SMD (Metric)', '#d62728')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    import cma
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()

    layers = model.get_prunable_layers()
    L = len(layers)

    # ── Step 1: Collect activations via hooks (works for any architecture) ──
    X_cal, y_cal = next(iter(train_dl))
    X_abs_list = []
    hooks_act = []
    act_order = []

    def _make_act_hook(idx):
        def hook(m, inp, out):
            x = inp[0].detach()
            if x.dim() == 4:  # Conv2d: [B, C, H, W] -> [B*H*W, C]
                x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            elif x.dim() > 2:
                x = x.view(x.size(0), -1)
            act_order.append((idx, x.abs()))
        return hook

    for i, (_, layer, _) in enumerate(layers):
        hooks_act.append(layer.register_forward_hook(_make_act_hook(i)))

    with torch.no_grad():
        model(X_cal)

    for h in hooks_act:
        h.remove()

    # Sort by layer order
    act_order.sort(key=lambda x: x[0])
    X_abs_list = [a for _, a in act_order]

    # ── Precompute log-space bases ──
    W_logs, Row_logs, X_abs = [], [], []
    with torch.no_grad():
        for i, (_, layer, _) in enumerate(layers):
            W = layer.weight.data
            W2d = W.view(W.size(0), -1) if W.dim() > 2 else W
            W_logs.append(torch.log(torch.abs(W2d) + 1e-12))
            Row_logs.append(torch.log(torch.sum(torch.abs(W2d), dim=1, keepdim=True) + 1e-12))
            X_abs.append(X_abs_list[i])

    # ── Step 2: CMA-ES over [alpha, beta, gamma, p] per layer ──
    # Start from RIA-like: alpha=1, beta=0, gamma=1, p=1
    x0 = np.tile([1.0, 0.0, 1.0, 1.0], L)
    sigma0 = 0.5
    pop_size = config.get('pop_size', 12)
    max_evals = config.get('max_evals', 200)
    criterion = nn.CrossEntropyLoss()

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': pop_size, 'verbose': -1, 'bounds': [0.0, 10.0]
    })

    while not es.stop() and es.countevals < max_evals:
        solutions = es.ask()
        fitnesses = []

        for theta in solutions:
            with torch.no_grad():
                for i, (_, layer, mask_name) in enumerate(layers):
                    alpha, beta, gamma, p_val = theta[i*4: i*4+4]
                    p_val = max(0.1, float(p_val))

                    W_log = W_logs[i]
                    X = X_abs[i]

                    # Lp-norm of activations
                    if W_log.shape[1] == X.shape[1]:
                        Lp_norm = torch.linalg.vector_norm(X, ord=p_val, dim=0)
                    else:
                        Lp_norm = torch.ones(W_log.shape[1])
                    X_log = torch.log(Lp_norm + 1e-12).unsqueeze(0)

                    # Universal metric in log-space
                    Score = (beta + alpha) * W_log - alpha * Row_logs[i] + gamma * X_log

                    # Row-wise pruning (prevents neuron collapse)
                    k = max(1, int(W_log.shape[1] * (1.0 - sp)))
                    thresh = torch.topk(Score, k, dim=1).values[:, -1].unsqueeze(1)
                    mask = (Score >= thresh).float()

                    if layer.weight.dim() > 2:
                        mask = mask.view_as(layer.weight)
                    getattr(model, mask_name).copy_(mask)

                out = _plain_forward(model, X_cal)
                loss = criterion(out, y_cal).item()
                fitnesses.append(loss)

            # Restore masks
            with torch.no_grad():
                for _, _, mn in layers:
                    getattr(model, mn).fill_(1.0)

        es.tell(solutions, fitnesses)

    # ── Step 3: Apply best ──
    best = es.result.xbest
    with torch.no_grad():
        for i, (_, layer, mask_name) in enumerate(layers):
            alpha, beta, gamma, p_val = best[i*4: i*4+4]
            p_val = max(0.1, float(p_val))

            W_log = W_logs[i]
            X = X_abs[i]
            if W_log.shape[1] == X.shape[1]:
                Lp_norm = torch.linalg.vector_norm(X, ord=p_val, dim=0)
            else:
                Lp_norm = torch.ones(W_log.shape[1])
            X_log = torch.log(Lp_norm + 1e-12).unsqueeze(0)

            Score = (beta + alpha) * W_log - alpha * Row_logs[i] + gamma * X_log
            k = max(1, int(W_log.shape[1] * (1.0 - sp)))
            thresh = torch.topk(Score, k, dim=1).values[:, -1].unsqueeze(1)
            mask = (Score >= thresh).float()

            if layer.weight.dim() > 2:
                mask = mask.view_as(layer.weight)
            getattr(model, mask_name).copy_(mask)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    train_finetune_micro(model, train_dl, config.get('finetune_batches_evo', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
