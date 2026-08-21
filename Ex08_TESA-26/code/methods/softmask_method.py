# methods/softmask_method.py — SoftMask: learnable thresholds (self-contained, Multi-Fidelity)
#
# Wanda scores → normalize per row → learn threshold t via gradient descent on soft mask
# §3: Quantile Init + Frozen Coreset + Exact Top-K Projection
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)


def _wanda_scores(W, x_norm):
    """S = |W| * ||X||_2."""
    if W.dim() == 4:
        return W.abs() * x_norm.view(1, -1, 1, 1)
    elif W.dim() == 2:
        return W.abs() * x_norm.view(1, -1)
    return W.abs()


def _normalize_uniform(S):
    """Percentile-rank normalization per output row → uniform [0, 1]."""
    if S.dim() == 4:
        S_flat = S.view(S.size(0), -1)
    elif S.dim() == 2:
        S_flat = S
    else:
        return S
    S_norm = torch.zeros_like(S_flat)
    for i in range(S_flat.size(0)):
        row = S_flat[i].flatten()
        _, idx = row.sort()
        ranks = torch.zeros_like(row)
        ranks[idx] = torch.linspace(0, 1, len(row), device=row.device)
        S_norm[i] = ranks.view_as(S_flat[i])
    return S_norm.view_as(S)


def _run_softmask(teacher_state, sparsity, seed, train_dl, device,
                   n_iters=30, lr=5e-3, weight_decay=0.05, lambda_reg=1.0):
    """SoftMask core: learn thresholds on frozen coreset, exact top-K projection."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    t0 = time.process_time()

    model = create_model()
    model.load_state_dict(copy.deepcopy(teacher_state))
    model.to(device)
    model.eval()

    # Collect activation norms (coreset: 1 batch)
    x_norms = {}
    hooks = []
    def hook_fn(name):
        def hook(_, inp, _out):
            x = inp[0].detach()
            if x.dim() == 4:
                x_norms[name] = x.pow(2).sum(dim=(0, 2, 3)).sqrt()
            elif x.dim() == 2:
                x_norms[name] = x.pow(2).sum(dim=0).sqrt()
            else:
                x_norms[name] = x.pow(2).sum(dim=list(range(x.dim()))).sqrt()
        return hook

    for name, layer, _ in model.get_prunable_layers():
        hooks.append(layer.register_forward_hook(hook_fn(name)))
    with torch.no_grad():
        X_batch, _ = next(iter(train_dl))
        model(X_batch.to(device))
    for h in hooks:
        h.remove()

    # Compute Wanda scores, normalize per row
    layers_info = []
    with torch.no_grad():
        for name, layer, mask_name in model.get_prunable_layers():
            x_n = x_norms.get(name, torch.ones(layer.weight.size(1) if layer.weight.dim() > 1 else 1, device=device))
            S = _wanda_scores(layer.weight, x_n.to(device))
            S_norm = _normalize_uniform(S)
            lam = layer.weight.size(1) if layer.weight.dim() >= 2 else 1
            layers_info.append({
                'name': name, 'mask_name': mask_name, 'layer': layer,
                'S_norm': S_norm.detach(), 'lambda': lam, 'n_out': S_norm.size(0),
                'shape': layer.weight.shape
            })

    # Learnable thresholds (one per output channel)
    thresholds = nn.ParameterList()
    for info in layers_info:
        t = nn.Parameter(torch.full((info['n_out'],), sparsity, device=info['S_norm'].device))
        thresholds.append(t)

    for p in model.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(thresholds.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    target_active = sum(int(info['S_norm'].numel() * (1 - sparsity)) for info in layers_info)

    # §3 Frozen calibration coreset
    calib_X, calib_y = next(iter(train_dl))
    calib_X, calib_y = calib_X.to(device), calib_y.to(device)

    for _ in range(n_iters):
        opt.zero_grad()
        soft_masks = []
        for i, info in enumerate(layers_info):
            s = info['S_norm']
            t = thresholds[i]
            lam = info['lambda']
            t_exp = t.view(-1, 1, 1, 1) if s.dim() == 4 else t.view(-1, 1)
            soft_masks.append(torch.sigmoid(lam * (s - t_exp)))

        for i, (_, _, mask_name) in enumerate(model.get_prunable_layers()):
            setattr(model, mask_name, soft_masks[i])

        model.train()
        L_task = crit(model(calib_X), calib_y)
        N_M = sum(m.sum() for m in soft_masks)
        L_reg = torch.relu(torch.log(N_M / target_active + 1e-8))
        loss = L_task + lambda_reg * L_reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            for t_param in thresholds:
                t_param.clamp_(0.0, 1.0)

    # §3 Exact Top-K Projection
    with torch.no_grad():
        all_soft = []
        for i, info in enumerate(layers_info):
            s = info['S_norm']
            t = thresholds[i]
            lam = info['lambda']
            t_exp = t.view(-1, 1, 1, 1) if s.dim() == 4 else t.view(-1, 1)
            all_soft.append((torch.sigmoid(lam * (s - t_exp)), info['mask_name']))

        all_flat = torch.cat([s.view(-1) for s, _ in all_soft])
        n_keep = max(1, int(all_flat.numel() * (1 - sparsity)))
        threshold = torch.topk(all_flat, n_keep).values[-1].item()
        for soft, mask_name in all_soft:
            getattr(model, mask_name).copy_((soft >= threshold).float())

    return model, time.process_time() - t0


@register('softmask', 'SoftMask', '#9467bd')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    device = config.get('device', 'cpu')
    n_iters = config.get('softmask_iters', 30)

    model, t_soft = _run_softmask(teacher_state, sp, seed, train_dl, device, n_iters=n_iters)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}

    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model_fresh = create_model()
    model_fresh.load_state_dict(state)
    model_fresh.to(device)

    train_finetune_micro(model_fresh, train_dl, config.get('finetune_batches_softmask', 20))
    _, f1, _, _ = evaluate_full(model_fresh, test_dl)
    return {'F1': f1}
