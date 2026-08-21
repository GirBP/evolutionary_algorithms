# methods/softmask_grad_v2.py — SoftMask-Grad with adaptive lambda + median init
import torch
import torch.nn as nn
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)


@register('softmask-grad-v2', 'SoftMask-Grad-v2', '#c5b0d5')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.train()

    # Adaptive lambda: softer at high sparsity
    lam_scale = max(10.0, 100.0 * (1.0 - sp) / 0.5)  # s=0.80→40, s=0.50→100, s=0.20→160(cap)

    # Median init instead of quantile
    t_params = {}
    for name, layer, _ in model.get_prunable_layers():
        t_params[name] = torch.tensor(layer.weight.abs().median().item(), requires_grad=True)

    n_iters = 30 + int(max(0, sp - 0.60) * 100)  # adaptive: s=0.80→50, s=0.50→30
    opt = torch.optim.Adam(list(t_params.values()), lr=config.get('softmask_lr', 5e-3))
    crit = nn.CrossEntropyLoss()
    X_cal, y_cal = next(iter(train_dl))

    for _ in range(n_iters):
        opt.zero_grad()
        for name, layer, mask_name in model.get_prunable_layers():
            t = t_params[name]
            soft_mask = torch.sigmoid(lam_scale * (layer.weight.abs() - t))
            getattr(model, mask_name).copy_(soft_mask.detach())
        loss = crit(model(X_cal), y_cal)
        total_w = sum(l.weight.numel() for _, l, _ in model.get_prunable_layers())
        active = sum(torch.sigmoid(lam_scale * (l.weight.abs() - t_params[n])).sum()
                     for n, l, _ in model.get_prunable_layers())
        loss = loss + 10.0 * ((1.0 - active / total_w) - sp) ** 2
        loss.backward()
        opt.step()

    # Exact Top-K
    with torch.no_grad():
        all_scores = []
        for name, layer, mask_name in model.get_prunable_layers():
            t = t_params[name].detach()
            score = torch.sigmoid(lam_scale * (layer.weight.abs() - t))
            all_scores.append((score, layer, mask_name))
        flat = torch.cat([s.view(-1) for s, _, _ in all_scores])
        n_keep = max(1, int(flat.numel() * (1.0 - sp)))
        thresh = flat.topk(n_keep).values[-1].item()
        for score, layer, mask_name in all_scores:
            mask = (score >= thresh).float()
            layer.weight.data.mul_(mask)
            getattr(model, mask_name).copy_(mask)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    train_finetune_micro(model, train_dl, config.get('finetune_batches_softmask', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
