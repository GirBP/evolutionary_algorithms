# methods/softmask_grad.py — SoftMask-Grad with Rank-Based Scoring + Exact Top-K
# Fix: replaced abs(W)-t with rank-based scoring to handle small uniform weights
import torch
import torch.nn as nn
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)


@register('softmask-grad', 'SoftMask-Grad', '#7f7f7f')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.train()

    # Rank-based init: convert abs(W) to ranks in [0,1], then init threshold at sp
    rank_cache = {}
    t_params = {}
    for name, layer, _ in model.get_prunable_layers():
        w_flat = layer.weight.abs().view(-1)
        ranks = w_flat.argsort().argsort().float() / (w_flat.numel() - 1)  # [0,1]
        rank_cache[name] = ranks.view_as(layer.weight)
        t_params[name] = torch.tensor(sp, requires_grad=True)  # threshold in rank space

    n_iters = config.get('softmask_iters', 30)
    opt = torch.optim.Adam(list(t_params.values()), lr=config.get('softmask_lr', 5e-3))
    crit = nn.CrossEntropyLoss()
    X_cal, y_cal = next(iter(train_dl))
    lam = 50.0  # sigmoid steepness in rank space (ranks are [0,1], so 50 is moderate)

    for _ in range(n_iters):
        opt.zero_grad()
        for name, layer, mask_name in model.get_prunable_layers():
            t = t_params[name]
            soft_mask = torch.sigmoid(lam * (rank_cache[name] - t))
            getattr(model, mask_name).copy_(soft_mask.detach())

        loss = crit(model(X_cal), y_cal)
        total_w = sum(l.weight.numel() for _, l, _ in model.get_prunable_layers())
        active = sum(
            torch.sigmoid(lam * (rank_cache[n] - t_params[n])).sum()
            for n, _, _ in model.get_prunable_layers()
        )
        sp_hat = 1.0 - active / total_w
        loss = loss + 10.0 * (sp_hat - sp) ** 2
        loss.backward()
        opt.step()

    # Exact Top-K using rank-based scores
    with torch.no_grad():
        scores_list = []
        for name, layer, mask_name in model.get_prunable_layers():
            t = t_params[name].detach()
            score = torch.sigmoid(lam * (rank_cache[name] - t))
            scores_list.append((score, layer, mask_name))

        all_scores = torch.cat([s.view(-1) for s, _, _ in scores_list])
        n_keep = max(1, int(all_scores.numel() * (1.0 - sp)))
        threshold = torch.topk(all_scores, n_keep).values[-1].item()

        for score, layer, mask_name in scores_list:
            hard_mask = (score >= threshold).float()
            layer.weight.data.mul_(hard_mask)
            getattr(model, mask_name).copy_(hard_mask)

    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    train_finetune_micro(model, train_dl, config.get('finetune_batches_softmask', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
