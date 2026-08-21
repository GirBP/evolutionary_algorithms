# methods/vpam.py — Variance-Penalized Activation Masking (Multi-Fidelity)
import torch
import torch.nn as nn
import numpy as np
from methods import register
from or08_01 import (create_model, train_finetune_micro, evaluate_full,
                     set_seed, check_mask_connectivity)


def _get_activations(model, train_dl):
    """Collect activation norms from single calibration micro-batch (coreset)."""
    hooks, activations = [], {}

    def hook_fn(name):
        def fn(m, inp, out):
            activations[name] = inp[0].detach()
        return fn

    for name, layer, _ in model.get_prunable_layers():
        hooks.append(layer.register_forward_hook(hook_fn(name)))

    model.eval()
    with torch.no_grad():
        X_batch, _ = next(iter(train_dl))
        model(X_batch)

    for h in hooks:
        h.remove()
    return activations


@register('vpam', 'VPAM (ours)', '#ff7f0e')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)

    # Coreset-cached activations
    activations = _get_activations(model, train_dl)

    with torch.no_grad():
        for name, layer, mask_name in model.get_prunable_layers():
            W = layer.weight
            X_act = activations[name]

            if isinstance(layer, nn.Conv2d):
                x_l2 = torch.norm(X_act, p=2, dim=(0, 2, 3))
                x_var = X_act.var(dim=(0, 2, 3))
                asf = x_l2 * torch.exp(-x_var)
                asf_shaped = asf.view(1, -1, 1, 1)
            else:
                x_l2 = torch.norm(X_act, p=2, dim=0)
                x_var = X_act.var(dim=0)
                asf = x_l2 * torch.exp(-x_var)
                asf_shaped = asf.view(1, -1)

            # Relative importance
            w_abs = W.abs()
            row_sum = w_abs.sum(dim=list(range(1, w_abs.ndim)), keepdim=True).clamp(min=1e-8)
            col_dims = [0] + list(range(2, w_abs.ndim))
            col_sum = w_abs.sum(dim=col_dims, keepdim=True).clamp(min=1e-8)
            r = w_abs / row_sum + w_abs / col_sum
            score = r * asf_shaped

            # Adaptive per-row budget: rows with higher importance get more weights
            n_out = score.shape[0]
            flat_per_row = score.view(n_out, -1)
            cols = flat_per_row.shape[1]
            total_keep = max(n_out, int(flat_per_row.numel() * (1 - sp)))

            # Row importance = sum of scores
            row_imp = flat_per_row.sum(dim=1)
            row_imp = row_imp / (row_imp.sum() + 1e-8)  # normalize to [0,1]
            # k_row = importance * total_budget, clamp to [1, cols]
            k_rows = (row_imp * total_keep).clamp(min=1, max=cols).int()
            # Adjust to match total budget exactly
            diff = total_keep - k_rows.sum().item()
            if diff > 0:
                # Add to rows with highest importance
                order = row_imp.argsort(descending=True)
                for idx in order:
                    add = min(diff, cols - k_rows[idx].item())
                    k_rows[idx] += add
                    diff -= add
                    if diff <= 0:
                        break
            elif diff < 0:
                # Remove from rows with lowest importance
                order = row_imp.argsort()
                for idx in order:
                    remove = min(-diff, k_rows[idx].item() - 1)
                    k_rows[idx] -= remove
                    diff += remove
                    if diff >= 0:
                        break

            mask = torch.zeros_like(flat_per_row)
            for row in range(n_out):
                k = k_rows[row].item()
                topk_idx = flat_per_row[row].topk(k).indices
                mask[row, topk_idx] = 1.0
            getattr(model, mask_name).copy_(mask.view_as(W))

    # §1 Topological filter
    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    # §2 Micro-finetuning
    train_finetune_micro(model, train_dl, config['finetune_batches'])
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
