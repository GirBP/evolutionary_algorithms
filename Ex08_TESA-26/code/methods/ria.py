# Wrapper: RIA SOTA method for Ex08
import torch
import torch.nn as nn
from methods import register
from methods._forward_wrapper import PlainForwardWrapper
from or08_01 import (create_model, train_finetune_micro, evaluate_full, set_seed)


@register('ria', 'RIA (SOTA)', '#4daf4a')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    model.eval()
    
    layers = model.get_prunable_layers()
    activation_norms = {}
    hooks = []
    
    def make_hook(name):
        def hook(m, inp, out):
            x = inp[0].detach()
            # Flatten spatial dims for conv layers: [B, C, H, W] -> [B*H*W, C]
            if x.dim() == 4:
                x = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            elif x.dim() == 3:
                x = x.view(-1, x.size(-1))
            # RIA використовує L1 норму
            norm = torch.norm(x, p=1, dim=0)
            if name in activation_norms:
                activation_norms[name] = (activation_norms[name] + norm) / 2.0
            else:
                activation_norms[name] = norm
        return hook
    
    for name, layer, _ in layers:
        hooks.append(layer.register_forward_hook(make_hook(name)))
    
    wrapper = PlainForwardWrapper(model)
    with torch.no_grad():
        for X, _ in train_dl:
            wrapper(X)
            break
    
    for h in hooks:
        h.remove()
    
    # Global unstructured pruning with Relative Importance
    with torch.no_grad():
        all_scores = []
        all_meta = []
        for name, layer, mask_name in layers:
            W = layer.weight.data
            x_norm = activation_norms[name]
            
            # Flatten weight for global scoring
            W_flat = W.view(W.size(0), -1)
            row_sum = torch.sum(torch.abs(W_flat), dim=1, keepdim=True) + 1e-12
            RI = torch.abs(W_flat) / row_sum
            
            if W.dim() == 4:  # Conv2d
                # x_norm is [in_ch], need to tile for kernel
                kH, kW = W.shape[2], W.shape[3]
                x_norm_tiled = x_norm.repeat(kH * kW)
                score_flat = RI * x_norm_tiled.unsqueeze(0)
            else:
                score_flat = RI * x_norm.unsqueeze(0)
            
            all_scores.append(score_flat.view(-1))
            all_meta.append((name, layer, mask_name, score_flat.view(W.shape)))
        
        # Global threshold
        all_flat = torch.cat(all_scores)
        n_keep = max(1, int(all_flat.numel() * (1.0 - sp)))
        thresh = torch.topk(all_flat, n_keep).values[-1]
        
        for name, layer, mask_name, score in all_meta:
            mask = (score >= thresh).float()
            getattr(model, mask_name).copy_(mask)
    
    train_finetune_micro(model, train_dl, config.get('finetune_batches', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
