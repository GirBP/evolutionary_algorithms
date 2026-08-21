# Wrapper: SparseGPT SOTA method for Ex08
import torch
import torch.nn as nn
from methods import register
from methods.sota_sparsegpt import prune_sparsegpt_layer
from methods._forward_wrapper import PlainForwardWrapper
from or08_01 import (create_model, train_finetune_micro, evaluate_full, set_seed)


@register('sparsegpt', 'SparseGPT (SOTA)', '#e41a1c')
def run(teacher_state, sp, seed, config, train_dl, test_dl, **kw):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    
    wrapper = PlainForwardWrapper(model)
    wrapper.eval()
    
    # Збираємо активації для кожного шару
    layers = model.get_prunable_layers()
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook(m, inp, out):
            activations[name] = inp[0].detach().clone()
        return hook
    
    for name, layer, _ in layers:
        hooks.append(layer.register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        X, _ = next(iter(train_dl))
        wrapper(X)
    
    for h in hooks:
        h.remove()
    
    # Прунінг кожного шару з компенсацією
    with torch.no_grad():
        for name, layer, mask_name in layers:
            X_layer = activations[name]
            prune_sparsegpt_layer(layer, X_layer, sp)
            # Оновлюємо маску SimpleMLP відповідно до нулів
            mask = (layer.weight.data != 0).float()
            getattr(model, mask_name).copy_(mask)
    
    train_finetune_micro(model, train_dl, config.get('finetune_batches', 20))
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
