# Shared forward wrapper that works with ANY model architecture (MLP, CNN, ResNet)
# Used by methods that need to bypass mask logic during optimization

import torch
import torch.nn as nn


class PlainForwardWrapper(nn.Module):
    """Wraps any model to bypass mask logic during optimization.
    
    Unlike the old per-method _PlainForwardWrapper that reimplemented forward
    through Linear layers only, this version delegates to model.forward()
    which works correctly for CNN, MLP, and ResNet architectures.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.prunable_modules = [layer for _, layer, _ in model.get_prunable_layers()]

    def forward(self, x):
        return self.model(x)

    def modules(self):
        yield self
        for m in self.prunable_modules:
            yield m

    def parameters(self):
        for m in self.prunable_modules:
            yield m.weight
            if m.bias is not None:
                yield m.bias

    def zero_grad(self, set_to_none=False):
        for m in self.prunable_modules:
            if m.weight.grad is not None:
                m.weight.grad = None
            if m.bias is not None and m.bias.grad is not None:
                m.bias.grad = None

    def eval(self):
        self.model.eval()
        return self

    def train(self, mode=True):
        self.model.train(mode)
        return self
