# TSPE: Taylor-Surrogate Proximal Evolution
# Magnitude pruning + evolutionary LR via second-order Taylor surrogate
import torch
import torch.nn as nn
from methods import register
from or08_01 import create_model, set_seed, evaluate_full, check_mask_connectivity
from methods.magnitude import apply_global


class TSPEOptimizer:
    """(1+λ)-ES over learning rates using BB secant curvature."""

    def __init__(self, params, base_lr=0.01, pop_size=10):
        self.params = [p for p in params if p.requires_grad]
        self.pop_eta = torch.full((pop_size,), base_lr)
        self.prev_theta = None
        self.prev_g = None

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return
        g_t = torch.cat([g.view(-1) for g in grads])
        theta_t = torch.cat([p.data.view(-1) for p in self.params if p.grad is not None])

        # BB secant curvature
        C_t = 1e-4
        if self.prev_g is not None:
            s_t = theta_t - self.prev_theta
            y_t = g_t - self.prev_g
            denom = torch.dot(s_t, s_t) + 1e-8
            C_t = torch.clamp(torch.abs(torch.dot(s_t, y_t)) / denom, min=1e-4).item()

        # Surrogate fitness
        g_norm_sq = torch.dot(g_t, g_t).item()
        fitness = -(self.pop_eta * g_norm_sq) + 0.5 * C_t * (self.pop_eta ** 2) * g_norm_sq

        best_eta = self.pop_eta[fitness.argmin()].item()

        # Apply physical step
        idx = 0
        for p in self.params:
            if p.grad is not None:
                numel = p.numel()
                p.data.add_(g_t[idx:idx + numel].view_as(p), alpha=-best_eta)
                idx += numel

        # (1+λ)-ES mutation
        self.pop_eta = best_eta * torch.exp(torch.randn(len(self.pop_eta)) * 0.2)
        self.pop_eta[0] = best_eta  # elitism
        self.pop_eta = torch.clamp(self.pop_eta, 1e-6, 1.0)
        self.prev_theta = theta_t.clone()
        self.prev_g = g_t.clone()


def finetune_tspe(model, loader, batches=25):
    """Finetuning with TSPE optimizer."""
    opt = TSPEOptimizer(model.parameters(), base_lr=0.01, pop_size=10)
    crit = nn.CrossEntropyLoss()
    model.train()
    iterator = iter(loader)
    for _ in range(batches):
        try:
            X, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            X, y = next(iterator)
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        with torch.no_grad():
            for _, layer, m_name in model.get_prunable_layers():
                if layer.weight.grad is not None:
                    layer.weight.grad.mul_(getattr(model, m_name))
        opt.step()


@register('tspe', 'TSPE', '#e377c2')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_global(model, sp)
    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    finetune_tspe(model, train_dl, config['finetune_batches'])
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
