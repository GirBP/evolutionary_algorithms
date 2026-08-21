# PSMR: Predictive Shadow-Momentum Routing
# Magnitude pruning + evolutionary momentum buffer selection
import math
import random
import torch
import torch.nn as nn
from methods import register
from or08_01 import create_model, set_seed, evaluate_full, check_mask_connectivity
from methods.magnitude import apply_global


class PSMROptimizer:
    """Evolves population of (beta, eta) momentum agents."""

    def __init__(self, params, pop_size=8):
        self.params = [p for p in params if p.requires_grad]
        total = sum(p.numel() for p in self.params)
        self.pop = [
            {'beta': 0.5 + 0.4 * random.random(), 'eta': 0.01 * math.exp(random.gauss(0, 0.3))}
            for _ in range(pop_size)
        ]
        self.pop[0] = {'beta': 0.9, 'eta': 0.01}  # standard momentum baseline
        self.buffers = [torch.zeros(total) for _ in range(pop_size)]

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

        # 1. Evaluate alignment of PAST momentum with CURRENT gradient
        fitness = torch.tensor([torch.dot(m, g_t).item() for m in self.buffers])
        best_idx = fitness.argmax().item()

        # 2. Update all buffers
        for i in range(len(self.pop)):
            beta = self.pop[i]['beta']
            self.buffers[i] = beta * self.buffers[i] + (1 - beta) * g_t

        # 3. Apply winning step
        best_eta = self.pop[best_idx]['eta']
        best_m = self.buffers[best_idx]
        idx = 0
        for p in self.params:
            if p.grad is not None:
                numel = p.numel()
                p.data.add_(best_m[idx:idx + numel].view_as(p), alpha=-best_eta)
                idx += numel

        # 4. Evolve: replace bottom half with mutated top half
        sort_idx = fitness.argsort(descending=True).tolist()
        half = len(self.pop) // 2
        for i in range(half, len(self.pop)):
            loser = sort_idx[i]
            winner = sort_idx[i - half]
            new_beta = self.pop[winner]['beta'] + random.gauss(0, 0.1)
            new_beta = max(0.0, min(0.99, new_beta))
            new_eta = self.pop[winner]['eta'] * math.exp(random.gauss(0, 0.2))
            new_eta = max(1e-6, min(1.0, new_eta))
            self.pop[loser] = {'beta': new_beta, 'eta': new_eta}
            self.buffers[loser].copy_(self.buffers[winner])


def finetune_psmr(model, loader, batches=25):
    """Finetuning with PSMR optimizer."""
    opt = PSMROptimizer(model.parameters(), pop_size=8)
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


@register('psmr', 'PSMR', '#17becf')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_global(model, sp)
    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    finetune_psmr(model, train_dl, config['finetune_batches'])
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
