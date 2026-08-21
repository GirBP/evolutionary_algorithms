# OBSE: Orthogonal Block-Swarm Evolution
# Magnitude pruning + per-layer LR evolution via gradient cosine similarity
import math
import random
import torch
import torch.nn as nn
from methods import register
from or08_01 import create_model, set_seed, evaluate_full, check_mask_connectivity
from methods.magnitude import apply_global


class OBSEOptimizer:
    """Layer-wise LR evolution driving gradient orthogonality."""

    def __init__(self, layers, init_lr=0.01):
        self.layers = list(layers)  # list of nn.Parameter
        self.etas = torch.full((len(self.layers),), init_lr)
        self.prev_g = [None] * len(self.layers)
        self.step_cnt = 0

    @torch.no_grad()
    def zero_grad(self):
        for p in self.layers:
            if p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self):
        fitness = torch.zeros(len(self.layers))

        for i, p in enumerate(self.layers):
            if p.grad is None:
                continue
            g_t = p.grad.view(-1)

            if self.prev_g[i] is not None and g_t.norm() > 1e-8 and self.prev_g[i].norm() > 1e-8:
                cos_sim = torch.nn.functional.cosine_similarity(
                    g_t.unsqueeze(0), self.prev_g[i].unsqueeze(0)
                ).item()
                fitness[i] = -abs(cos_sim)  # 0 is optimal (orthogonal)

            p.data.add_(g_t.view_as(p), alpha=-self.etas[i].item())
            self.prev_g[i] = g_t.clone()

        self.step_cnt += 1
        if self.step_cnt % 5 == 0 and len(self.layers) >= 3:
            # Symbiotic evolution: bottom 30% inherit from top 30%
            sort_idx = fitness.argsort(descending=True)  # best first (closest to 0)
            k = max(1, len(self.layers) // 3)
            elites = sort_idx[:k]
            losers = sort_idx[-k:]

            for loser in losers:
                parent_idx = elites[torch.randint(0, k, (1,)).item()]
                new_eta = self.etas[parent_idx].item() * math.exp(random.gauss(0, 0.2))
                self.etas[loser] = max(1e-6, min(1.0, new_eta))


def finetune_obse(model, loader, batches=25):
    """Finetuning with OBSE optimizer."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = OBSEOptimizer(params, init_lr=0.01)
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


@register('obse', 'OBSE', '#bcbd22')
def run(teacher_state, sp, seed, config, train_dl, test_dl):
    set_seed(seed)
    model = create_model()
    model.load_state_dict(teacher_state)
    apply_global(model, sp)
    if not check_mask_connectivity(model):
        return {'F1': 0.333}
    finetune_obse(model, train_dl, config['finetune_batches'])
    _, f1, _, _ = evaluate_full(model, test_dl)
    return {'F1': f1}
