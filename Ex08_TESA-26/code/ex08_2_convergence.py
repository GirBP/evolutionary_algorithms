#!/usr/bin/env python3
# ex08_2_convergence.py — Відстеження траєкторії оптимізації
# Порівнює зміну F1-міри ітеративних методів (TESA-26 та DSA) шар за шаром.

import sys, os, copy
from pathlib import Path

# Paths to use exact same env as Ex08
ROOT = Path('/Users/bibo/Desktop/cs_dev')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Ex08' / 'code'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import cma

from or08_01 import set_seed, set_model_class, create_model, get_dataloaders
from methods.tesa26 import _compute_saliency_multi, _masks_from_k, normalize_genome
from methods.dsa import _compute_taylor_saliency, _project_to_budget

torch.set_num_threads(1)
OUT_DIR = ROOT / 'Ex08_1' / 'figs'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_f1_proxy(model, test_dl):
    """Швидка оцінка F1 без мікро-файнтюнінгу для відстеження."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in test_dl:
            preds.extend(model(X).argmax(dim=1).numpy())
            labels.extend(y.numpy())
    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average='macro')


def track_tesa26(teacher_state, sp, train_dl, test_dl, max_evals=40):
    model = create_model()
    model.load_state_dict(teacher_state)
    layers = model.get_prunable_layers()
    layer_counts = np.array([l.weight.data.numel() for _, l, _ in layers])
    
    # Init
    saliency = _compute_saliency_multi(model, train_dl, n_batches=3)
    x0 = np.array([getattr(model, mn).sum().item() / getattr(model, mn).numel() 
                   for _, _, mn in layers])
    sigma0 = 0.15 * (1.0 - sp)
    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'popsize': 8, 'verbose': -1, 'bounds': [0, None]
    })

    history = []
    
    # ── Fitness proxy для відстеження (те, що бачить CMA) ──
    from methods.tesa26 import _evaluate_fitness
    
    evals = 0
    recalibrated = False
    best_genome = x0.copy()
    
    # Record initial state
    history.append((0, 0.0))

    while not es.stop() and evals < max_evals:
        if not recalibrated and evals >= max_evals // 2:
            recalibrated = True
            ratios = normalize_genome(best_genome, layer_counts, sp)
            masks = _masks_from_k(saliency, ratios)
            for m_meta, mask in zip(layers, masks):
                getattr(model, m_meta[2]).copy_(mask)
            saliency = _compute_saliency_multi(model, train_dl, 3)
            model.load_state_dict(teacher_state)

        solutions = es.ask()
        fitnesses = [_evaluate_fitness(s, layer_counts, sp, saliency) for s in solutions]
        es.tell(solutions, fitnesses)
        evals += len(solutions)

        # Evaluate best so far
        best_idx = np.argmin(fitnesses)
        cand_genome = solutions[best_idx]
        if fitnesses[best_idx] < _evaluate_fitness(best_genome, layer_counts, sp, saliency):
            best_genome = cand_genome.copy()

        # Заміряємо реальну F1-міру для найкращого геному цiєї ітерації
        with torch.no_grad():
            ratios = normalize_genome(best_genome, layer_counts, sp)
            masks = _masks_from_k(saliency, ratios)
            for m_meta, mask in zip(layers, masks):
                getattr(model, m_meta[2]).copy_(mask)
            f1 = evaluate_f1_proxy(model, test_dl)
            history.append((evals, f1))
            model.load_state_dict(teacher_state)

    return history


def track_dsa(teacher_state, sp, train_dl, test_dl, steps=20):
    model = create_model()
    model.load_state_dict(teacher_state)
    layers = model.get_prunable_layers()
    n_params = [l.weight.data.numel() for _, l, _ in layers]
    target_params = (1.0 - sp) * sum(n_params)

    sensitivity = _compute_taylor_saliency(model, train_dl, n_batches=3)
    s_arr = np.array(sensitivity, dtype=np.float64)
    s_norm = s_arr / (s_arr.max() + 1e-12)

    ratios = [(1.0 - sp)] * len(layers)
    history = []
    
    alpha = 0.05
    for step in range(steps):
        # Apply current state
        with torch.no_grad():
            for m_meta, kr in zip(layers, ratios):
                W = m_meta[1].weight.data
                w_abs = W.abs().view(-1)
                n_keep = max(1, int(round(kr * w_abs.numel())))
                if n_keep < w_abs.numel():
                    thresh = torch.topk(w_abs, n_keep).values[-1]
                    mask = (W.abs() >= thresh).float()
                else:
                    mask = torch.ones_like(W)
                getattr(model, m_meta[2]).copy_(mask)
            f1 = evaluate_f1_proxy(model, test_dl)
            history.append((step, f1))
            model.load_state_dict(teacher_state) # res

        # Gradient step
        r_arr = np.array(ratios, dtype=np.float64)
        r_arr = r_arr + alpha * s_norm
        ratios = _project_to_budget(r_arr.tolist(), n_params, target_params)
        alpha = 0.05 * (1.0 + np.cos(np.pi * step / steps)) / 2.0

    return history


def main():
    print("Initializing...")
    set_model_class('SimpleMLP')
    set_seed(42)
    sp = 0.95
    
    # Load Teacher
    teacher_cache = ROOT / 'Ex08_1' / 'data' / 'base_models' / 'teacher_SimpleMLP_seed42.pt'
    teacher_state = torch.load(teacher_cache, map_location='cpu', weights_only=True)
    
    train_dl, _, test_dl = get_dataloaders(42, 'moons')

    # Baseline: Magnitude directly (no iterations, horizontal line)
    print("Computing Magnitude baseline...")
    from methods.magnitude import run as run_mag
    # Just manual apply to get proxy F1
    model = create_model()
    model.load_state_dict(teacher_state)
    from or08_01 import apply_global
    apply_global(model, sp)
    f1_mag = evaluate_f1_proxy(model, test_dl)

    print("Tracking TESA-26...")
    tesa_hist = track_tesa26(teacher_state, sp, train_dl, test_dl, max_evals=40)
    
    print("Tracking DSA...")
    # Scale DSA steps to visually match TESA's eval count (0 -> 40)
    dsa_hist = track_dsa(teacher_state, sp, train_dl, test_dl, steps=20)
    dsa_hist_scaled = [(step * 2, f1) for step, f1 in dsa_hist]

    # Plot
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#0f0f0f')
    ax.set_facecolor('#1a1a1a')

    # TESA
    tx, ty = zip(*tesa_hist)
    ax.plot(tx, ty, label='TESA-26 (авт.)', color='#e41a1c', marker='o', lw=2.5)
    
    # DSA
    dx, dy = zip(*dsa_hist_scaled)
    ax.plot(dx, dy, label='DSA', color='#9467bd', marker='s', lw=2.0)
    
    # Magnitude
    ax.axhline(f1_mag, label='Magnitude (Базлайн)', color='#1f77b4', ls='--', lw=2.0)

    # Note: the ISR recalibration point for TESA is at eval=20
    ax.axvline(20, color='#888', ls=':', ymax=0.3)
    ax.text(20, ax.get_ylim()[0] + 0.05, 'Запуск ISR\n(перерахунок)', 
            color='#aaa', fontsize=9, ha='center')

    ax.set_xlabel('Кількість ітерацій (Evaluations/Steps)', color='#ccc')
    ax.set_ylabel('F1-міра (без мікро-файнтюнінгу)', color='#ccc')
    ax.set_title(f'Динаміка збіжності оптимізації (Sparsity={sp}, SimpleMLP)', 
                 color='white', fontweight='bold', fontsize=12)

    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')
    ax.grid(color='#333', lw=0.5)
    
    ax.legend(facecolor='#111', edgecolor='#444', labelcolor='white')
    
    plt.tight_layout()
    out_path = OUT_DIR / 'ex08_2_convergence.png'
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
