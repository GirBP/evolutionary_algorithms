#!/usr/bin/env python3
# ex08_visualize_all_datasets.py — Побудова графіка Топ-11 методів для кожної архітектури

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import seaborn as sns

ROOT_DIR = Path('/Users/bibo/Desktop/cs_dev/Ex08')
DATA_DIR = ROOT_DIR / 'data'
OUT_DIR  = ROOT_DIR / 'figs' / 'all_datasets'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'lines.linewidth': 2.5, 'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# 10 методів-переможців + SET
METHODS = {
    'tesa26':        ('TESA-26 (авт.)',     '#e41a1c', 'o', 2.5),
    'evo-synflow-v2':('EvoSynFlow',         '#ff7f0e', 'v', 1.5),
    'ehta':          ('E-HTA',              '#2ca02c', '^', 1.5),
    'fes-nsde':      ('FES-NSDE',           '#9467bd', 's', 1.5),
    'acde':          ('ACDE',               '#8c564b', 'p', 1.5),
    'set-v2':        ('SET-v2',             '#e377c2', 'h', 2.0),
    'set':           ('SET (Оригінал)',     '#7f7f7f', '8', 1.5),
    'ria':           ('RIA (SOTA)',         '#17becf', 'D', 2.0),
    'wanda-sota':    ('WANDA (SOTA)',       '#ffbb78', 'X', 1.5),
    'sparsegpt':     ('SparseGPT',          '#98df8a', 'd', 1.5),
    'magnitude':     ('Magnitude (Базлайн)','#1f77b4', '*', 1.5),
}

# Визначаємо гарні заголовки
TITLES = {
    'root': 'SimpleMLP (Набір: Moons)',
    'blobs': 'SimpleMLP (Набір: Blobs)',
    'circles': 'SimpleMLP (Набір: Circles)',
    'spirals': 'SimpleMLP (Набір: Spirals)',
    'FashionMNIST': 'CompactCNN (Набір: FashionMNIST)',
    'cnn': 'CompactCNN (Набір: CIFAR-like/Custom)',
    'compactresnet': 'CompactResNet (Складний граф)',
    'resnet': 'ResNet18 (Standard)' 
}

def load_data(dir_path):
    data = {}
    for key in METHODS:
        p = dir_path / f'results_{key}.json'
        if p.exists():
            try:
                raw = json.loads(p.read_text()).get('results', [])
                by_sp = {}
                for r in raw:
                    by_sp.setdefault(r['Sparsity'], []).append(r['F1'])
                data[key] = {float(sp): (np.mean(vs), np.std(vs)) for sp, vs in by_sp.items()}
            except Exception:
                pass
    return data

def plot_for_directory(dir_name, dir_path):
    loaded = load_data(dir_path)
    if not loaded:
        return # Skip empty folders

    print(f"[{dir_name}] Loaded {len(loaded)} methods.")
    
    fig, ax = plt.subplots(figsize=(14, 8))

    for key, (label, color, marker, lw) in METHODS.items():
        if key not in loaded:
            continue
            
        method_data = loaded[key]
        sp_vals = [sp for sp in sorted(method_data.keys()) if sp >= 0.70]
        if not sp_vals:
            continue
            
        f1_vals = [method_data[sp][0] for sp in sp_vals]
        f1_stds = [method_data[sp][1] for sp in sp_vals]
        
        alpha = 1.0 if 'авт' in label or 'SOTA' in label else 0.8
        zorder = 5 if 'TESA' in label else (4 if 'SOTA' in label else 3)
        
        ax.plot(sp_vals, f1_vals,
                color=color, marker=marker, linewidth=lw,
                markersize=8 if 'TESA' in label else 7,
                label=label, zorder=zorder, alpha=alpha)
                
        # Додаємо тінь стандартного відхилення
        y_lower = np.array(f1_vals) - np.array(f1_stds)
        y_upper = np.array(f1_vals) + np.array(f1_stds)
        
        fill_alpha = 0.15 if 'авт' in label or 'SOTA' in label else 0.05
        ax.fill_between(sp_vals, y_lower, y_upper, color=color, alpha=fill_alpha, zorder=zorder-1)

    ax.axhline(0.35, color='gray', lw=1.5, ls='--', alpha=0.7, label='Поріг колапсу мережі')

    ax.set_xlabel('Рівень розрідженості (sp)', color='black', fontsize=14)
    ax.set_ylabel('F1-міра (macro-weighted)', color='black', fontsize=14)
    
    title_suffix = TITLES.get(dir_name, f'Набір: {dir_name.capitalize()}')
    ax.set_title(f'Ex08 — Топ-11 методів | {title_suffix} | Розрідженість $\geq 70\%$', 
                 color='black', fontsize=16, fontweight='bold')

    ax.tick_params(colors='black', labelsize=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    
    ax.set_xlim(0.69, 0.985)
    # y-axis auto adjusting based on dataset performance
    all_f1 = [v[0] for md in loaded.values() for sp, v in md.items() if sp >= 0.70]
    if all_f1:
        y_min = max(0.0, min(all_f1) - 0.05)
        # some datasets collapse to 0.1 on 10-class problems, not 0.33
        ax.set_ylim(min(y_min, 0.3), 1.0) 

    legend = ax.legend(loc='lower left', framealpha=0.9,
                       facecolor='white', edgecolor='gray', 
                       fontsize=12, ncol=2)

    plt.tight_layout()
    out_file = OUT_DIR / f'ex08_winners_{dir_name}.png'
    plt.savefig(out_file, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved -> {out_file}")

def main():
    # Process root (moons)
    plot_for_directory('root', DATA_DIR)
    
    # Process subdirectories
    for item in DATA_DIR.iterdir():
        if item.is_dir() and item.name not in ['base_models', '__pycache__']:
            plot_for_directory(item.name, item)

if __name__ == '__main__':
    main()
