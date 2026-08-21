#!/usr/bin/env python3
# ex08_visualize_winners.py — Візуалізація методів-переможців Ex08

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA_DIR = Path('/Users/bibo/Desktop/cs_dev/Ex08/data')
OUT_DIR  = Path('/Users/bibo/Desktop/cs_dev/Ex08/figs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 10 методів-переможців з Ex08 (зафіксовано в 2.3.4.md та таблицях) + оригінальний SET
METHODS = {
    'tesa26':        ('TESA-26 (авт.)',     '#e41a1c', 'o', 2.5),
    'evo-synflow-v2':('EvoSynFlow',         '#ff7f0e', 'v', 1.5),
    'ehta':          ('E-HTA',              '#2ca02c', '^', 1.5),
    'fes-nsde':      ('FES-NSDE',           '#9467bd', 's', 1.5),
    'acde':          ('ACDE',               '#8c564b', 'p', 1.5),
    'set-v2':        ('SET-v2',             '#e377c2', 'h', 2.0),
    'set':           ('SET (Оригінал)',     '#7f7f7f', '8', 1.5),  # Додано на запит
    'ria':           ('RIA (SOTA)',         '#17becf', 'D', 2.0),
    'wanda-sota':    ('WANDA (SOTA)',       '#ffbb78', 'X', 1.5),
    'sparsegpt':     ('SparseGPT',          '#98df8a', 'd', 1.5),
    'magnitude':     ('Magnitude (Базлайн)','#1f77b4', '*', 1.5),
}

def load(key):
    p = DATA_DIR / f'results_{key}.json'
    if not p.exists():
        # Fallbacks for naming conventions if needed
        return {}
        
    try:
        data = json.loads(p.read_text())
        rows = data.get('results', [])
        
        by_sp = {}
        for r in rows:
            # Group by Sparsity
            sp = r['Sparsity']
            # We want to average F1 across whatever dataset/seed combination exists here
            by_sp.setdefault(sp, []).append(r['F1'])
            
        return {float(sp): np.mean(vs) for sp, vs in by_sp.items()}
    except Exception as e:
        print(f"Error loading {key}: {e}")
        return {}

def main():
    data = {k: load(k) for k in METHODS}
    
    # Check what loaded
    loaded_methods = {k: v for k, v in data.items() if len(v) > 0}
    if not loaded_methods:
        print("No data found!")
        return
        
    print(f"Loaded {len(loaded_methods)} out of {len(METHODS)} methods.")

    # Apply research style
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'lines.linewidth': 2.5, 'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

    fig, ax = plt.subplots(figsize=(14, 8))

    for key, (label, color, marker, lw) in METHODS.items():
        if key not in loaded_methods:
            continue
            
        method_data = loaded_methods[key]
        
        # Відфільтровуємо лише значення >= 0.70
        sp_vals = [sp for sp in sorted(method_data.keys()) if sp >= 0.70]
        f1_vals = [method_data[sp] for sp in sp_vals]
        
        # Highlight our methods vs baseline vs literature SOTA
        alpha = 1.0 if 'авт' in label or 'SOTA' in label else 0.8
        zorder = 5 if 'TESA' in label else (4 if 'SOTA' in label else 3)
        
        ax.plot(sp_vals, f1_vals,
                color=color, marker=marker, linewidth=lw,
                markersize=8 if 'TESA' in label else 7,
                label=label, zorder=zorder, alpha=alpha)

    # Thresholds and annotations
    ax.axhline(0.35, color='gray', lw=1.5, ls='--', alpha=0.7, label='Поріг колапсу мережі')

    ax.set_xlabel('Рівень розрідженості (sp)', color='black', fontsize=14)
    ax.set_ylabel('F1-міра (macro-weighted)', color='black', fontsize=14)
    ax.set_title('Ex08 — Порівняння Топ-10 методів-переможців (high sparsity)', 
                 color='black', fontsize=16, fontweight='bold')

    ax.tick_params(colors='black', labelsize=12)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    
    # Restrict X limits to 70% and above
    ax.set_xlim(0.69, 0.985)
    ax.set_ylim(0.20, 0.90)

    legend = ax.legend(loc='lower left', framealpha=0.9,
                       facecolor='white',
                       edgecolor='gray', fontsize=12, ncol=2)

    plt.tight_layout()
    p1 = OUT_DIR / 'ex08_winners_f1_vs_sparsity.png'
    plt.savefig(p1, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'Saved plot to: {p1}')

if __name__ == '__main__':
    main()
