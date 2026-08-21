#!/usr/bin/env python3
# ex08_cost_vs_f1.py — Побудова діаграми Вартість (RCU) vs Якість (F1)

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_DIR = Path('/Users/bibo/Desktop/cs_dev/Ex08/data')
OUT_DIR  = Path('/Users/bibo/Desktop/cs_dev/Ex08/figs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'lines.linewidth': 2.5, 'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Використовуємо всі доступні методи, а не лише 10 переможців, щоб побачити загальну картину
METHODS_TO_PLOT = {
    'tesa26':        ('TESA-26 (авт.)',     '#e41a1c', 'o', 120),
    'evo-synflow-v2':('EvoSynFlow',         '#ff7f0e', 'v', 90),
    'ehta':          ('E-HTA',              '#2ca02c', '^', 90),
    'fes-nsde':      ('FES-NSDE',           '#9467bd', 's', 90),
    'acde':          ('ACDE',               '#8c564b', 'p', 90),
    'set-v2':        ('SET-v2',             '#e377c2', 'h', 100),
    'set':           ('SET (Оригінал)',     '#7f7f7f', '8', 80),
    'ria':           ('RIA (SOTA)',         '#17becf', 'D', 100),
    'wanda-sota':    ('WANDA (SOTA)',       '#ffbb78', 'X', 90),
    'sparsegpt':     ('SparseGPT',          '#98df8a', 'd', 90),
    'magnitude':     ('Magnitude',          '#1f77b4', '*', 150),
    'lamp':          ('LAMP',               '#888888', 'P', 60),
    'dsa':           ('DSA',                '#bcbd22', '<', 60),
    'vpam':          ('VPAM',               '#17becf', '>', 60)
}

def load_stats():
    stats = {}
    for key in METHODS_TO_PLOT.keys():
        f1_vals = []
        rcu_vals = []
        
        # Обходимо всі датасети (кореневий moons + підпапки)
        dirs = [DATA_DIR] + [d for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in ['base_models', '__pycache__']]
        
        for d_path in dirs:
            p = d_path / f'results_{key}.json'
            if p.exists():
                try:
                    data = json.loads(p.read_text()).get('results', [])
                    for r in data:
                        # Беремо продуктивність на екстремальних рівнях >= 0.90
                        if float(r['Sparsity']) >= 0.90:
                            f1_vals.append(r['F1'])
                        # RCU записуємо завжди, бо його вартість не залежить від осі F1
                        if 'Time_RCU' in r:
                            rcu_vals.append(r['Time_RCU'])
                except Exception:
                    pass
        
        if f1_vals and rcu_vals:
            stats[key] = {
                'f1_mean': np.mean(f1_vals),
                'rcu_mean': np.mean(rcu_vals)
            }
    return stats

def main():
    stats = load_stats()
    if not stats:
        print("No data loaded!")
        return
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Pareto frontier logic (minimize RCU, maximize F1)
    points = [(stats[k]['rcu_mean'], stats[k]['f1_mean'], k) for k in stats]
    points.sort(key=lambda x: x[0])  # sort by RCU asc
    
    pareto_x = []
    pareto_y = []
    current_max_f1 = -float('inf')
    
    for rcu, f1, k in points:
        if f1 > current_max_f1:
            pareto_x.append(rcu)
            pareto_y.append(f1)
            current_max_f1 = f1
            
    # Plot pareto
    if pareto_x:
        ax.plot(pareto_x, pareto_y, color='gray', linestyle='--', alpha=0.5, zorder=1, label='Pareto Frontier')

    # Plot scatters
    for key, data in stats.items():
        label, color, marker, size = METHODS_TO_PLOT[key]
        x = data['rcu_mean']
        y = data['f1_mean']
        
        alpha = 1.0 if 'авт' in label or 'SOTA' in label else 0.7
        edgecolor = 'black' if 'авт' in label else 'none'
        
        ax.scatter(x, y, s=size, c=color, marker=marker, 
                   label=label, alpha=alpha, edgecolors=edgecolor, zorder=5)
                   
        # Анотація деяких ключових точок (TESA, Magnitude, SOTA)
        if 'TESA' in label or 'Magnitude' in label or 'SET' in label or 'RIA' in label:
            x_offset = x * 0.05
            ax.text(x + x_offset, y, label.replace(' (авт.)', '').replace(' (Оригінал)', ''), 
                    fontsize=10, verticalalignment='center', zorder=6)

    ax.set_xscale('log')
    ax.set_xlabel('Вартість пошуку алгоритму (Time RCU) [Log Scale]', color='black', fontsize=14)
    ax.set_ylabel('Середня F1-міра на екстрем. розрідженості ($\geq 90\%$)', color='black', fontsize=14)
    ax.set_title('Ex08 — Аналіз Cost vs Performance (Pareto Frontier)', color='black', fontsize=16, fontweight='bold')
    
    ax.tick_params(colors='black', labelsize=12)
    ax.grid(True, which="both", ls="-", alpha=0.2, color='gray')
    
    # Legend outside
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, fontsize=11)

    out_file = OUT_DIR / 'ex08_cost_vs_f1_pareto.png'
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {out_file}")

if __name__ == '__main__':
    main()
