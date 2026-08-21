#!/usr/bin/env python3
# ex08_visualize_v2.py — Тимчасові графіки з ATSE-CMA доданим до всіх попередніх методів
# Зберігає у Ex08/figs/v2/ — НЕ перезаписує оригінальні графіки у Ex08/figs/

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import seaborn as sns

ROOT_DIR = Path('/Users/bibo/Desktop/cs_dev/Ex08')
DATA_DIR = ROOT_DIR / 'data'
OUT_DIR  = ROOT_DIR / 'figs' / 'v2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({'lines.linewidth': 2.5, 'font.size': 12,
                     'axes.labelsize': 14, 'axes.titlesize': 16})

# Топ-10 методів-переможців + SET (Оригінал) + ATSE-CMA
METHODS = {
    'atse_cma':      ('ATSE-CMA (авт.)',    '#1b7837', 'D', 2.5),   # авторський
    'atse_cma_v2':   ('ATSE-CMA v2 (авт.)', '#7b3294', 'P', 2.5),  # авторський v2
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

# Людські заголовки для кожного датасету
TITLES = {
    'root':    'SimpleMLP (Набір: Moons)',
    'blobs':   'SimpleMLP (Набір: Blobs)',
    'circles': 'SimpleMLP (Набір: Circles)',
    'spirals': 'SimpleMLP (Набір: Spirals)',
    'cnn':     'CompactCNN (Набір: CIFAR-like/Custom)',
    'resnet':  'ResNet18 (Standard)',
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
                data[key] = {float(sp): (np.mean(vs), np.std(vs))
                             for sp, vs in by_sp.items()}
            except Exception:
                pass
    return data


def plot_for_directory(dir_name, dir_path):
    loaded = load_data(dir_path)
    if not loaded:
        return

    has_atse = 'atse_cma' in loaded
    print(f"[{dir_name}] Loaded {len(loaded)} methods. ATSE-CMA: {'✅' if has_atse else '❌ MISSING'}")

    fig, ax = plt.subplots(figsize=(14, 8))

    for key, (label, color, marker, lw) in METHODS.items():
        if key not in loaded:
            continue

        method_data = loaded[key]
        sp_vals = sorted(sp for sp in method_data if sp >= 0.70)
        if not sp_vals:
            continue

        f1_vals = [method_data[sp][0] for sp in sp_vals]
        f1_stds = [method_data[sp][1] for sp in sp_vals]

        is_author = 'авт.' in label
        is_sota   = 'SOTA' in label
        alpha  = 1.0 if (is_author or is_sota) else 0.8
        zorder = 6 if 'ATSE' in label else (5 if 'TESA' in label else
                 (4 if is_sota else 3))
        msize  = 9 if is_author else 7

        ax.plot(sp_vals, f1_vals,
                color=color, marker=marker, linewidth=lw,
                markersize=msize, label=label, zorder=zorder, alpha=alpha)

        # Тінь розкиду по seed-ах
        fill_alpha = 0.18 if is_author else (0.10 if is_sota else 0.05)
        ax.fill_between(sp_vals,
                        np.array(f1_vals) - np.array(f1_stds),
                        np.array(f1_vals) + np.array(f1_stds),
                        color=color, alpha=fill_alpha, zorder=zorder - 1)

    ax.axhline(0.35, color='gray', lw=1.5, ls='--', alpha=0.7,
               label='Поріг колапсу мережі')

    ax.set_xlabel('Рівень розрідженості (sp)', fontsize=14)
    ax.set_ylabel('F1-міра (macro-weighted)', fontsize=14)

    suffix = TITLES.get(dir_name, f'Набір: {dir_name.capitalize()}')
    ax.set_title(rf'Ex08 — Топ-12 методів (+ ATSE-CMA) | {suffix} | $\geq 70\%$',
                 fontsize=16, fontweight='bold')

    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
    ax.set_xlim(0.69, 0.985)

    all_f1 = [v[0] for md in loaded.values() for sp, v in md.items() if sp >= 0.70]
    if all_f1:
        y_min = max(0.0, min(all_f1) - 0.05)
        ax.set_ylim(min(y_min, 0.3), 1.0)

    ax.legend(loc='lower left', framealpha=0.9, facecolor='white',
              edgecolor='gray', fontsize=11, ncol=2)

    plt.tight_layout()
    out_file = OUT_DIR / f'ex08_v2_{dir_name}.png'
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {out_file}")


# ── Cost vs F1 Pareto (v2) ────────────────────────────────────────────────────
def plot_pareto_v2():
    all_methods = list(METHODS.keys())
    results = {m: {'f1': [], 'rcu': []} for m in all_methods}

    dirs = [DATA_DIR] + [d for d in DATA_DIR.iterdir()
                         if d.is_dir() and d.name not in ['base_models', '__pycache__']]

    for d_path in dirs:
        for m in all_methods:
            p = d_path / f'results_{m}.json'
            if p.exists():
                try:
                    data = json.loads(p.read_text()).get('results', [])
                    results[m]['f1'].extend(
                        r['F1'] for r in data if float(r['Sparsity']) >= 0.90)
                    results[m]['rcu'].extend(
                        r['Time_RCU'] for r in data if 'Time_RCU' in r)
                except Exception:
                    pass

    stats = {}
    for m, d in results.items():
        if d['f1'] and d['rcu']:
            stats[m] = (np.mean(d['rcu']), np.mean(d['f1']))

    # Pareto frontier
    pts = sorted(stats.items(), key=lambda x: x[1][0])
    pf_x, pf_y, cur_max = [], [], -1e9
    for m, (rcu, f1) in pts:
        if f1 > cur_max:
            pf_x.append(rcu); pf_y.append(f1); cur_max = f1

    fig, ax = plt.subplots(figsize=(12, 8))
    if pf_x:
        ax.plot(pf_x, pf_y, color='gray', ls='--', alpha=0.5, zorder=1,
                label='Pareto Frontier')

    for key, (label, color, marker, _) in METHODS.items():
        if key not in stats:
            continue
        rcu, f1 = stats[key]
        is_author = 'авт.' in label
        ax.scatter(rcu, f1, s=130 if is_author else 90,
                   c=color, marker=marker, label=label,
                   alpha=1.0 if is_author else 0.75,
                   edgecolors='black' if is_author else 'none', zorder=5)
        if is_author or 'SET' in label or '(SOTA)' in label or 'Magnitude' in label:
            ax.text(rcu * 1.05, f1, label.replace(' (авт.)', '').replace(' (Оригінал)', ''),
                    fontsize=10, va='center', zorder=6)

    ax.set_xscale('log')
    ax.set_xlabel('Вартість (Time RCU) [Log Scale]', fontsize=14)
    ax.set_ylabel(r'Середня F1 ($\geq 90\%$ sp)', fontsize=14)
    ax.set_title('Ex08 v2 — Cost vs Performance (Pareto Frontier) + ATSE-CMA',
                 fontsize=16, fontweight='bold')
    ax.grid(True, which='both', ls='-', alpha=0.2)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.80, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9, fontsize=11)

    out_file = OUT_DIR / 'ex08_v2_cost_vs_f1_pareto.png'
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Pareto saved → {out_file}")


def main():
    print(f"\nBuilding v2 plots → {OUT_DIR}\n")

    # Per-dataset F1 vs Sparsity
    plot_for_directory('root', DATA_DIR)
    for item in DATA_DIR.iterdir():
        if item.is_dir() and item.name not in ['base_models', '__pycache__']:
            plot_for_directory(item.name, item)

    # Cost vs F1 Pareto
    plot_pareto_v2()

    print(f"\n✅ All v2 plots saved to {OUT_DIR}")
    print("   Original plots in Ex08/figs/ are UNTOUCHED.")


if __name__ == '__main__':
    main()
