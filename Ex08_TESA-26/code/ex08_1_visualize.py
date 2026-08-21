#!/usr/bin/env python3
# ex08_1_visualize.py — Візуалізація результатів Ex08.1
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

DATA_DIR = Path('/Users/bibo/Desktop/cs_dev/Ex08_1/data')
OUT_DIR  = Path('/Users/bibo/Desktop/cs_dev/Ex08_1/figs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    'tesa26':   ('TESA-26 (авт.)',  '#e41a1c', 'o', 2.5),
    'lamp':     ('LAMP',            '#ff7f0e', 's', 1.5),
    'erk':      ('ERK',             '#2ca02c', '^', 1.5),
    'dsa':      ('DSA',             '#9467bd', 'D', 1.5),
    'magnitude':('Magnitude',       '#1f77b4', 'v', 1.5),
}

SPARSITIES = [0.50, 0.70, 0.80, 0.85,
              0.90, 0.91, 0.92, 0.93,
              0.94, 0.95, 0.96, 0.97]

# ── Load data ────────────────────────────────────────────────────────────
def load(key):
    p = DATA_DIR / f'results_{key}.json'
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())['results']
    by_sp = {}
    for r in rows:
        by_sp.setdefault(r['Sparsity'], []).append(r['F1'])
    return {sp: np.mean(vs) for sp, vs in by_sp.items()}

data = {k: load(k) for k in METHODS}

# ── Figure 1: F1 vs Sparsity ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')

for key, (label, color, marker, lw) in METHODS.items():
    sp_vals = sorted(data[key].keys())
    f1_vals = [data[key][sp] for sp in sp_vals]
    ax.plot(sp_vals, f1_vals,
            color=color, marker=marker, linewidth=lw,
            markersize=6, label=label,
            zorder=3 if key == 'tesa26' else 2,
            alpha=1.0 if key == 'tesa26' else 0.85)

# Collapse threshold line
ax.axhline(0.35, color='#888', lw=0.8, ls='--', alpha=0.5, label='Поріг колапсу')
ax.axvspan(0.90, 0.97, alpha=0.07, color='#ff4444', label='Екстремальна розрідженість')

ax.set_xlabel('Рівень розрідженості (sp)', color='#ccc', fontsize=12)
ax.set_ylabel('F1-міра (macro-weighted)', color='#ccc', fontsize=12)
ax.set_title('Ex08.1 — TESA-26 vs DSA / LAMP / ERK\n(moons, SimpleMLP, 3 seeds)',
             color='white', fontsize=13, fontweight='bold')

ax.tick_params(colors='#aaa')
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
ax.set_xlim(0.48, 0.98)
ax.set_ylim(0.25, 0.90)
ax.grid(axis='y', color='#333', lw=0.5)
ax.grid(axis='x', color='#2a2a2a', lw=0.3)

legend = ax.legend(loc='lower left', framealpha=0.15,
                   labelcolor='white', facecolor='#111',
                   edgecolor='#444', fontsize=10)

# Annotate TESA-26 stability
tesa_sp = sorted(data['tesa26'].keys())
tesa_f1 = [data['tesa26'][sp] for sp in tesa_sp]
ax.annotate('TESA-26: стабільний\nна всіх рівнях',
            xy=(0.95, np.mean(tesa_f1)), xytext=(0.86, 0.75),
            fontsize=9, color='#ff6666',
            arrowprops=dict(arrowstyle='->', color='#ff6666', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a1a',
                      edgecolor='#ff6666', alpha=0.8))

plt.tight_layout()
p1 = OUT_DIR / 'ex08_1_f1_vs_sparsity.png'
plt.savefig(p1, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {p1}')

# ── Figure 2: Summary bar chart (mean F1) ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')

method_labels = []
means = []
stds  = []
colors_bar = []

for key, (label, color, _, _) in METHODS.items():
    f1s = []
    for sp_data in data[key].values():
        f1s.append(sp_data)
    method_labels.append(label)
    means.append(np.mean(f1s))
    stds.append(np.std(f1s))
    colors_bar.append(color)

x = np.arange(len(method_labels))
bars = ax.bar(x, means, yerr=stds, capsize=5,
              color=colors_bar, edgecolor='#333',
              error_kw={'ecolor': '#888', 'lw': 1.5})

# Highlight TESA-26
bars[0].set_edgecolor('#ff4444')
bars[0].set_linewidth(2)

ax.set_xticks(x)
ax.set_xticklabels(method_labels, color='#ccc', fontsize=10)
ax.set_ylabel('Середня F1-міра (всі рівні)', color='#ccc', fontsize=11)
ax.set_title('Ex08.1 — Середня F1 по методах', color='white', fontsize=12, fontweight='bold')
ax.tick_params(colors='#aaa')
ax.set_ylim(0.3, 0.92)
for spine in ax.spines.values():
    spine.set_edgecolor('#444')
ax.grid(axis='y', color='#333', lw=0.5)

for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.01,
            f'{m:.3f}', ha='center', va='bottom', color='#ccc', fontsize=9)

plt.tight_layout()
p2 = OUT_DIR / 'ex08_1_mean_f1_bar.png'
plt.savefig(p2, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'Saved: {p2}')

print('\nDone. Figures in:', OUT_DIR)
