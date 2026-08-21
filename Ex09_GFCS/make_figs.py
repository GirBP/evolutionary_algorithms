#!/usr/bin/env python3
"""
Ex09 GFCS: фігури для публічного репозиторію дисертації (§2.4).

Скрипт лише ЧИТАЄ наявні файли результатів (results/full_benchmark_rcu.json) —
жодні числа не перераховуються експериментально повторно, лише агрегуються
для візуалізації. Це дозволяє будь-кому відтворити фігури без повторного
запуску бенчмарку.

Джерело чисел: results/full_benchmark_rcu.json — 6 методів конверсії
(neuron_removal, svd_compression, knowledge_distill, weight_redistribution,
evomerge, gfcs) × 8 датасетів × 2 seeds = 96 записів.

ΔF1 у цьому файлі визначено як final_f1 − sparse_f1 (якість компактної моделі
відносно розрідженої моделі-донора ДО конверсії, а не відносно вчителя) —
див. ex09_full_benchmark.py:205.

Фігури:
  (a) fig_a_compression_quality_by_method.png — стиснення×якість по 6 методах,
      mean±std ΔF1 та compression, агреговані по 8 датасетах (спершу середнє
      по 2 seeds на датасет, потім mean/std по 8 датасетах).
  (b) fig_b_rcu_conversion_by_method.png — середній RCU конверсії по методах
      (логарифмічна шкала, бо діапазон 0.08–650).
  (c) fig_c_compression_vs_delta_f1_scatter.png — розсіювання «стиснення vs
      ΔF1» для всіх 96 записів, кольори за методом.

Запуск: python3 make_figs.py  (з кореня Ex09_GFCS/)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / 'results'
FIGS_DIR = RESULTS_DIR / 'figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

METHOD_ORDER = ['gfcs', 'neuron_removal', 'svd_compression',
                'knowledge_distill', 'weight_redistribution', 'evomerge']
METHOD_LABELS = {
    'gfcs': 'GFCS',
    'neuron_removal': 'NeuronRemoval',
    'svd_compression': 'SVD',
    'knowledge_distill': 'KD',
    'weight_redistribution': 'WeightRedist',
    'evomerge': 'EvoMerge',
}
BLUE = '#2563eb'
GRAY = '#94a3b8'
DARK = '#1e293b'
PALETTE = ['#2563eb', '#94a3b8', '#16a34a', '#f59e0b', '#dc2626', '#7c3aed']


def load_records():
    with open(RESULTS_DIR / 'full_benchmark_rcu.json', encoding='utf-8') as f:
        data = json.load(f)
    return data['results']


def per_dataset_means(records, method, field):
    """Середнє по 2 seeds для кожного з 8 датасетів окремо."""
    by_ds = defaultdict(list)
    for r in records:
        if r['method'] == method:
            by_ds[r['dataset']].append(r[field])
    return [float(np.mean(vals)) for vals in by_ds.values()]


# ═══ FIG A: стиснення×якість по 6 методах (mean±std по 8 датасетах) ═══
def fig_a(records):
    delta_means, delta_stds = [], []
    comp_means, comp_stds = [], []
    for m in METHOD_ORDER:
        ds_delta = per_dataset_means(records, m, 'delta_f1')
        ds_comp = per_dataset_means(records, m, 'compression')
        assert len(ds_delta) == 8 and len(ds_comp) == 8, f'{m}: очікується 8 датасетів'
        delta_means.append(np.mean(ds_delta)); delta_stds.append(np.std(ds_delta))
        comp_means.append(np.mean(ds_comp)); comp_stds.append(np.std(ds_comp))

    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    colors = [BLUE if m == 'gfcs' else GRAY for m in METHOD_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(labels, delta_means, yerr=delta_stds, capsize=4,
            color=colors, edgecolor='white', linewidth=0.8)
    ax1.axhline(0, color='gray', ls='--', alpha=0.4, linewidth=1)
    ax1.set_ylabel('ΔF1 (компактна − розріджена)')
    ax1.set_title('Зміна якості після конверсії')
    ax1.grid(axis='y', alpha=0.15)
    ax1.tick_params(axis='x', rotation=20)

    ax2.bar(labels, comp_means, yerr=comp_stds, capsize=4,
            color=colors, edgecolor='white', linewidth=0.8)
    ax2.set_ylabel('Коефіцієнт стиснення, разів')
    ax2.set_title('Стиснення параметрів')
    ax2.grid(axis='y', alpha=0.15)
    ax2.tick_params(axis='x', rotation=20)

    fig.suptitle('Ex09: стиснення та якість по 6 методах конверсії '
                  '(mean±std, 8 датасетів × 2 seeds)', fontsize=13, y=1.02, fontweight='bold')
    fig.tight_layout()
    out = FIGS_DIR / 'fig_a_compression_quality_by_method.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out.relative_to(ROOT)}')


# ═══ FIG B: RCU конверсії по методах (log scale) ═══
def fig_b(records):
    rcu_means = []
    for m in METHOD_ORDER:
        vals = [r['rcu_conversion'] for r in records if r['method'] == m]
        rcu_means.append(float(np.mean(vals)))

    labels = [METHOD_LABELS[m] for m in METHOD_ORDER]
    colors = [BLUE if m == 'gfcs' else GRAY for m in METHOD_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, rcu_means, color=colors, edgecolor='white', linewidth=0.8)
    ax.set_yscale('log')
    for bar, val in zip(bars, rcu_means):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.15, f'{val:.2f}',
                ha='center', fontsize=9, fontweight='bold', color=DARK)
    ax.set_ylabel('RCU конверсії (лог. шкала)')
    ax.set_title('Ex09: обчислювальна вартість конверсії по методах')
    ax.grid(axis='y', alpha=0.15, which='both')
    ax.tick_params(axis='x', rotation=20)
    fig.tight_layout()
    out = FIGS_DIR / 'fig_b_rcu_conversion_by_method.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out.relative_to(ROOT)}')


# ═══ FIG C: розсіювання «стиснення vs ΔF1» для всіх 96 записів ═══
def fig_c(records):
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(METHOD_ORDER):
        rows = [r for r in records if r['method'] == m]
        xs = [r['compression'] for r in rows]
        ys = [r['delta_f1'] for r in rows]
        if m == 'gfcs':
            ax.scatter(xs, ys, label=METHOD_LABELS[m], color=PALETTE[i],
                       marker='o', s=55, alpha=0.8,
                       edgecolors='white', linewidth=0.6)
        else:
            ax.scatter(xs, ys, label=METHOD_LABELS[m], color=PALETTE[i],
                       marker='x', s=40, alpha=0.8, linewidth=0.6)

    ax.axhline(0, color='gray', ls='--', alpha=0.4, linewidth=1)
    ax.set_xscale('log')
    ax.set_xlabel('Коефіцієнт стиснення, разів (лог. шкала)')
    ax.set_ylabel('ΔF1 (компактна − розріджена)')
    ax.set_title(f'Ex09: стиснення vs ΔF1, усі {len(records)} записів '
                 '(6 методів × 8 датасетів × 2 seeds)')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.15)
    fig.tight_layout()
    out = FIGS_DIR / 'fig_c_compression_vs_delta_f1_scatter.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out.relative_to(ROOT)}')


def main():
    records = load_records()
    assert len(records) == 96, f'очікувалось 96 записів, отримано {len(records)}'
    print(f'Завантажено {len(records)} записів з full_benchmark_rcu.json')
    fig_a(records)
    fig_b(records)
    fig_c(records)
    print(f'Готово. Фігури: {FIGS_DIR.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
