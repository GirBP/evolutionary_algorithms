#!/usr/bin/env python3
"""
Ex30 ENT: фігури для публічного репозиторію дисертації (§4.2, табл. 4.3, 4.6).

Скрипт лише ЧИТАЄ наявні файли результатів — жодні числа не перераховуються
експериментально повторно, лише агрегуються для візуалізації. Це дозволяє
будь-кому відтворити фігури без повторного запуску e34_benchmark.py.

Джерела чисел:
  - results_e34.json — 9 методів злиття на комплементарному MNIST (0-4 vs
    5-9), поле per_class для кожного методу (породжується e34_benchmark.py).
  - results_e34.txt — той самий запуск; стовпець «Parent» у розділі
    «Per-class breakdown» (рядки 66-77) не зберігається в JSON, тому
    береться з текстового логу того самого запуску. Parent(c) = найкраща
    з двох батьківських точностей на власному класі: max(acc_A(c), acc_B(c)).

Фігури:
  (a) fig_a_mnist_per_class.png — покласова точність: Батьківська модель
      (max з двох) / TIES / Sakana-CMA / ENT, 10 класів (табл. 4.6).
  (b) fig_b_mnist_aggregates.png — точність, баланс груп A/B та
      мінімальна класова точність по всіх 9 методах (табл. 4.3 + інші
      baseline-методи з того самого прогону).

Запуск: python3 make_figs_mnist.py  (з кореня Ex30_HetMerge_ENT/)
"""
import json
import re
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
FIGS_DIR = ROOT / 'figs'
FIGS_DIR.mkdir(parents=True, exist_ok=True)

BLUE = '#2563eb'   # ENT
GRAY = '#94a3b8'   # Parent (довідкова верхня межа)
RED = '#dc2626'    # TIES
AMBER = '#f59e0b'  # Sakana-CMA
PALETTE9 = ['#94a3b8', '#a3a3a3', '#78716c', '#dc2626', '#f97316',
            '#eab308', '#16a34a', '#f59e0b', '#2563eb']


def load_json_methods():
    with open(ROOT / 'results_e34.json', encoding='utf-8') as f:
        return json.load(f)


def load_parent_per_class():
    """Парсить стовпець «Parent» з розділу «Per-class breakdown» у results_e34.txt."""
    text = (ROOT / 'results_e34.txt').read_text(encoding='utf-8')
    m = re.search(r'Per-class breakdown.*?\n(.*)', text, re.S)
    assert m, 'не знайдено розділ "Per-class breakdown" у results_e34.txt'
    rows = {}
    for line in m.group(1).splitlines():
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        cls = int(parts[0])
        parent_acc = float(parts[1])
        rows[cls] = parent_acc
    assert len(rows) == 10, f'очікувалось 10 класів, отримано {len(rows)}'
    return rows


# ═══ FIG A: покласова точність — Parent / TIES / Sakana-CMA / ENT ═══
def fig_a(methods, parent_per_class):
    by_name = {m['name']: m for m in methods}
    ties = by_name['TIES(d=0.3)']
    sakana = by_name['Sakana-CMA']
    ent = by_name['ENT']

    classes = list(range(10))
    parent_vals = [parent_per_class[c] for c in classes]
    ties_vals = [ties['per_class'][str(c)] for c in classes]
    sakana_vals = [sakana['per_class'][str(c)] for c in classes]
    ent_vals = [ent['per_class'][str(c)] for c in classes]

    x = np.arange(10)
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - 1.5 * w, parent_vals, w, label='Батьківська модель (max)', color=GRAY,
           edgecolor='white', linewidth=0.6)
    ax.bar(x - 0.5 * w, ties_vals, w, label='TIES-Merging', color=RED,
           edgecolor='white', linewidth=0.6)
    ax.bar(x + 0.5 * w, sakana_vals, w, label='Sakana-CMA', color=AMBER,
           edgecolor='white', linewidth=0.6)
    ax.bar(x + 1.5 * w, ent_vals, w, label='ENT (наш)', color=BLUE,
           edgecolor='white', linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel('Клас MNIST')
    ax.set_ylabel('Точність')
    ax.set_ylim(0, 1.05)
    ax.set_title('Ex30: покласова точність на комплементарному MNIST '
                  '(0-4 проти 5-9), табл. 4.6')
    ax.legend(fontsize=9, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis='y', alpha=0.15)
    fig.tight_layout()
    out = FIGS_DIR / 'fig_a_mnist_per_class.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out.relative_to(ROOT)}')


# ═══ FIG B: агрегати acc / balance / min_c по 9 методах ═══
def fig_b(methods):
    order = sorted(range(len(methods)), key=lambda i: methods[i]['acc'])
    names = [methods[i]['name'] for i in order]
    accs = [methods[i]['acc'] for i in order]
    bals = [methods[i]['bal'] for i in order]
    mins = [methods[i]['min'] for i in order]
    colors = [BLUE if methods[i]['name'] == 'ENT' else '#94a3b8' for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    panels = [
        (axes[0], accs, 'Точність (усі 10 класів)'),
        (axes[1], bals, 'Баланс груп A/B (min/max групових середніх)'),
        (axes[2], mins, 'Мінімальна класова точність'),
    ]
    for ax, vals, title in panels:
        bars = ax.barh(names, vals, color=colors, edgecolor='white', linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.015, bar.get_y() + bar.get_height() / 2, f'{val:.3f}',
                    va='center', fontsize=8, color='#1e293b')
        ax.set_title(title)
        ax.set_xlim(0, 1.12)
        ax.grid(axis='x', alpha=0.15)

    fig.suptitle('Ex30: агрегати по 9 методах злиття на комплементарному MNIST '
                  '(results_e34.json, табл. 4.3)', fontsize=13, y=1.03, fontweight='bold')
    fig.tight_layout()
    out = FIGS_DIR / 'fig_b_mnist_aggregates.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out.relative_to(ROOT)}')


def main():
    methods = load_json_methods()
    assert len(methods) == 9, f'очікувалось 9 методів, отримано {len(methods)}'
    parent_per_class = load_parent_per_class()

    # Звірка: TIES[9] має дорівнювати 0.959 (перевірка, що файл не підмінено).
    by_name = {m['name']: m for m in methods}
    assert by_name['TIES(d=0.3)']['per_class']['9'] == 0.959, \
        'TIES[9] != 0.959 — results_e34.json відрізняється від очікуваного'

    print(f'Завантажено {len(methods)} методів з results_e34.json')
    fig_a(methods, parent_per_class)
    fig_b(methods)
    print(f'Готово. Фігури: {FIGS_DIR.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
