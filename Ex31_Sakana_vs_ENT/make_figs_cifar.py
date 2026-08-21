#!/usr/bin/env python3
"""
Ex31 ENT vs Sakana-CMA: фігури для публічного репозиторію дисертації
(§4.4, табл. 4.5, рисунок 4.1).

Скрипт лише ЧИТАЄ наявний файл результатів (results/sakana_vs_ent.tsv) —
жодні числа не перераховуються експериментально повторно, лише агрегуються
для візуалізації. Це дозволяє будь-кому відтворити фігуру без повторного
запуску ex31_benchmark.py.

Джерело чисел: results/sakana_vs_ent.tsv — 7 методів злиття на
комплементарному CIFAR-10 (SmallCNN, класи 0-4 проти 5-9, seed=42),
породжено ex31_benchmark.py.

Фігура (у стилі рисунка 4.1 дисертації, 4 панелі):
  (а) покласова точність по 10 класах для всіх 7 методів;
  (б) мінімальна точність по класу — захист найслабшого класу;
  (в) баланс розподілу знань (мінімум / максимум по класах);
  (г) окупність обчислень: час злиття (с) на один збережений клас
      (n_ok з accuracy > 30%, як у табл. 4.5).

Запуск: python3 make_figs_cifar.py  (з кореня Ex31_Sakana_vs_ENT/)
"""
import ast
import csv
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

METHOD_ORDER = ['WA', 'TaskArith', 'TIES', 'DARE-TIES', 'Sakana-CMA', 'ENT', 'ENT-FT']
METHOD_LABELS = {
    'WA': 'Усереднення ваг', 'TaskArith': 'Task Arithmetic',
    'TIES': 'TIES-Merging', 'DARE-TIES': 'DARE-TIES',
    'Sakana-CMA': 'Sakana-CMA', 'ENT': 'ENT (наш)', 'ENT-FT': 'ENT-FT (наш)',
}
PALETTE = {
    'WA': '#94a3b8', 'TaskArith': '#a3a3a3', 'TIES': '#dc2626',
    'DARE-TIES': '#f97316', 'Sakana-CMA': '#f59e0b',
    'ENT': '#2563eb', 'ENT-FT': '#16a34a',
}


def load_records():
    with open(ROOT / 'results' / 'sakana_vs_ent.tsv', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        rows = list(reader)
    by_method = {}
    for r in rows:
        r['accuracy'] = float(r['accuracy'])
        r['balance'] = float(r['balance'])
        r['min_class'] = float(r['min_class'])
        r['n_ok'] = int(r['n_ok'])
        r['n_total'] = int(r['n_total'])
        r['time_s'] = float(r['time_s'])
        r['per_class'] = ast.literal_eval(r['per_class'])
        by_method[r['method']] = r
    return by_method


def fig_4panel(by_method):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # (a) покласова точність, 7 методів × 10 класів
    classes = list(range(10))
    n_m = len(METHOD_ORDER)
    w = 0.8 / n_m
    x = np.arange(10)
    for i, m in enumerate(METHOD_ORDER):
        vals = [by_method[m]['per_class'][c] for c in classes]
        ax_a.bar(x + (i - (n_m - 1) / 2) * w, vals, w,
                  label=METHOD_LABELS[m], color=PALETTE[m],
                  edgecolor='white', linewidth=0.3)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(c) for c in classes])
    ax_a.set_xlabel('Клас CIFAR-10')
    ax_a.set_ylabel('Точність')
    ax_a.set_title('(а) Покласова точність')
    ax_a.grid(axis='y', alpha=0.15)

    names = [METHOD_LABELS[m] for m in METHOD_ORDER]
    colors = [PALETTE[m] for m in METHOD_ORDER]

    # (b) мінімальна точність по класу
    mins = [by_method[m]['min_class'] for m in METHOD_ORDER]
    bars = ax_b.bar(names, mins, color=colors, edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, mins):
        ax_b.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f'{val:.3f}',
                   ha='center', fontsize=8, color='#1e293b')
    ax_b.axhline(0.30, color='gray', ls='--', alpha=0.5, linewidth=1)
    ax_b.set_ylabel('min_c Acc$^{(c)}$')
    ax_b.set_title('(б) Мінімальна точність по класу')
    ax_b.grid(axis='y', alpha=0.15)
    ax_b.tick_params(axis='x', rotation=20)

    # (c) баланс
    bals = [by_method[m]['balance'] for m in METHOD_ORDER]
    bars = ax_c.bar(names, bals, color=colors, edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, bals):
        ax_c.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f'{val:.3f}',
                   ha='center', fontsize=8, color='#1e293b')
    ax_c.set_ylabel('Баланс (min/max по класах)')
    ax_c.set_title('(в) Баланс розподілу знань')
    ax_c.grid(axis='y', alpha=0.15)
    ax_c.tick_params(axis='x', rotation=20)

    # (d) час на один збережений клас
    tpc = [by_method[m]['time_s'] / by_method[m]['n_ok'] for m in METHOD_ORDER]
    bars = ax_d.bar(names, tpc, color=colors, edgecolor='white', linewidth=0.6)
    for bar, val in zip(bars, tpc):
        ax_d.text(bar.get_x() + bar.get_width() / 2, val + max(tpc) * 0.02,
                   f'{val:.1f}', ha='center', fontsize=8, color='#1e293b')
    ax_d.set_ylabel('Час злиття, с / збережений клас')
    ax_d.set_title('(г) Окупність обчислювальних витрат')
    ax_d.grid(axis='y', alpha=0.15)
    ax_d.tick_params(axis='x', rotation=20)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Ex31: ENT проти Sakana-CMA та 5 інших методів на CIFAR-10 '
                  '(SmallCNN, комплементарне злиття 0-4 проти 5-9), рис. 4.1',
                  fontsize=13, y=1.0, fontweight='bold')
    fig.tight_layout(rect=(0, 0.03, 1, 0.99))
    out = FIGS_DIR / 'fig_cifar_4panel_ent_vs_sakana.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'  saved {out.relative_to(ROOT)}')


def main():
    by_method = load_records()
    assert set(by_method) == set(METHOD_ORDER), \
        f'непередбачений набір методів: {sorted(by_method)}'
    # Звірка з табл. 4.5: ENT acc=0.626, bal=0.490, 10/10.
    assert abs(by_method['ENT']['accuracy'] - 0.626) < 1e-3
    print(f'Завантажено {len(by_method)} методів з results/sakana_vs_ent.tsv')
    fig_4panel(by_method)
    print(f'Готово. Фігура: {FIGS_DIR.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
