#!/usr/bin/env python3
"""
Ex30 ENT-FT: фігура ефекту калібрації на чемпіоні e34 (§4.3, табл. 4.4).

Скрипт лише ЧИТАЄ results_ent_ft_on_e34.json — жодні числа не перераховуються
експериментально повторно, лише агрегуються для візуалізації.

Джерело чисел: results_ent_ft_on_e34.json (породжується
ent_ft_on_e34_champion.py — ENT-FT застосований до точного чемпіона
e34_benchmark.py, звіреного з results_e34.json асертом).

Фігура: fig_c_entft_effect.png — два панелі:
  (a) агрегати до/після калібрації: точність, мінімальна класова точність,
      міжгруповий баланс (min(A,B)/max(A,B));
  (b) покласова точність до/після калібрації, 10 класів MNIST.

Запуск: python3 make_fig_entft.py  (з кореня Ex30_HetMerge_ENT/)
"""
import json
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
RESULTS_FILE = ROOT / 'results_ent_ft_on_e34.json'

GRAY = '#94a3b8'   # до калібрації (чемпіон ENT)
BLUE = '#2563eb'   # після калібрації (ENT-FT)


def load_results() -> dict:
    with open(RESULTS_FILE, encoding='utf-8') as f:
        data = json.load(f)
    assert data['champion_verification']['matched'] is True, \
        'results_ent_ft_on_e34.json: чемпіон не пройшов звірку з results_e34.json'
    return data


def fig_aggregates(ax, data: dict) -> None:
    before = data['before_calibration']
    after = data['after_calibration_ent_ft']
    labels = ['Точність', 'Мінімальна\nкласова точність', 'Баланс\n(міжгруп. A/B)']
    vals_before = [before['accuracy'], before['balance']['min_class_acc'],
                   before['balance']['group_balance_min_over_max']]
    vals_after = [after['accuracy'], after['balance']['min_class_acc'],
                  after['balance']['group_balance_min_over_max']]

    x = np.arange(len(labels))
    w = 0.32
    ax.bar(x - w / 2, vals_before, w, label='До калібрації (ENT)', color=GRAY,
           edgecolor='white', linewidth=0.6)
    ax.bar(x + w / 2, vals_after, w, label='Після калібрації (ENT-FT)', color=BLUE,
           edgecolor='white', linewidth=0.6)
    for xi, v in zip(x - w / 2, vals_before):
        ax.text(xi, v + 0.015, f'{v:.3f}', ha='center', fontsize=9, color='#1e293b')
    for xi, v in zip(x + w / 2, vals_after):
        ax.text(xi, v + 0.015, f'{v:.3f}', ha='center', fontsize=9, color='#1e293b')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Значення')
    ax.set_ylim(0, 1.08)
    ax.set_title('Агрегати до/після ENT-FT')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.15)


def fig_per_class(ax, data: dict) -> None:
    pc_before = data['before_calibration']['per_class']
    pc_after = data['after_calibration_ent_ft']['per_class']
    classes = list(range(10))
    vals_before = [pc_before[str(c)] for c in classes]
    vals_after = [pc_after[str(c)] for c in classes]

    x = np.arange(10)
    w = 0.35
    ax.bar(x - w / 2, vals_before, w, label='До калібрації (ENT)', color=GRAY,
           edgecolor='white', linewidth=0.6)
    ax.bar(x + w / 2, vals_after, w, label='Після калібрації (ENT-FT)', color=BLUE,
           edgecolor='white', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_xlabel('Клас MNIST')
    ax.set_ylabel('Точність')
    ax.set_ylim(0, 1.08)
    ax.set_title('Покласова точність до/після ENT-FT')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.15)


def main() -> None:
    data = load_results()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1.6]})
    fig_aggregates(axes[0], data)
    fig_per_class(axes[1], data)
    fig.suptitle('Ex30: ENT-FT — калібрація точного чемпіона e34_benchmark.py '
                  '(results_ent_ft_on_e34.json, табл. 4.4)', fontsize=13, y=1.03,
                  fontweight='bold')
    fig.tight_layout()
    out = FIGS_DIR / 'fig_c_entft_effect.png'
    fig.savefig(out)
    plt.close(fig)
    print(f'Завантажено {RESULTS_FILE.name}, звірку з results_e34.json пройдено.')
    print(f'  saved {out.relative_to(ROOT)}')
    print(f'Готово. Фігура: {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
