#!/usr/bin/env python3
"""Generate GFCS-only figures and tables for Ex09v2 dissertation.
Narrative: found leader on SimpleMLP → validated GFCS alone on CNN/ResNet."""
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})

RESULTS_DIR = Path(__file__).parent / 'results'
FIGS_DIR = RESULTS_DIR / 'figs'
TABLES_DIR = RESULTS_DIR / 'tables'
FIGS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

with open(RESULTS_DIR / 'ex09v2_benchmark.json') as f:
    mlp_data = json.load(f)['results']
with open(RESULTS_DIR / 'ex09v2_cnn_resnet_verified.json') as f:
    cnn_data = json.load(f)['results']

BLUE = '#2563eb'
GRAY = '#94a3b8'
DARK = '#1e293b'
GREEN = '#16a34a'
ORANGE = '#f59e0b'

def mlp_agg():
    methods = defaultdict(list)
    for r in mlp_data:
        methods[r['method']].append(r)
    agg = {}
    for m, rs in methods.items():
        agg[m] = {
            'f1': np.mean([r['final_f1'] for r in rs]),
            'rpr': np.mean([r['recovery'] for r in rs]),
            'comp': np.mean([r['compression'] for r in rs]),
            'infer': np.mean([r['infer_speedup'] for r in rs]),
            'params': int(np.mean([r['compact_params'] for r in rs])),
            'rcu': np.mean([r['rcu_conversion'] for r in rs]),
            'n': len(rs),
        }
    return agg


# ═══ FIG 1: Відбір лідера серед 5 методів (SimpleMLP) ═══
def fig1():
    agg = mlp_agg()
    order = ['GFCS', 'EAIB_e25', 'MOEA_e07', 'EPSS_e48', 'NeuronRem']
    labels = ['GFCS', 'EAIB', 'MOEA', 'EPSS', 'NeuronRem.']
    rprs = [agg[m]['rpr'] for m in order]
    comps = [agg[m]['comp'] for m in order]
    colors = [BLUE if m == 'GFCS' else GRAY for m in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(labels, rprs, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val, m in zip(bars, rprs, order):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.0004,
                f'{val:.4f}', ha='center', fontsize=10,
                fontweight='bold' if m == 'GFCS' else 'normal',
                color=BLUE if m == 'GFCS' else DARK)
    ax1.set_ylabel('RPR'); ax1.set_title('Збереження якості')
    ax1.set_ylim(0.993, 1.004); ax1.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax1.grid(axis='y', alpha=0.15)

    bars2 = ax2.bar(labels, comps, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val, m in zip(bars2, comps, order):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.15,
                f'{val:.1f}×', ha='center', fontsize=10,
                fontweight='bold' if m == 'GFCS' else 'normal',
                color=BLUE if m == 'GFCS' else DARK)
    ax2.set_ylabel('Comp×'); ax2.set_title('Ступінь стиснення')
    ax2.grid(axis='y', alpha=0.15)

    fig.suptitle('Етап 1: Відбір найкращого методу (SimpleMLP, 5 методів × 8 датасетів)',
                 fontsize=14, y=1.02, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGS_DIR / 'fig1_gfcs_mlp_leader.png')
    plt.close(fig)
    print("   fig1_gfcs_mlp_leader.png")


# ═══ FIG 2: GFCS on CNN & ResNet (ONLY GFCS, no comparison) ═══
def fig2():
    archs = ['SimpleMLP\n(34K params)', 'CNN\n(422K params)', 'ResNet\n(914K params)']
    mlp = mlp_agg()

    rprs = [
        mlp['GFCS']['rpr'],
        np.mean([r['gfcs']['rpr'] for r in cnn_data if r['arch'] == 'cnn']),
        np.mean([r['gfcs']['rpr'] for r in cnn_data if r['arch'] == 'resnet']),
    ]
    comps = [
        mlp['GFCS']['comp'],
        np.mean([r['gfcs']['comp'] for r in cnn_data if r['arch'] == 'cnn']),
        np.mean([r['gfcs']['comp'] for r in cnn_data if r['arch'] == 'resnet']),
    ]
    f1_teacher = [
        np.mean([r['teacher_f1'] for r in mlp_data if r['method'] == 'GFCS']),
        np.mean([r['teacher_f1'] for r in cnn_data if r['arch'] == 'cnn']),
        np.mean([r['teacher_f1'] for r in cnn_data if r['arch'] == 'resnet']),
    ]
    f1_gfcs = [
        mlp['GFCS']['f1'],
        np.mean([r['gfcs']['f1_post'] for r in cnn_data if r['arch'] == 'cnn']),
        np.mean([r['gfcs']['f1_post'] for r in cnn_data if r['arch'] == 'resnet']),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: F1 teacher vs GFCS
    ax = axes[0]
    x = np.arange(3); w = 0.32
    ax.bar(x - w/2, f1_teacher, w, label='Учитель', color=GRAY, edgecolor='white')
    ax.bar(x + w/2, f1_gfcs, w, label='GFCS', color=BLUE, edgecolor='white')
    for i in range(3):
        ax.text(i + w/2, f1_gfcs[i] + 0.005, f'{f1_gfcs[i]:.3f}',
               ha='center', fontsize=9, fontweight='bold', color=BLUE)
    ax.set_ylabel('F1-score (macro)')
    ax.set_title('Точність класифікації')
    ax.set_xticks(x); ax.set_xticklabels(archs, fontsize=9)
    ax.legend(fontsize=9); ax.set_ylim(0.8, 1.0); ax.grid(axis='y', alpha=0.15)

    # Panel B: RPR
    ax = axes[1]
    ax.bar(archs, rprs, color=BLUE, edgecolor='white', width=0.5)
    for i, v in enumerate(rprs):
        ax.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=11, fontweight='bold', color=BLUE)
    ax.set_ylabel('RPR')
    ax.set_title('Збереження якості (RPR)')
    ax.set_ylim(0.9, 1.02)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.15)

    # Panel C: Compression
    ax = axes[2]
    ax.bar(archs, comps, color=BLUE, edgecolor='white', width=0.5)
    for i, v in enumerate(comps):
        ax.text(i, v + 0.15, f'{v:.1f}×', ha='center', fontsize=11, fontweight='bold', color=BLUE)
    ax.set_ylabel('Comp×')
    ax.set_title('Ступінь стиснення')
    ax.grid(axis='y', alpha=0.15)

    fig.suptitle('Етап 2: GFCS на архітектурах різної складності (FashionMNIST, 95% sparsity)',
                 fontsize=14, y=1.02, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIGS_DIR / 'fig2_gfcs_scaling.png')
    plt.close(fig)
    print("   fig2_gfcs_scaling.png")


# ═══ FIG 3: Pipeline ═══
def fig3():
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    stages = [
        ('Учитель\n(pretrained)', '914K params', '#64748b'),
        ('Прунінг\nTESA-26 @ 95%', '95% ваг = 0', ORANGE),
        ('GFCS\nстиснення', 'EA + flow\nimportance', BLUE),
        ('Compact\nмодель', '308K params', GREEN),
        ('Finetune\n25 batches', 'SGD lr=0.01', '#8b5cf6'),
        ('Результат', 'RPR=0.940\n3.4× comp', '#dc2626'),
    ]
    for i, (title, sub, color) in enumerate(stages):
        x = 0.08 + i * 0.155
        rect = plt.Rectangle((x, 0.25), 0.12, 0.5, lw=2, edgecolor=color,
                             facecolor=color, alpha=0.15, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(x+0.06, 0.58, title, transform=ax.transAxes, ha='center', va='center',
               fontsize=11, fontweight='bold', color=color)
        ax.text(x+0.06, 0.38, sub, transform=ax.transAxes, ha='center', va='center',
               fontsize=9, color=DARK)
        if i < len(stages)-1:
            ax.annotate('', xy=(x+0.135, 0.5), xytext=(x+0.12, 0.5),
                       xycoords='axes fraction', textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))
    ax.set_title('Конвеєр прунінг → GFCS → compact (на прикладі ResNet)',
                fontsize=14, fontweight='bold', pad=15)
    fig.savefig(FIGS_DIR / 'fig3_gfcs_pipeline.png')
    plt.close(fig)
    print("   fig3_gfcs_pipeline.png")


# ═══ FIG 4: GFCS per-dataset F1 ═══
def fig4():
    ds_order = ['moons','circles','spirals','blobs','gaussian_q','classification','highdim','sequence_cls']
    labels = ['Moons','Circles','Spirals','Blobs','Gaussian','Classif.','HighDim','SeqCls']
    fig, ax = plt.subplots(figsize=(10, 5))

    tvs, gvs = [], []
    valid_labels = []
    for ds, lb in zip(ds_order, labels):
        rs = [r for r in mlp_data if r['dataset'] == ds and r['method'] == 'GFCS']
        if rs:
            tvs.append(np.mean([r['teacher_f1'] for r in rs]))
            gvs.append(np.mean([r['final_f1'] for r in rs]))
            valid_labels.append(lb)

    ax.plot(valid_labels, tvs, 's--', color=GRAY, alpha=0.6, label='Учитель',
            markersize=8, linewidth=1.5)
    ax.plot(valid_labels, gvs, 'o-', color=BLUE, label='GFCS compact',
            markersize=10, linewidth=2.5, markeredgecolor='white', markeredgewidth=1.5)
    ax.fill_between(range(len(valid_labels)), tvs, gvs, alpha=0.08, color=BLUE)
    ax.set_ylabel('F1-score'); ax.set_ylim(0.8, 1.02)
    ax.set_title('GFCS: збереження якості по кожному датасету (SimpleMLP)', fontweight='bold')
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.15)
    fig.savefig(FIGS_DIR / 'fig4_gfcs_per_dataset.png')
    plt.close(fig)
    print("   fig4_gfcs_per_dataset.png")


# ═══ TABLES ═══
def tables():
    agg = mlp_agg()

    # T1: MLP 5 methods
    with open(TABLES_DIR / 'table1_mlp_methods.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Метод','F1','RPR','Comp×','Infer×','Params','RCU_conv','N'])
        for m, lb in [('GFCS','GFCS'),('NeuronRem','NeuronRemoval'),
                      ('EAIB_e25','EAIB'),('EPSS_e48','EPSS'),('MOEA_e07','MOEA')]:
            a = agg[m]
            w.writerow([lb, f"{a['f1']:.4f}", f"{a['rpr']:.4f}", f"{a['comp']:.1f}",
                       f"{a['infer']:.2f}", a['params'], f"{a['rcu']:.1f}", a['n']])
    print("   table1_mlp_methods.csv")

    # T2: GFCS on CNN/ResNet (ONLY GFCS)
    with open(TABLES_DIR / 'table2_gfcs_large_models.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Архітектура','Seed','Teacher F1','Teacher params',
                    'GFCS F1','GFCS params','RPR','Comp×','Time(ms)'])
        for r in cnn_data:
            g = r['gfcs']
            w.writerow([r['arch'].upper(), r['seed'],
                       f"{r['teacher_f1']:.4f}", r['teacher_params'],
                       f"{g['f1_post']:.4f}", g['params'],
                       f"{g['rpr']:.4f}", f"{g['comp']:.1f}", g['time_ms']])
    print("   table2_gfcs_large_models.csv")

    # T3: Cross-arch summary (GFCS only)
    with open(TABLES_DIR / 'table3_gfcs_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Архітектура','Params (teacher)','Params (GFCS)','F1','RPR','Comp×'])
        # MLP
        w.writerow(['SimpleMLP', '33,666', f"{agg['GFCS']['params']:,}",
                    f"{agg['GFCS']['f1']:.4f}", f"{agg['GFCS']['rpr']:.4f}",
                    f"{agg['GFCS']['comp']:.1f}"])
        # CNN
        cr = [r for r in cnn_data if r['arch'] == 'cnn']
        w.writerow(['CNN', '421,642',
                    f"{int(np.mean([r['gfcs']['params'] for r in cr])):,}",
                    f"{np.mean([r['gfcs']['f1_post'] for r in cr]):.4f}",
                    f"{np.mean([r['gfcs']['rpr'] for r in cr]):.4f}",
                    f"{np.mean([r['gfcs']['comp'] for r in cr]):.1f}"])
        # ResNet
        rr = [r for r in cnn_data if r['arch'] == 'resnet']
        w.writerow(['ResNet', '914,122',
                    f"{int(np.mean([r['gfcs']['params'] for r in rr])):,}",
                    f"{np.mean([r['gfcs']['f1_post'] for r in rr]):.4f}",
                    f"{np.mean([r['gfcs']['rpr'] for r in rr]):.4f}",
                    f"{np.mean([r['gfcs']['comp'] for r in rr]):.1f}"])
    print("   table3_gfcs_summary.csv")


if __name__ == '__main__':
    print("Figures...", flush=True); fig1(); fig2(); fig3(); fig4()
    print("\nTables...", flush=True); tables()
    print(f"\nDone. Figs: {FIGS_DIR}, Tables: {TABLES_DIR}")
