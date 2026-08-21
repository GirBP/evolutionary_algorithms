import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: try to import yahpo_gym for dataset metadata
try:
    import yahpo_gym
    from yahpo_gym import local_config
    local_config.init_config()
    target_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'yahpo_data')
    local_config.set_data_path(target_dir)
    YAHPO_AVAILABLE = True
except ImportError:
    YAHPO_AVAILABLE = False

import sys
from collections import defaultdict

TARGET_TIER = sys.argv[1] if len(sys.argv) > 1 else 'L2'
RESULTS_DIR = f'results/{TARGET_TIER}/'
OUTPUT_DIR = f'results/{TARGET_TIER}_visuals/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

NAME_MAP = {
    'sacma_v3': 'SACMA-DAC (Ours)',
    'sacma_base': 'SACMA-base (Ours)',
    'sacma_mab': 'SACMA-MAB (Ours)',
    'sacma_lazy': 'SACMA-lazy (Ours)',
    'whales_cma': 'WL-CMA (Ours)',
    'iw_moea': 'IW-MOEA (Ours)',
    'antivanila': 'Sigma-CMA (Ours)',
    'tpe': 'TPE',
    'tpe_optuna': 'TPE',
    'bo_gp': 'BO (GP)',
    'shade': 'SHADE',
    'shade_real': 'SHADE',
    'random_search': 'Random Search',
    'lshade': 'L-SHADE',
    'cmaes_pure': 'CMA-ES',
    'smac_method': 'SMAC-RF',
    'dehb_method': 'DEHB-DE',
}

def load_data():
    files = glob.glob(os.path.join(RESULTS_DIR, '**', '*.json'), recursive=True)
    
    losses = defaultdict(lambda: defaultdict(dict))
    curves = defaultdict(lambda: defaultdict(dict))
    rcus = defaultdict(lambda: defaultdict(dict))
    rcus_train = defaultdict(lambda: defaultdict(dict))
    
    datasets = set()
    methods = set()
    
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        
        m = r['method']
        d = r['dataset']
        s = r['seed']
        
        methods.add(m)
        datasets.add(d)
        
        losses[m][d][s] = r['loss']
        rcus[m][d][s] = r.get('rcu_hpo', r.get('rcu_method', 0.0))
        rcu_t = r.get('rcu_train_best', None)
        if rcu_t is not None:
            rcus_train[m][d][s] = float(rcu_t)
        if 'curve' in r:
            curves[m][d][s] = r['curve']
            
    return losses, curves, rcus, rcus_train, sorted(list(methods)), sorted(list(datasets))

from collections import defaultdict

def main():
    losses, curves, rcus, rcus_train, methods, datasets = load_data()
    print(f"Loaded data for {len(methods)} methods across {len(datasets)} datasets.")
    
    # ---------------------------------------------------------
    # Aggregate Mean Loss and Ranks
    # ---------------------------------------------------------
    mean_losses = pd.DataFrame(index=methods, columns=datasets)
    std_losses = pd.DataFrame(index=methods, columns=datasets)
    mean_auccs = pd.DataFrame(index=methods, columns=datasets)
    for m in methods:
        for d in datasets:
            loss_vals = [losses[m][d][s] for s in losses[m][d]]
            mean_losses.at[m, d] = np.mean(loss_vals)
            std_losses.at[m, d] = np.std(loss_vals) if len(loss_vals) > 1 else 0.0
            
            # calculate AUCC
            auccs = []
            trapz_fn = getattr(np, 'trapz', getattr(np, 'trapezoid', None))
            for s in curves[m][d]:
                c = curves[m][d][s]
                auccs.append(trapz_fn(c) if len(c) > 1 else losses[m][d][s])
            mean_auccs.at[m, d] = np.mean(auccs) if auccs else np.inf
            
    mean_losses = mean_losses.astype(float)
    # Ranks per dataset (1 is best)
    ranks = mean_losses.rank(method='min', ascending=True)
    
    # ---------------------------------------------------------
    # 1. & 3. WIN MATRIX & HEATMAP OF RANKS
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 8))
    # Custom colormap: Green (1) to Red (14)
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    
    # Sort methods by average rank for better visual flow
    avg_ranks = ranks.mean(axis=1).sort_values()
    sorted_methods = avg_ranks.index.tolist()
    
    sns.heatmap(ranks.loc[sorted_methods], annot=True, cmap=cmap, cbar_kws={'label': 'Rank (Lower is Better)'})
    plt.title("Heatmap of Ranks (Method x Dataset)")
    plt.xlabel("LCBench Datasets")
    plt.ylabel("Algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "1_3_Rank_Heatmap.png"), dpi=300)
    plt.close()
    print("Saved 1_3_Rank_Heatmap.png")

    # ---------------------------------------------------------
    # 2. CLUSTERING BY METADATA (LCBench meta-features)
    # ---------------------------------------------------------
    print("\n--- 2. Dataset Clustering (Empirical Performance Similarity) ---")
    try:
        # We cluster results empirically based on algorithm performance similarity
        # This effectively groups tasks with similar "difficulty surfaces"
        corr = ranks.corr()
        plt.figure(figsize=(10, 8))
        cg = sns.clustermap(ranks.T, cmap=cmap, figsize=(10, 8), metric="correlation")
        cg.fig.suptitle("Task Clustering by Algorithm Performance")
        plt.savefig(os.path.join(OUTPUT_DIR, "2_Task_Clustermap.png"), dpi=300)
        plt.close()
        print("Saved 2_Task_Clustermap.png (Empirical task grouping based on algorithm performance similarity)")
    except Exception as e:
        print(f"Skipping clustering plot due to error: {e}")

    # ---------------------------------------------------------
    # 4. RADAR CHART (Top 5 methods — Multi-Metric Balance)
    # Axes: Accuracy (1-NormMean), Stability (1-NormSTD), Convergence (1-NormAUCC), Efficiency (1-NormRCU)
    # ---------------------------------------------------------
    top5_methods = avg_ranks.head(5).index.tolist()
    
    # Pre-calculate global averages per method
    m_means = {m: np.mean([mean_losses[d].loc[m] for d in datasets]) for m in top5_methods}
    m_stds = {m: np.mean([std_losses[d].loc[m] for d in datasets]) for m in top5_methods}
    m_auccs = {m: np.mean([mean_auccs[d].loc[m] for d in datasets]) for m in top5_methods}
    # we need RCU info if possible, currently we only have losses/auccs clearly available. 
    # Let's extract RCU if available, or just omit.
    # Actually, RCU is printed to Terminal, let's use the actual data!
    metrics = ["Accuracy", "Stability", "Speed (AUCC)"]
    
    # Normalization (Max becomes 0, Min becomes 1, so outer = best)
    def rescale(metric_dict):
        vals = list(metric_dict.values())
        min_v, max_v = min(vals), max(vals)
        denom = max_v - min_v if max_v != min_v else 1e-9
        return {k: 1.0 - ((v - min_v)/denom) for k, v in metric_dict.items()}

    scores = {}
    for m in top5_methods:
        scores[m] = []
        
    s_means = rescale(m_means)
    s_stds = rescale(m_stds)
    s_auccs = rescale(m_auccs)
    
    for m in top5_methods:
        scores[m].extend([s_means[m], s_stds[m], s_auccs[m]])

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1] # close loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for m in top5_methods:
        values = scores[m] + scores[m][:1]
        ax.plot(angles, values, linewidth=2, label=m)
        ax.fill(angles, values, alpha=0.1)
        
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    # Fix the scale to [0, 1]
    ax.set_ylim(0, 1)
    plt.title("Performance Profile (Outer Edge = Best)")
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "4_Radar_Chart.png"), dpi=300)
    plt.close()
    print("Saved 4_Radar_Chart.png")

    # ---------------------------------------------------------
    # 5. PIAS (Per-Instance Algorithm Selection)
    # ---------------------------------------------------------
    print("\n--- 5. Per-Instance Algorithm Selection (PIAS) ---")
    best_methods = mean_losses.idxmin()
    portfolio_counts = best_methods.value_counts()
    
    print("Virtual Best Solver (VBS) Portfolio Makeup:")
    for method, count in portfolio_counts.items():
        print(f" - {method}: Wins on {count}/{len(datasets)} datasets")
        
    if portfolio_counts.max() / len(datasets) >= 0.8:
        print(f"\nCONCLUSION: {portfolio_counts.idxmax()} is a Universal Solver (>= 80% wins).")
    else:
        print("\nCONCLUSION: No dominant universal solver. A meta-selector logic is strongly justified!")

    # ---------------------------------------------------------
    # 6. PER-TASK OUTPUT (Tables & Curves in Subdirectories)
    # ---------------------------------------------------------
    print("\n--- 6. Generating Per-Task Visuals and Tables ---")
    for d in datasets:
        task_dir = os.path.join(OUTPUT_DIR, d)
        os.makedirs(task_dir, exist_ok=True)
        
        # 6.1 Convergence Curve specific to this dataset
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Determine best methods for THIS dataset based on mean loss
        task_losses = mean_losses[d].sort_values()
        top_task_methods = task_losses.head(5).index.tolist()
        baseline_methods = ['random_search', 'bo_gp', 'cmaes_vanilla', 'tpe_optuna', 'sacma_v3']
        plot_m = list(set(top_task_methods + baseline_methods))
        
        start_losses = []
        min_loss_global = float('inf')
        
        for m in sorted(plot_m):
            if m not in curves or d not in curves[m]: continue
            all_curves = []
            for s in curves[m][d]:
                c = curves[m][d][s]
                if len(c) > 0:
                    all_curves.append(c)
                    
            if not all_curves: continue
            
            # Ensure equal length padding if needed
            max_len = max([len(c) for c in all_curves])
            padded = []
            for c in all_curves:
                c = np.array(c)
                if len(c) < max_len:
                    c = np.pad(c, (0, max_len - len(c)), 'edge')
                padded.append(c)
                
            curve_mean = np.mean(padded, axis=0)
            curve_std = np.std(padded, axis=0)
            
            start_losses.append(curve_mean[0])
            min_loss_global = min(min_loss_global, np.min(curve_mean))
            
            x = np.arange(1, len(curve_mean) + 1)
            plt.plot(x, curve_mean, label=m, linewidth=2)
            plt.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.15)
            
        plt.title(f"Convergence Curves (Mean ± Std) across Seeds\nTask: {d}")
        plt.xlabel("Evaluations (Budget)")
        plt.ylabel("Validation Loss")
        
        # Smart Y-axis scaling to ignore extreme starting outliers
        if start_losses:
            median_start = np.median(start_losses)
            # If the max starting loss is insanely high, cap the plot
            if max(start_losses) > median_start * 3:
                plt.ylim(bottom=min_loss_global * 0.9, top=median_start + (median_start - min_loss_global))
                
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, "Convergence_Curve.png"), dpi=300)
        plt.close()
        
        # --- 6.1a Loss Stability (Boxplot) ---
        plt.figure(figsize=(10, 6))
        data_to_plot = []
        labels_to_plot = []
        for m in task_losses.index: # sorted by mean loss
            m_losses = [losses[m][d][s] for s in losses[m][d]]
            if len(m_losses) > 0:
                data_to_plot.append(m_losses)
                labels_to_plot.append(m)
        if data_to_plot:
            sns.boxplot(data=data_to_plot, orient='h', palette="Set2")
            plt.yticks(ticks=range(len(labels_to_plot)), labels=labels_to_plot)
            plt.title(f"Loss Stability across Seeds\nTask: {d}")
            plt.xlabel("Validation Loss (Lower is Better)")
            plt.tight_layout()
            plt.savefig(os.path.join(task_dir, "Loss_Stability_Boxplot.png"), dpi=300)
        plt.close()

        # --- 6.1b RCU_hpo Overhead (Bar chart) ---
        plt.figure(figsize=(10, 6))
        # Fetch mean rcu for this dataset
        mean_rcu = {}
        for m in methods:
            m_rcus = [rcus[m][d][s] for s in rcus[m][d]]
            if m_rcus:
                mean_rcu[m] = np.mean(m_rcus)
        # Sort by RCU ascending
        sorted_rcu = sorted(mean_rcu.items(), key=lambda x: x[1])
        rcu_labels = [x[0] for x in sorted_rcu]
        rcu_vals = [x[1] for x in sorted_rcu]
        
        sns.barplot(x=rcu_vals, y=rcu_labels, palette="viridis")
        plt.title(f"Computational Overhead (RCU_hpo) per Method\nTask: {d}")
        plt.xlabel("RCU (Relative Computing Units) - Lower is Better")
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, "RCU_Overhead.png"), dpi=300)
        plt.close()

        # --- 6.1c Loss vs RCU Trade-off (Scatter plot) ---
        plt.figure(figsize=(10, 6))
        for m in task_losses.index:
            m_mean = mean_losses.at[m, d]
            m_rcu = mean_rcu.get(m, 0)
            if pd.isna(m_mean): continue
            
            # Highlight sacma methods
            color = 'red' if 'sacma' in m else 'blue'
            marker = '*' if 'sacma' in m else 'o'
            size = 150 if 'sacma' in m else 80
            
            plt.scatter(m_rcu, m_mean, c=color, marker=marker, s=size, alpha=0.7, edgecolors='k')
            plt.text(m_rcu * 1.05, m_mean, m, fontsize=9, ha='left', va='center')
            
        plt.title(f"Trade-off: Loss vs RCU Overhead\nTask: {d}")
        plt.xlabel("RCU Overhead (Lower is Better)")
        plt.ylabel("Mean Validation Loss (Lower is Better)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(task_dir, "Loss_vs_RCU_Tradeoff.png"), dpi=300)
        plt.close()
        
        # Check if we have RCU_train data
        has_train_data = any(d in rcus_train[m] and rcus_train[m][d] for m in rcus_train)
        
        # 6.2 Markdown Summary Table
        md_path = os.path.join(task_dir, "Summary.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Зведені результати для задачі: {d}\n\n")
            if has_train_data:
                f.write("| Алгоритм | Ранг | Середній Loss | STD | Найкращий Loss (Min) | RCU_hpo | RCU_train |\n")
                f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n")
            else:
                f.write("| Алгоритм | Ранг | Середній Loss | STD | Найкращий Loss (Min) | RCU_hpo |\n")
                f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
                
            for m in task_losses.index:
                # get stats
                m_losses = [losses[m][d][s] for s in losses[m][d]]
                if not m_losses: continue
                m_mean = np.mean(m_losses)
                m_std = np.std(m_losses)
                m_min = np.min(m_losses)
                m_rank = ranks.at[m, d]
                
                m_rcu_hpo = np.mean([rcus[m][d][s] for s in rcus[m][d]]) if rcus[m][d] else 0.0
                
                # Bold for author's methods, display name from NAME_MAP
                m_name = NAME_MAP.get(m, m)
                m_name = f"**{m_name}**" if "(Ours)" in m_name else m_name
                
                if has_train_data:
                    m_rcu_train = np.mean([rcus_train[m][d][s] for s in rcus_train[m][d]]) if rcus_train[m][d] else 0.0
                    train_str = f"{m_rcu_train:.0f}" if m_rcu_train > 0 else "N/A"
                    f.write(f"| {m_name} | {int(m_rank)} | {m_mean:.4f} | {m_std:.4f} | {m_min:.4f} | {m_rcu_hpo:.0f} | {train_str} |\n")
                else:
                    f.write(f"| {m_name} | {int(m_rank)} | {m_mean:.4f} | {m_std:.4f} | {m_min:.4f} | {m_rcu_hpo:.0f} |\n")
                
    print(f"Saved Convergence Curves and Summary.md files to subdirectories in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
