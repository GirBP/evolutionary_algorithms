#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.figsize": (7, 5),
})

COLORS = {
    'DEHB': '#6B7280',        # Gray
    'SACMA-DAC': '#C0392B'    # Deep Red (Author method)
}

def plot_function(func_name, data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for method_name, runs in data.items():
        curves = np.array([run['curve'] for run in runs])
        x = np.arange(1, curves.shape[1] + 1)
        
        # Calculate median and IQR
        q50 = np.median(curves, axis=0)
        q25 = np.percentile(curves, 25, axis=0)
        q75 = np.percentile(curves, 75, axis=0)
        
        c = COLORS.get(method_name, '#000000')
        label = f"{method_name} (Запропоновано)" if method_name == 'SACMA-DAC' else f"{method_name} (Базовий)"

        ax.plot(x, q50, label=label, linewidth=2.5, color=c)
        ax.fill_between(x, q25, q75, color=c, alpha=0.15, linewidth=0)

    ax.set_title(f"Криві збіжності D=120: {func_name}\\n(Мінімізація за 120 обчислень)")
    ax.set_xlabel("Кількість оцінок функції (Evaluations)")
    ax.set_ylabel("Цільове значення (Loss)")
    
    ax.grid(True, linestyle="--", alpha=0.6, color="#E5E7EB")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, fancybox=True, edgecolor='#E5E7EB')
    
    plt.tight_layout()
    safe_name = func_name.split(' ')[0]
    out_path = f"results/05_Convergence_Ex63_{safe_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")

def main():
    json_path = 'results/ex63_highdim.json'
    if not os.path.exists(json_path):
        print(f"File {json_path} not found.")
        return
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    for func_name, methods_data in data.items():
        plot_function(func_name, methods_data)

if __name__ == '__main__':
    main()
