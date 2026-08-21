import pandas as pd
import matplotlib.pyplot as plt
import re
import os

md_file = 'results/L2_visuals/L2_Tables.md'
out_dir = 'results/L2_visuals/'

with open(md_file, 'r') as f:
    content = f.read()

def parse_all_md_tables(md_text):
    # Find all table blocks
    table_pattern = re.compile(r'(\|.*\|[\r\n]+)(?:\|[ :\-]+\|[\r\n]+)?((?:\|.*\|[\r\n]*)+)')
    tables = table_pattern.finditer(md_text)
    
    parsed_tables = []
    for match in tables:
        raw_table = match.group(0)
        lines = [l.strip() for l in raw_table.strip().split('\n') if '|' in l]
        if len(lines) < 3: continue
        
        headers = [c.strip() for c in lines[0].split('|')[1:-1]]
        
        data = []
        for line in lines[1:]:
            if '---' in line or set(line.strip().replace('|', '').replace(':', '').replace('-', '').replace(' ', '')) == set():
                continue
            row = [c.strip() for c in line.split('|')[1:-1]]
            if len(row) == len(headers):
                data.append(row)
                
        if data:
            parsed_tables.append(pd.DataFrame(data, columns=headers))
            
    return parsed_tables

def save_df_as_image(df, title, filename):
    # Dynamic sizing based on table content + increased width
    fig, ax = plt.subplots(figsize=(14, max(4, len(df)*0.6 + 1.5)))
    ax.axis('off')
    ax.axis('tight')
    
    plt.title(title, fontsize=16, pad=20, weight='bold')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)
    
    # Format Header
    for i, _ in enumerate(df.columns):
        table[(0, i)].set_facecolor('#2d3748')
        table[(0, i)].set_text_props(color='white', weight='bold')
        
    # Format Body
    for i in range(1, len(df)+1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f7fafc')
            cell_val = str(df.iloc[i-1, 0])
            if 'sacma_v' in cell_val:
                table[(i, j)].set_facecolor('#dcfce7') # Light green highlight
                
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

dfs = parse_all_md_tables(content)
titles = [
    "1. Зведена таблиця (Avg Rank, Normalized Regret, AUCC, Overhead)",
    "2. Wilcoxon Signed-Rank Test (sacma_v3 vs Base)",
    "3. Win Matrix (Task Ranks)",
    "4. Стабільність методів (STD по сідах)"
]

for i, df in enumerate(dfs):
    t_name = titles[i] if i < len(titles) else f"Table {i+1}"
    fname = f"Table_{i+1}_Render.png"
    save_df_as_image(df, t_name, fname)
    print(f"Saved: {fname}")
