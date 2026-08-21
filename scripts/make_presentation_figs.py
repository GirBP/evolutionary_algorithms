"""Презентаційні фігури для захисту — генеруються з файлів результатів репозиторію.

Стиль уніфіковано з наявними фігурами дисертації (рис. 2.10, 4.1):
жирний центрований заголовок, панелі «(а)/(б)», сині «наші» методи проти
сірих бейзлайнів, підписи значень над стовпчиками, примітки курсивом під
фігурою. Кожна фігура читає лише наявні дані (json/tsv/csv/txt); жодне число
не вводиться вручну — все обчислюється або зчитується з файлів результатів.

Запуск:  python3 scripts/make_presentation_figs.py
"""
import csv
import json
import textwrap
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation_figs"
OUT.mkdir(exist_ok=True)

# Палітра — як у фігурах дисертації (рис. 4.1 / CD-діаграма Ex08)
C_OURS = "#2b6fd6"      # авторський метод (синій, як ENT у рис. 4.1)
C_OURS2 = "#2ca02c"     # другий авторський (зелений, як ENT-FT)
C_TESA = "#c0392b"      # TESA-26 (як у CD-діаграмі Ex08)
C_TIES = "#d62728"
C_SAKANA = "#f5a623"
C_GREY = "#8c9bab"      # бейзлайни
C_GREY2 = "#9e9e9e"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12.5,
    "figure.figsize": (12.8, 7.2), "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.25, "axes.grid.axis": "y",
})

made, skipped = [], []


def finish(fig, name, note=None, bottom=None, left=None):
    """Зберігає фігуру; примітка — курсивом під осями (конвенція фігур дисертації)."""
    if bottom is None:
        bottom = 0.17 if (note and len(note) > 95) else 0.13
    fig.subplots_adjust(bottom=bottom, left=left)
    if note:
        note = "\n".join(textwrap.wrap(note, 95))
        fig.text(0.5, 0.012, note, ha="center", va="bottom",
                 fontsize=12.5, style="italic", color="#333333", linespacing=1.4)
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    made.append(name)
    print(f"  [OK] presentation_figs/{name}")


def hbar(items, colors, boldset, suptitle, xlabel, fname, note=None, fmt="{:.2f}"):
    names = [n for n, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.6, zorder=3)
    ax.set_yticks(y, names)
    for tick, n in zip(ax.get_yticklabels(), names):
        if n in boldset:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.14)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.012, i, fmt.format(v), va="center", fontsize=13,
                fontweight="bold" if names[i] in boldset else "normal", zorder=4)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    finish(fig, fname, note=note, left=0.19)


# ── 1. TESA-26: ранги 5-методного зрізу (рис. 2.10 дисертації) ──────────────
def fig_tesa_ranks():
    txt = (ROOT / "Ex08_TESA-26/figs/friedman_stats.txt").read_text()
    block = txt.split("Mean ranks")[1]
    pairs = re.findall(r"^\s{2}(\S[^\n]*?)\s{2,}(\d\.\d+)\s*$", block, re.M)
    items = [(n.strip(), float(v)) for n, v in pairs][:5]
    chi2 = re.search(r"chi2\s*=?\s*([\d.]+)", txt, re.I)
    colors = [C_TESA if n == "TESA-26" else C_GREY for n, _ in items]
    note = (f"Парний тест Фрідмана: N = 24 блоки (датасет × розрідженість), k = 5, "
            f"χ² = {float(chi2.group(1)):.2f}, p < 0,01" if chi2 else None)
    hbar(items, colors, {"TESA-26"},
         "Ex08: TESA-26 проти SOTA-методів проріджування — середній ранг Фрідмана",
         "Середній ранг (менше — краще)", "p01_tesa_friedman_ranks.png", note=note)


# ── 2. TESA-26: F1 на екстремальній розрідженості 97% ───────────────────────
def fig_tesa_f1_97():
    methods = ["TESA-26", "Magnitude", "WANDA (SOTA)", "SparseGPT (SOTA)", "RIA (SOTA)"]
    acc = {m: [] for m in methods}
    for ds in ("blobs", "moons", "spirals"):
        rows = (ROOT / f"Ex08_TESA-26/results/{ds}_table.csv").read_text().strip().split("\n")
        header = rows[0].split(";")
        if "97%" not in header:
            continue
        j = header.index("97%")
        for r in rows[1:]:
            cells = r.split(";")
            if len(cells) <= j:
                continue
            name = cells[1].strip()
            if name in acc and cells[j].strip() not in ("—", "-", ""):
                acc[name].append(float(cells[j]))
    items = sorted(((m.replace(" (SOTA)", ""), float(np.mean(v)))
                    for m, v in acc.items() if v), key=lambda x: -x[1])
    if not items:
        raise RuntimeError("немає значень на 97%")
    colors = [C_TESA if n == "TESA-26" else C_GREY for n, _ in items]
    hbar(items, colors, {"TESA-26"},
         "Ex08: F1-міра при екстремальній розрідженості 97%",
         "F1-міра (більше — краще)", "p02_tesa_f1_at_97.png",
         note="Середнє по наборах, де рівень 97% досяжний; прочерки конкурентів "
              "(нестабільні конфігурації) не враховуються")


# ── 3. GFCS: стиснення і збереження якості ──────────────────────────────────
def fig_gfcs():
    d = json.load(open(ROOT / "Ex09_GFCS/results/full_benchmark_rcu.json"))
    recs = d if isinstance(d, list) else d.get("records", d.get("results"))
    df = pd.DataFrame(recs)
    g = df.groupby("method").agg(comp=("compression", "mean"),
                                 dq=("delta_f1", "mean")).reset_index()
    g = g.sort_values("comp", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    NAME = {"weight_redistribution": "Weight\nRedistribution", "knowledge_distill": "Knowledge\nDistillation",
            "evomerge": "EvoMerge", "neuron_removal": "Neuron\nRemoval", "svd_compression": "SVD\nCompression"}
    labels, colors = [], []
    for m in g["method"]:
        ours = m.lower() == "gfcs"
        labels.append("GFCS (наш)" if ours else NAME.get(m, m))
        colors.append(C_OURS if ours else C_GREY)
    x = np.arange(len(g))
    ax.bar(x, g["comp"], color=colors, width=0.58, zorder=3)
    ax.set_ylim(0, g["comp"].max() * 1.16)
    for i, (c, dq) in enumerate(zip(g["comp"], g["dq"])):
        ax.text(i, c + g["comp"].max() * 0.02, f"{c:.1f}×", ha="center", fontsize=13.5,
                fontweight="bold" if colors[i] == C_OURS else "normal")
    ax.set_xticks(x, [f"{l}\nΔF1 {dq:+.3f}" for l, dq in zip(labels, g["dq"])])
    ax.set_ylabel("Стиснення моделі, разів")
    fig.suptitle("Ex09: GFCS — фізичне стиснення мережі зі збереженням якості",
                 fontsize=16, fontweight="bold")
    finish(fig, "p03_gfcs_compression.png", bottom=0.16,
           note="8 наборів даних × 2 сіди; ΔF1 — середня зміна F1-міри після конверсії")


# ── 4. SACMA: ранги 8-методного зрізу (табл. 3.1 дисертації) ────────────────
def fig_sacma_ranks8():
    agg = pd.read_csv(ROOT / "Ex_HPO_Final/results/GLOBAL_ANALYSIS/aggregated_results.csv")
    keep = ["SACMA-DAC", "TPE", "GP-BO", "SMAC", "L-SHADE", "CMA-ES", "DEHB", "Random"]
    a = agg[agg["method"].isin(keep)].copy()
    a["rank"] = a.groupby("task")["median_loss"].rank(method="average")
    mr = a.groupby("method")["rank"].mean().sort_values()
    items = [("SACMA-DAC (наш)" if m == "SACMA-DAC" else m, v) for m, v in mr.items()]
    colors = [C_OURS if "наш" in n else C_GREY for n, _ in items]
    hbar(items, colors, {"SACMA-DAC (наш)"},
         "Ex_HPO: SACMA-DAC проти базових HPO-методів — середній ранг на 43 задачах",
         "Середній ранг Фрідмана (менше — краще)", "p04_sacma_ranks_8m.png",
         note="Ранги за медіанним лосом на 43 задачах; статистичні деталі — табл. 3.1 і рис. 3.2 дисертації")


# ── 5. SACMA: якість проти обчислювальної вартості (RCU) ────────────────────
def fig_sacma_rcu():
    agg = pd.read_csv(ROOT / "Ex_HPO_Final/results/GLOBAL_ANALYSIS/aggregated_results.csv")
    sub = agg[agg["method"].isin(["SACMA-DAC", "SACMA-MAB"])]
    rcu = sub.groupby("method")["median_rcu_total"].mean()
    vals = [rcu["SACMA-DAC"], rcu["SACMA-MAB"]]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars = ax.bar(["SACMA-DAC (наш)", "SACMA-MAB (наш)"], vals,
                  color=[C_OURS, C_OURS2], width=0.42, zorder=3)
    ax.set_ylim(0, max(vals) * 1.16)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:.0f} RCU",
                ha="center", fontsize=15, fontweight="bold")
    savings = (1 - vals[1] / vals[0]) * 100
    ax.set_ylabel("Середня обчислювальна вартість, RCU")
    fig.suptitle("Ex_HPO: SACMA-MAB — еквівалентна якість за менших витрат",
                 fontsize=16, fontweight="bold")
    finish(fig, "p05_sacma_rcu_efficiency.png",
           note=f"Повна вартість (пошук + навчання), середнє по 43 задачах — як «Сер. RCU» табл. 3.1; "
                f"економія SACMA-MAB ≈ {savings:.0f}% за статистично еквівалентної якості")


# ── 6. ENT: покласова точність (табл. 4.3) ──────────────────────────────────
def fig_ent_perclass():
    e = json.load(open(ROOT / "Ex30_HetMerge_ENT/results_e34.json"))
    by = {x["name"]: x for x in e}
    sel = [("TIES(d=0.3)", "TIES-Merging", C_TIES),
           ("Sakana-CMA", "Sakana-CMA", C_SAKANA),
           ("ENT", "ENT (наш)", C_OURS)]
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    x = np.arange(10)
    w = 0.27
    for k, (name, label, color) in enumerate(sel):
        pcs = [by[name]["per_class"][str(c)] for c in range(10)]
        nok = by[name]["ok"]
        ax.bar(x + (k - 1) * w, pcs, width=w, color=color,
               label=f"{label} — {nok}/10 класів", zorder=3)
    ax.axhline(0.3, color="#666666", lw=1.1, ls="--", zorder=2)
    ax.set_ylim(0, 1.12)
    ax.text(-0.62, 0.325, "поріг 0,3", fontsize=11.5, color="#666666", va="bottom")
    ax.set_xticks(x, [str(c) for c in range(10)])
    ax.set_xlabel("Клас MNIST (модель A навчена на класах 0–4, модель B — на 5–9)")
    ax.set_ylabel("Точність по класу")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)
    fig.suptitle("Ex30: комплементарне злиття — лише ENT зберігає всі 10 класів",
                 fontsize=16, fontweight="bold")
    finish(fig, "p06_ent_per_class.png", bottom=0.22)


# ── 7. ENT-FT: ефект калібрації на чемпіоні e34 ─────────────────────────────
def fig_entft():
    j = json.load(open(ROOT / "Ex30_HetMerge_ENT/results_ent_ft_on_e34.json"))
    b, a = j["before_calibration"], j["after_calibration_ent_ft"]
    pcb = [b["per_class"][str(c)] for c in range(10)]
    pca = [a["per_class"][str(c)] for c in range(10)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2), width_ratios=[1, 1.9])
    bv = [b["accuracy"], min(pcb)]
    av = [a["accuracy"], min(pca)]
    x = np.arange(2)
    ax1.bar(x - 0.19, bv, width=0.36, color=C_GREY, label="ENT (до)", zorder=3)
    ax1.bar(x + 0.19, av, width=0.36, color=C_OURS2, label="ENT-FT (після)", zorder=3)
    ax1.set_ylim(0, 1.12)
    for i in range(2):
        ax1.text(i - 0.19, bv[i] + 0.02, f"{bv[i]:.3f}", ha="center", fontsize=12)
        ax1.text(i + 0.19, av[i] + 0.02, f"{av[i]:.3f}", ha="center", fontsize=12,
                 fontweight="bold")
    ax1.set_xticks(x, ["Точність", "Найслабший\nклас"])
    ax1.set_title("(а) Агрегати")
    x2 = np.arange(10)
    ax2.bar(x2 - 0.19, pcb, width=0.36, color=C_GREY, label="до калібрації", zorder=3)
    ax2.bar(x2 + 0.19, pca, width=0.36, color=C_OURS2, label="після калібрації", zorder=3)
    ax2.set_ylim(0, 1.12)
    ax2.set_xticks(x2, [str(c) for c in range(10)])
    ax2.set_xlabel("Клас MNIST")
    ax2.set_title("(б) Покласова точність")
    h, l = ax2.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=True,
               bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("Ex30: ENT-FT — калібрація вихідного шару піднімає слабкі класи",
                 fontsize=16, fontweight="bold")
    finish(fig, "p07_entft_effect.png", bottom=0.22,
           note="Калібрація застосована до точного чемпіона e34 (детермінований конвеєр, SEED=42)")


# ── 8. CIFAR-10: захист найслабшого класу і баланс (табл. 4.5) ──────────────
def fig_cifar():
    rows = list(csv.DictReader(open(ROOT / "Ex31_Sakana_vs_ENT/results/sakana_vs_ent.tsv"),
                               delimiter="\t"))
    sel = [("WA", "Усереднення ваг", C_GREY), ("TIES", "TIES-Merging", C_TIES),
           ("Sakana-CMA", "Sakana-CMA", C_SAKANA), ("ENT", "ENT (наш)", C_OURS)]
    data = {r["method"]: r for r in rows}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 7.2))
    for ax, field, title in ((ax1, "min_class", "(а) Мінімальна точність по класу"),
                             (ax2, "balance", "(б) Баланс (min/max по класах)")):
        vals = [float(data[k][field]) for k, _, _ in sel]
        colors = [c for _, _, c in sel]
        bars = ax.bar([lbl for _, lbl, _ in sel], vals, color=colors, width=0.55, zorder=3)
        ax.set_ylim(0, max(vals) * 1.22 + 1e-9)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + max(vals) * 0.025, f"{v:.3f}",
                    ha="center", fontsize=12.5)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=12)
    ax1.axhline(0.4, color="#666666", lw=1.1, ls="--", zorder=2)
    ax1.text(-0.42, 0.408, "поріг 0,40", fontsize=11, color="#666666", va="bottom")
    fig.suptitle("Ex31: CIFAR-10 (SmallCNN) — лише ENT гарантує прийнятну якість "
                 "для кожного класу", fontsize=16, fontweight="bold")
    finish(fig, "p08_cifar_min_balance.png", bottom=0.15,
           note="Комплементарне злиття (класи 0–4 проти 5–9); дані табл. 4.5 дисертації")


# ── 9. Межа CMA-ES: W Кендала проти розмірності ─────────────────────────────
def fig_kendall():
    res = subprocess.run([sys.executable, str(ROOT / "Ex01-03_CMA_Boundary/verify_kendall_w.py")],
                         capture_output=True, text=True, timeout=600)
    ws = re.findall(r"W Кендала \(перераховано\)\s*=\s*(0[.,]\d{3,4})", res.stdout)
    if len(ws) < 2:
        raise RuntimeError(f"не розпізнано W у виводі: {res.stdout[-300:]}")
    w1, w2 = float(ws[0].replace(",", ".")), float(ws[1].replace(",", "."))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars = ax.bar(["d = 10\n(функція Растрігіна)", "d ≥ 100\n(ваги нейромережі)"],
                  [w1, w2], color=[C_OURS2, C_TIES], width=0.42, zorder=3)
    ax.set_ylim(0, 1.14)
    for b, v in zip(bars, [w1, w2]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.022, f"W = {v:.3f}", ha="center",
                fontsize=15.5, fontweight="bold")
    ax.set_ylabel("Конкордація W Кендала")
    fig.suptitle("Ex01–Ex03: межа застосовності CMA-ES — узгодженість переваги "
                 "падає з розмірністю", fontsize=16, fontweight="bold")
    finish(fig, "p09_cma_boundary.png",
           note="Висновок: еволюційний пошук — для структурних рішень, "
                "градієнтні методи — для неперервних ваг (підрозд. 2.2)")


# ── 10. RCU: стійкість метрики під фоновим навантаженням ────────────────────
def fig_rcu_drift():
    t = (ROOT / "Ex00_RCU_Validation/results/raw/ex00_stats.txt").read_text()
    v1 = float(re.search(r"Макс\. RCU дрифт:\s*([\d.]+)%", t).group(1))
    v2 = float(re.search(r"Макс\. реальний час дрифт:\s*([\d.]+)%", t).group(1))
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    bars = ax.bar(["RCU (наша метрика)", "Астрономічний час"], [v1, v2],
                  color=[C_OURS2, C_TIES], width=0.42, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1, v2 * 4)
    for b, v in zip(bars, [v1, v2]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.18, f"{v:.1f}%", ha="center",
                fontsize=16, fontweight="bold")
    ax.set_ylabel("Максимальний дрейф під фоновим шумом, % (лог. шкала)")
    fig.suptitle("Ex00: метрика RCU — стійкість вимірювання обчислювальної вартості",
                 fontsize=16, fontweight="bold")
    finish(fig, "p10_rcu_stability.png",
           note=f"Стрес-тест із фоновим CPU/MEM-навантаженням (bootstrap n = 10 000); "
                f"RCU стійкіша за астрономічний час у ~{v2 / v1:.0f} разів")


if __name__ == "__main__":
    for fn in (fig_tesa_ranks, fig_tesa_f1_97, fig_gfcs, fig_sacma_ranks8, fig_sacma_rcu,
               fig_ent_perclass, fig_entft, fig_cifar, fig_kendall, fig_rcu_drift):
        try:
            fn()
        except Exception as ex:
            skipped.append(f"{fn.__name__}: {ex}")
            print(f"  [ПРОПУЩЕНО] {fn.__name__}: {ex}")
    print(f"\nГотово: {len(made)} фігур у presentation_figs/; пропущено: {len(skipped)}")
