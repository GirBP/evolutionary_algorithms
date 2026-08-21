# Ex01-03_CMA_Boundary: фігура «межа застосовності CMA-ES» (§2.2 дисертації).
# Читає ЛИШЕ наявні файли результатів через verify_kendall_w.py (перерахунок
# W Кендала з сирих даних) і будує PNG з двома зведеними значеннями, які
# цитує §2.2: W = 0,943 (Ex01, Растрігін, d=10) та W = 0,700 (Ex03, ваги
# нейромережі, зведено по 3 датасетах). Жодне число не вводиться вручну.

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent          # Ex01-03_CMA_Boundary/
OUT = ROOT / "figs"
OUT.mkdir(exist_ok=True)


def main() -> None:
    res = subprocess.run([sys.executable, str(ROOT / "verify_kendall_w.py")],
                         capture_output=True, text=True, timeout=600)
    ws = re.findall(r"W Кендала \(перераховано\)\s*=\s*(0[.,]\d{3,4})", res.stdout)
    if len(ws) < 2:
        raise SystemExit(f"не розпізнано W у виводі verify_kendall_w.py: {res.stdout[-300:]}")
    w_low_d, w_high_d = float(ws[0].replace(",", ".")), float(ws[1].replace(",", "."))

    fig, ax = plt.subplots(figsize=(9.6, 6.0), dpi=150)
    bars = ax.bar(["d = 10\n(Ex01: функція Растрігіна)", "d ≥ 10²\n(Ex03: ваги нейромережі, зведено)"],
                  [w_low_d, w_high_d], color=["#2e7d4f", "#c0392b"], width=0.45, zorder=3)
    for b, v in zip(bars, [w_low_d, w_high_d]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"W = {v:.3f}",
                ha="center", fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Конкорданс W Кендала (ефект-розмір тесту Фрідмана)")
    ax.set_title("Межа застосовності CMA-ES за розмірністю задачі (§2.2 дисертації)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    fig.text(0.5, 0.015,
             "Обидва значення перераховуються з сирих даних скриптом verify_kendall_w.py",
             ha="center", fontsize=11, style="italic", color="#333333")
    fig.subplots_adjust(bottom=0.18)
    fig.savefig(OUT / "kendall_w_boundary.png", dpi=150)
    print(f"[OK] figs/kendall_w_boundary.png  (W: {w_low_d:.3f} / {w_high_d:.3f})")


if __name__ == "__main__":
    main()
