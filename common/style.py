# common/style.py — єдиний стиль візуалізації для наукової роботи (повторюваність).
# ДСТУ 3008:2015 «Звіти у сфері науки і техніки»: шрифт Times New Roman, 12–14 pt.

import matplotlib.pyplot as plt


# --- Єдина візуальна норма: співвідношення сторін і розміри фігур ---
# Усі графіки експериментів варто будувати через create_figure(), щоб не хардкодити figsize.
FIG_ASPECT_WIDE = 1.6          # ширина/висота для основних графіків (збіжність, accuracy vs time, розподіл)
FIG_ASPECT_FRIEDMAN = 2.0     # для діаграми рангів Фрідмана (горизонтальні бари)
FIG_ASPECT_LANDSCAPE = 5 / 3.5
FIG_WIDTH_WIDE = 10.0         # дюйми
FIG_WIDTH_WIDE_LEGEND_OUTSIDE = 12.0  # ширша фігура, коли легенда справа — щоб зона даних не стискалась
FIG_RECT_RIGHT_LEGEND_OUTSIDE = 0.82  # частка ширини під область малюнка (решта — під легенду)
FIG_WIDTH_FRIEDMAN = 8.0
FIG_WIDTH_LANDSCAPE = 5.0


def create_figure(kind="wide", legend_outside=False):
    """Створити фігуру з єдиним співвідношенням сторін (візуальна норма).
    kind: 'wide' | 'friedman' | 'landscape'. legend_outside: для 'wide' — ширша фігура і rect для tight_layout.
    Повертає (fig, ax, rect): rect — для plt.tight_layout(rect=rect), якщо legend_outside=True для wide, інакше None."""
    if kind == "wide":
        if legend_outside:
            w, h = FIG_WIDTH_WIDE_LEGEND_OUTSIDE, FIG_WIDTH_WIDE_LEGEND_OUTSIDE / FIG_ASPECT_WIDE
            rect = (0, 0, FIG_RECT_RIGHT_LEGEND_OUTSIDE, 1)
        else:
            w, h = FIG_WIDTH_WIDE, FIG_WIDTH_WIDE / FIG_ASPECT_WIDE
            rect = None
    elif kind == "friedman":
        w = FIG_WIDTH_FRIEDMAN
        h = FIG_WIDTH_FRIEDMAN / FIG_ASPECT_FRIEDMAN
        rect = None
    elif kind == "landscape":
        w = FIG_WIDTH_LANDSCAPE
        h = FIG_WIDTH_LANDSCAPE / FIG_ASPECT_LANDSCAPE
        rect = None
    else:
        w, h = FIG_WIDTH_WIDE, FIG_WIDTH_WIDE / FIG_ASPECT_WIDE
        rect = None
    fig = plt.figure(figsize=(w, h))
    ax = fig.gca()
    return fig, ax, rect


# Параметри за ДСТУ 3008:2015 (таблиці, графіки, підписи — не менше 12 pt)
DSTU_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "lines.linewidth": 2,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": True,
}


def set_dstu_style():
    """Стиль оформлення графіків і таблиць за ДСТУ 3008:2015 (Times New Roman, 12 pt)."""
    plt.style.use("default")
    plt.rcParams.update(DSTU_RC)


def set_thesis_style():
    """Єдиний стиль оформлення графіків для дисертації/статей (за ДСТУ 3008:2015)."""
    set_dstu_style()


def legend_outside(ax, side="right", **kwargs):
    """Розмістити легенду за межами області малюнка, щоб не перекривати дані.
    Рекомендовано для scatter/line графіків з щільними даними.
    side: 'right' | 'left'. Решта kwargs передаються в ax.legend()."""
    if side == "right":
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), **kwargs)
    else:
        ax.legend(loc="upper right", bbox_to_anchor=(-0.02, 1), **kwargs)


# Стандартизована палітра для методів (можна розширювати в наступних експериментах)
PALETTE = {
    "Adam (GD)": "#D32F2F",
    "CMA-ES (EA)": "#2E8B57",
    "Random Search": "#757575",
}
