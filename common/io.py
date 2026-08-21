# common/io.py — збереження графіків і таблиць (LaTeX + PNG) у папку експерименту.
# Таблиці та графіки оформлюються за ДСТУ 3008:2015 (Times New Roman, 12 pt).
# Table formatting (Ukraine): knowledge_base/rules/formatting_numbering.md
# Відступи/падінги: внутрішній відступ клітинки ≥1 мм, бокові марджини ≥5 мм, відступ під таблицею при підписі.
#
# Візуалізація таблиць: єдиний спільний модуль для всіх експериментів (Ex01, Ex02, Ex03, …).
# Використовуйте save_table_latex() та save_table_png() з common — не дублюйте логіку в ex0N_visualize.
# Ширина/висота клітинок визначаються автоматично від тексту; таблиці не роздуваються (обгортання лише при довгих рядках, мінімуми клітинок мінімальні).

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from common.style import DSTU_RC

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def ensure_dir(path: Path | str) -> Path:
    """Створює директорію, якщо її немає."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_output_dir(output_dir: Path | str) -> None:
    """
    Очищає папку з результатами від старих файлів перед генерацією нових.
    Видаляє всі файли у папці, але зберігає саму папку.
    
    Args:
        output_dir: Path до папки з результатами
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    
    # Видаляємо всі файли у папці
    for file_path in output_path.iterdir():
        if file_path.is_file():
            file_path.unlink()
            print(f"[IO] Removed old file: {file_path.name}")


def clean_pycache(root: Path | str) -> None:
    """
    Видаляє всі директорії __pycache__ під заданим коренем (каталог експерименту).
    Викликається при завершенні програми експерименту.
    """
    root_path = Path(root)
    if not root_path.exists():
        return
    for d in list(root_path.rglob("__pycache__")):
        if d.is_dir():
            try:
                shutil.rmtree(d)
            except OSError:
                pass


def save_figure(fig: Figure, filepath: Path | str) -> None:
    """Зберігає figure у файл (напр. output_dir / 'convergence.png')."""
    path = Path(filepath)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=plt.rcParams.get("savefig.dpi", 300), bbox_inches="tight")
    print(f"[IO] Saved figure: {path}")


def _caption_no_trailing_period(caption: str | None) -> str | None:
    """За ДСТУ/ВАК: підпис таблиці без крапки в кінці."""
    if not caption:
        return caption
    s = caption.rstrip()
    if s.endswith("."):
        return s[:-1].rstrip()
    return s


def format_adaptive_decimal(value: object) -> str:
    """
    Форматує число з адаптивною кількістю знаків після коми для протоколу/документу.
    Малі величини — більше знаків, великі — менше.
    Не-числа (NaN, ---, текст) повертаються як є (рядок).
    """
    if pd.isna(value) or value is None or (isinstance(value, str) and value.strip() in ("", "---", "—", "–")):
        return "---"
    if isinstance(value, str):
        return value
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_x = abs(x)
    if abs_x >= 100:
        return f"{x:.1f}"
    if abs_x >= 1:
        return f"{x:.2f}"
    if abs_x >= 0.01:
        return f"{x:.2f}" if abs_x >= 0.1 else f"{x:.3f}"
    if abs_x >= 0.0001:
        return f"{x:.4f}"
    return f"{x:.4g}"


def save_table_markdown(df: pd.DataFrame, filepath: Path | str) -> None:
    """Зберігає DataFrame у .md (таблиця для протоколу) з адаптивними знаками після коми."""
    path = Path(filepath)
    if path.suffix != ".md":
        path = path.with_suffix(".md")
    ensure_dir(path.parent)
    # Заголовок: | col1 | col2 | ...
    def cell_str(val: object) -> str:
        if pd.isna(val) or val is None:
            return "---"
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return format_adaptive_decimal(val)
        s = str(val).strip()
        return "---" if s in ("", "---", "—", "–") else s

    header = [str(c) for c in df.columns]
    sep = "| " + " | ".join(header) + " |"
    line = "|" + "|".join(["-------"] * len(header)) + "|"
    rows = ["| " + " | ".join(cell_str(v) for v in row) + " |" for _, row in df.iterrows()]
    body = "\n".join([sep, line] + rows)
    path.write_text(body, encoding="utf-8")
    print(f"[IO] Saved table (Markdown): {path}")


def save_table_latex(
    df: pd.DataFrame, filepath: Path | str, caption: str | None = None, **to_latex_kwargs: object
) -> None:
    """Save DataFrame to .tex (for article/thesis). Follow knowledge_base/rules/formatting_numbering.md."""
    path = Path(filepath)
    if path.suffix != ".tex":
        path = path.with_suffix(".tex")
    ensure_dir(path.parent)
    # Порожні клітинки (NaN, порожній рядок) — тире в LaTeX (---)
    to_latex_kwargs.setdefault("na_rep", "---")
    df_out = df.copy()
    for c in df_out.columns:
        if df_out[c].dtype == object or df_out[c].dtype.name == "string":
            df_out[c] = df_out[c].replace("", pd.NA)
    buf = df_out.to_latex(index=False, **to_latex_kwargs)
    buf = buf.replace("—", "---")
    cap = _caption_no_trailing_period(caption)
    if cap:
        buf = buf.replace("\\begin{tabular}", "\\begin{table}[t]\n\\centering\n\\caption{" + cap.replace("---", "—") + "}\n\\begin{tabular}", 1)
        buf = buf.replace("\\end{tabular}", "\\end{tabular}\n\\end{table}", 1)
    path.write_text(buf, encoding="utf-8")
    print(f"[IO] Saved table (LaTeX): {path}")


def _estimate_cell_size_inches(
    text: str,
    fontsize_pt: int,
    *,
    bold: bool = False,
) -> tuple[float, float]:
    """
    Орієнтовна ширина і висота тексту в дюймах (пропорційний шрифт, кирилиця).
    Враховує переноси рядків (\\n). Коефіцієнти підібрані для Times + українська (ширші літери).
    bold=True — запас для жирного заголовка (~+12% ширини).
    """
    lines = str(text).split("\n") or [""]
    # Ширина: кирилиця/латиниця — орієнтовно 0.6× fontsize (pt → inch)
    char_width_inch = (fontsize_pt * 0.60) / 72.0
    if bold:
        char_width_inch *= 1.12
    max_chars = max(len(ln) for ln in lines)
    width_inch = max_chars * char_width_inch
    # Висота: кількість рядків × висота рядка
    line_height_inch = (fontsize_pt * 1.25) / 72.0
    height_inch = len(lines) * line_height_inch
    return (max(0.25, width_inch), max(0.16, height_inch))


# --- Компактні таблиці: розміри від контенту, перенос лише по словах, текст не накладається на межі ---
_CHAR_W_INCH_12 = (12 * 0.58) / 72.0
_LINE_H_INCH_12 = (12 * 1.2) / 72.0
_TABLE_PAD_H = 0.04
_TABLE_PAD_V = 0.035
_TABLE_SIDE_MARGIN_INCH = 0.15
_TABLE_CAPTION_INCH = 0.5
_TABLE_BOTTOM_MARGIN_INCH = 0.06


def _wrap_at_words_only(text: str, soft_max_chars: int) -> str:
    """Перенос лише по пробілах; слова не розриваються. Довге слово залишається в один рядок."""
    s = str(text).strip() or "—"
    if len(s) <= soft_max_chars:
        return s
    words = s.split()
    if not words:
        return s
    lines, current = [], ""
    for w in words:
        if not current:
            current = w
            continue
        if len(current) + 1 + len(w) <= soft_max_chars:
            current = current + " " + w
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return "\n".join(lines)


def _cell_size_inch(lines: list[str], fontsize_pt: int, bold: bool = False) -> tuple[float, float]:
    """Ширина/висота від контенту (символи × char_width, рядки × line_height)."""
    if not lines:
        return (0.15, _LINE_H_INCH_12 * fontsize_pt / 12.0)
    cw = _CHAR_W_INCH_12 * (fontsize_pt / 12.0) * (1.15 if bold else 1.0)
    lh = _LINE_H_INCH_12 * fontsize_pt / 12.0
    return (max(len(ln) for ln in lines) * cw, len(lines) * lh)


def _render_table_to_png(
    header: list[str],
    rows: list[list[str]],
    path: Path,
    *,
    caption: str | None = None,
    fontsize: int = 12,
    dpi: int = 300,
    soft_wrap: int | list[int] = 50,
    cell_colors: list[list[str | None]] | None = None,
) -> None:
    """Відмальовує сітку (header + rows) у PNG: компактність, без накладання на межі, без розриву слів."""
    n_cols = len(header)
    if not n_cols or any(len(r) != n_cols for r in rows):
        raise ValueError("header і rows повинні мати однакову кількість колонок")
    wrap_per_col = soft_wrap if isinstance(soft_wrap, list) and len(soft_wrap) == n_cols else [soft_wrap if isinstance(soft_wrap, int) else 50] * n_cols
    def norm(c: str) -> str:
        t = str(c).strip()
        return t if t else "—"
    header = [norm(h) for h in header]
    rows = [[norm(c) for c in r] for r in rows]
    header_w = [_wrap_at_words_only(header[j], wrap_per_col[j]) for j in range(n_cols)]
    data_w = [[_wrap_at_words_only(rows[i][j], wrap_per_col[j]) for j in range(n_cols)] for i in range(len(rows))]
    grid = [header_w] + data_w
    n_rows = len(grid)
    cell_w = [[0.0] * n_cols for _ in range(n_rows)]
    cell_h = [[0.0] * n_cols for _ in range(n_rows)]
    for i, row_cells in enumerate(grid):
        for j, cell in enumerate(row_cells):
            lines = cell.split("\n")
            w, h = _cell_size_inch(lines, fontsize, bold=(i == 0))
            cell_w[i][j] = w + 2 * _TABLE_PAD_H
            cell_h[i][j] = h + 2 * _TABLE_PAD_V
    col_w = [max(cell_w[i][j] for i in range(n_rows)) for j in range(n_cols)]
    row_h = [max(cell_h[i][j] for j in range(n_cols)) for i in range(n_rows)]
    min_w = _CHAR_W_INCH_12 * (fontsize / 12.0) + 2 * _TABLE_PAD_H
    min_h = _LINE_H_INCH_12 * (fontsize / 12.0) + 2 * _TABLE_PAD_V
    col_w = [max(w, min_w) for w in col_w]
    row_h = [max(h, min_h) for h in row_h]
    total_w = sum(col_w)
    total_h = sum(row_h)
    fig_w = total_w + 2 * _TABLE_SIDE_MARGIN_INCH
    fig_h = total_h + (_TABLE_CAPTION_INCH if caption else 0.0) + _TABLE_BOTTOM_MARGIN_INCH
    with plt.rc_context(rc=DSTU_RC):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.axis("off")
        # Позиція осей під пропорції таблиці (total_w × total_h), щоб PNG не роздувався по ширині/висоті
        ax_left = _TABLE_SIDE_MARGIN_INCH / fig_w
        ax_bottom = _TABLE_BOTTOM_MARGIN_INCH / fig_h
        ax_width = total_w / fig_w
        ax_height = total_h / fig_h
        ax.set_position([ax_left, ax_bottom, ax_width, ax_height])
        table = ax.table(
            cellText=data_w,
            colLabels=header_w,
            loc="center",
            cellLoc="center",
            colWidths=[w / total_w for w in col_w],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1.0, 1.0)
        for key, cell in table.get_celld().items():
            ri, ci = key[0], key[1]
            cell.set_height(row_h[ri] / total_h)
            cell.PAD = 0.02
            if cell_colors and ri < len(cell_colors) and ci < len(cell_colors[ri]) and cell_colors[ri][ci] is not None:
                cell.set_facecolor(cell_colors[ri][ci])
            cell.set_text_props(ha="center", va="center", wrap=True)
            if ri == 0:
                cell.set_text_props(weight="bold", ha="center", va="center", wrap=True)
        if caption:
            cap = _caption_no_trailing_period(caption) or caption
            fig.suptitle(cap, fontsize=fontsize, fontweight="bold", y=0.98)
        # Не використовуємо tight_layout і bbox_inches="tight" — зберігають пропорції fig_w/fig_h
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    print(f"[IO] Saved table (PNG): {path}")


# Константи (legacy, для сумісності)
_TABLE_CELL_PADDING_PX = 24
_TABLE_TOP_CAPTION_INCH = 0.45  # зона для підпису зверху (включає висоту підпису + зазор)
_TABLE_CAPTION_TO_TABLE_GAP_INCH = 0.17  # мін. відстань заголовок–таблиця: 1 інтервал при 12 pt ≈ 4,3 мм
_TABLE_CAPTION_PLUS_GAP_INCH = 0.38  # мін. зона зверху: висота підпису (~0,21) + зазор (~0,17)
_TABLE_BOTTOM_MARGIN_INCH = 0.15  # відступ знизу при наявному підписі ≈4 мм
_TABLE_BOTTOM_MARGIN_NO_CAPTION_INCH = 0.08  # відступ знизу без підпису
_TABLE_CELL_PAD_FRAC = 0.02  # додатковий внутрішній зазор клітинки (2 % висоти) — уникнення «прилипання» тексту
_TABLE_MIN_VERTICAL_PADDING_PX = 5  # мін. відступ по висоті від тексту до межі клітинки (при багатьох рядках — менша висота клітинок)


def save_table_png(
    df: pd.DataFrame,
    filepath: Path | str,
    caption: str | None = None,
    fontsize: int = 12,
    *,
    cell_padding_px: int = _TABLE_CELL_PADDING_PX,
    dpi: int = 300,
    cell_colors: list[list[str | None]] | None = None,
    col_wrap_max_chars: list[int] | None = None,
) -> None:
    """
    Рендерить таблицю в PNG: компактність, перенос лише по словах (без розриву слів), дані не накладаються на межі.
    Times New Roman 12 pt, порожні клітинки — «—», підпис без крапки, 300 DPI.
    cell_colors: опційно (1 + n_rows) x n_cols. col_wrap_max_chars: опційно список n_cols — м'який макс. символів на рядок.
    """
    path = Path(filepath)
    if path.suffix != ".png":
        path = path.with_suffix(".png")
    ensure_dir(path.parent)

    n_cols = len(df.columns)

    def _col_label(c: object) -> str:
        if isinstance(c, tuple):
            return f"{c[0]}\n{c[1]}" if c[1] else str(c[0])
        return str(c)

    header = [_col_label(c) for c in df.columns]
    rows = [
        [str(val) if pd.notna(val) and str(val).strip() != "" else "—" for val in row]
        for row in df.values
    ]
    soft_wrap: int | list[int] = (
        col_wrap_max_chars
        if col_wrap_max_chars is not None and len(col_wrap_max_chars) == n_cols
        else 50
    )
    _render_table_to_png(
        header,
        rows,
        path,
        caption=caption,
        fontsize=fontsize,
        dpi=dpi,
        soft_wrap=soft_wrap,
        cell_colors=cell_colors,
    )
