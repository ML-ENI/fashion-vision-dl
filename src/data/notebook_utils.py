"""Presentación compartida de las auditorías de datos; no contiene filtros."""
from pathlib import Path
import json
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.visualization.style import (
    BLUE, TEAL, CORAL, GRID, INK, MUTED, PAPER, SPLIT_COLORS,
    apply_style, format_number,
)
from src.visualization.themes import render, display_figure

SPLIT_COLOURS = SPLIT_COLORS


def es(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def setup():
    apply_style()
    # También los display(df) directos deben respetar el estilo compartido.
    # Solo cambia la representación HTML; no modifica los datos ni los cálculos.
    from IPython import get_ipython
    shell = get_ipython()
    if shell is not None:
        import pandas as pd
        from src.visualization.tables import report_table
        html = shell.display_formatter.formatters["text/html"]
        html.for_type(pd.DataFrame, lambda frame: report_table(frame).to_html())
        html.for_type(pd.Series, lambda series: report_table(series).to_html())


def numeric_axis(ax, axis="x", digits=0):
    getattr(ax, f"{axis}axis").set_major_formatter(FuncFormatter(lambda v, _: es(v, digits)))


def save_figure(fig, root, name, source, T, folder="data"):
    """Guardar una versión de la figura en results/figures/<estilo>/<carpeta>."""
    directory = Path(root) / "results/figures" / T.name / folder
    directory.mkdir(parents=True, exist_ok=True)
    # Como add_title en ML: sin un ajuste automático, add_source no puede
    # reservar sitio para la nota y quedan márgenes vacíos.
    if fig.get_layout_engine() is None:
        fig.set_layout_engine("constrained")
    # Los títulos largos se ajustan al ancho de la figura según la tipografía del estilo.
    title_width = max(T.min_title, int(fig.get_figwidth() * T.title_chars))
    if fig._suptitle is not None:
        # Dejar la posición automática permite que constrained_layout reserve
        # espacio por encima de las etiquetas de la primera fila de imágenes.
        fig.suptitle(textwrap.fill(fig._suptitle.get_text(), title_width),
                     fontsize=T.title_size, fontweight="bold", y=None)
    for ax in fig.axes:
        if ax.get_title():
            image_panel = bool(ax.images)
            panel_width = max(16, int(ax.get_position().width * fig.get_figwidth() * T.panel_chars))
            title = ("\n".join(textwrap.fill(line, panel_width) for line in ax.get_title().splitlines())
                     if image_panel else textwrap.fill(ax.get_title(), title_width))
            panel_size = T.pick(plt.rcParams["axes.titlesize"], T.title_size)
            ax.set_title(title,
                         fontsize=T.image_title_size if image_panel else panel_size,
                         fontweight="normal" if image_panel else "bold")
    # La nota puede depender del estilo, p. ej. cuando nombra un color.
    T.add_note(fig, source(T) if callable(source) else source)
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())


def show_figure(draw, root, name, source, width=None, folder="data"):
    """Dibujar la figura con los dos estilos, guardar ambos y mostrar el de ML.

    ``draw(T)`` recibe el tema y devuelve la figura. Los datos se calculan antes,
    fuera de ``draw``: aquí solo se repite el dibujo.
    """
    fig = render(draw, lambda figure, T: save_figure(figure, root, name, source, T, folder))
    display_figure(fig, width)
    return fig


def styled(frame, digits=0):
    from src.visualization.tables import report_table
    return report_table(frame, digits=digits)


def visual_thumbnail(row, raw_base, root):
    """Guardar solo la pequeña muestra de auditoría para consultar sin RAW."""
    from PIL import Image, ImageOps
    from data_contract import sha256, write_json
    directory = Path(root) / "data/audit/visual_samples"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{row.article_id}-{row.file_sha256[:16]}.jpg"
    info = path.with_suffix(".json")
    if path.exists() and info.exists():
        report = json.loads(info.read_text())
        if sha256(path) == report["prepared_sha256"] and report["source_sha256"] == row.file_sha256:
            with Image.open(path) as image:
                return image.convert("RGB")
        raise ValueError(f"Muestra visual modificada: {path.name}")
    source = Path(raw_base) / row.raw_path
    if sha256(source) != row.file_sha256:
        raise ValueError(f"Imagen RAW modificada: {row.article_id}")
    with Image.open(source) as image:
        thumb = ImageOps.exif_transpose(image).convert("RGB")
        thumb.thumbnail((320, 320))
        thumb.save(path, quality=90)
    write_json(info, {"source_sha256": row.file_sha256, "prepared_sha256": sha256(path),
                      "purpose": "visual explanation: training-side or excluded task category; never test"})
    with Image.open(path) as image:
        return image.convert("RGB")
