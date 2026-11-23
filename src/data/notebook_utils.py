"""Presentación compartida de las auditorías de datos; no contiene filtros."""
from pathlib import Path
import json
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from src.visualization.style import (
    BLUE, ORANGE, SPLIT_COLORS, TITLE_SIZE, TICK_SIZE,
    apply_style, add_source, format_number,
)

SPLIT_COLOURS = SPLIT_COLORS


def es(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def setup():
    apply_style()


def numeric_axis(ax, axis="x", digits=0):
    getattr(ax, f"{axis}axis").set_major_formatter(FuncFormatter(lambda v, _: es(v, digits)))


def save_figure(fig, root, name, source):
    directory = Path(root) / "results/figures/data"
    directory.mkdir(parents=True, exist_ok=True)
    # Los títulos largos se ajustan al ancho con la tipografía grande de ML.
    title_width = max(32, int(fig.get_figwidth() * 5))
    if fig._suptitle is not None:
        # Dejar la posición automática permite que constrained_layout reserve
        # espacio por encima de las etiquetas de la primera fila de imágenes.
        fig.suptitle(textwrap.fill(fig._suptitle.get_text(), title_width),
                     fontsize=TITLE_SIZE, fontweight="bold", y=None)
    for ax in fig.axes:
        if ax.get_title():
            image_panel = bool(ax.images)
            panel_width = max(16, int(ax.get_position().width * fig.get_figwidth() * 5.8))
            title = ("\n".join(textwrap.fill(line, panel_width) for line in ax.get_title().splitlines())
                     if image_panel else textwrap.fill(ax.get_title(), title_width))
            ax.set_title(title,
                         fontsize=TICK_SIZE if image_panel else TITLE_SIZE,
                         fontweight="normal" if image_panel else "bold")
    add_source(fig, textwrap.fill(source, max(45, int(fig.get_figwidth() * 10))))
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight", facecolor=fig.get_facecolor())


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
