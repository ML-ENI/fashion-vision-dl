"""Dos estilos para las mismas figuras: ML (principal) y ETSIT (copia aparte).

Cada figura se escribe una sola vez como una función ``draw(T)`` que recibe un
tema ``T``. ``render`` la dibuja con cada estilo, usando los mismos datos ya
calculados, guarda ambas versiones y devuelve la de ML para mostrarla.

- ML: ``style.py``, copia literal de fashion-demand-ml. Se muestra en el notebook
  y se guarda en ``results/figures/ml/``.
- ETSIT: ``style_etsit.py``. Solo se guarda, en ``results/figures/etsit/``.

Los cálculos no se repiten: solo se repite el dibujo.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import textwrap
from types import ModuleType

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.visualization import style as ml_style
from src.visualization import style_etsit

SHOWN = "ml"
THEME_NAMES = ("ml", "etsit")


@dataclass(frozen=True)
class Theme:
    """Papeles de color y tamaños que cambian entre estilos."""

    name: str
    style: ModuleType
    primary: str        # serie principal
    secondary: str      # segunda serie o variable distinta
    reference: str      # referencia, «otras» o línea de comparación
    title_accent: str   # títulos que el estilo colorea (ETSIT: azul)
    split: dict         # colores fijos de train / validation / test
    ink: str
    muted: str
    grid: str
    paper: str
    sequential: LinearSegmentedColormap
    diverging: LinearSegmentedColormap
    title_size: float
    image_title_size: float   # títulos de paneles con fotos
    title_chars: float        # caracteres por pulgada en títulos
    panel_chars: float        # caracteres por pulgada en títulos de panel
    source_chars: float       # caracteres por pulgada en la nota de fuente
    min_title: int
    min_source: int

    def pick(self, ml, etsit):
        """Valor propio de cada estilo, p. ej. ``T.pick((13, 5.5), (16, 7))``."""
        return ml if self.name == "ml" else etsit

    def wrap_title(self, fig):
        if fig._suptitle is not None:
            width = max(self.min_title, int(fig.get_figwidth() * self.title_chars))
            fig._suptitle.set_text(textwrap.fill(fig._suptitle.get_text(), width))

    def add_note(self, fig, text):
        width = max(self.min_source, int(fig.get_figwidth() * self.source_chars))
        self.style.add_source(fig, textwrap.fill(text, width))


def _ml():
    s = ml_style
    return Theme(
        name="ml", style=s, primary=s.BLUE, secondary=s.TEAL, reference=s.MUTED,
        title_accent=s.INK, split=dict(s.SPLIT_COLORS), ink=s.INK, muted=s.MUTED,
        grid=s.GRID, paper=s.PAPER,
        sequential=LinearSegmentedColormap.from_list("sequential", [s.PAPER, s.BLUE]),
        diverging=LinearSegmentedColormap.from_list("diverging", [s.CORAL, s.PAPER, s.BLUE]),
        title_size=s.TITLE_SIZE, image_title_size=9,
        title_chars=9, panel_chars=11, source_chars=16, min_title=45, min_source=60,
    )


def _etsit():
    s = style_etsit
    return Theme(
        name="etsit", style=s, primary=s.BLUE, secondary=s.ORANGE, reference=s.ORANGE,
        title_accent=s.BLUE, split=dict(s.SPLIT_COLORS), ink=s.INK, muted=s.MUTED,
        grid=s.GRID, paper=s.PAPER,
        sequential=LinearSegmentedColormap.from_list("etsit_confusion", ["#FFFFFF", s.BLUE]),
        diverging=LinearSegmentedColormap.from_list("etsit_signed_weights", [s.ORANGE, "#FFFFFF", s.BLUE]),
        title_size=s.TITLE_SIZE, image_title_size=s.TICK_SIZE,
        title_chars=5, panel_chars=5.8, source_chars=10, min_title=32, min_source=45,
    )


_BUILDERS = {"ml": _ml, "etsit": _etsit}


def theme(name):
    return _BUILDERS[name]()


@contextmanager
def use_theme(name):
    """Aplicar un estilo solo mientras se dibuja; después se restaura el anterior."""
    with plt.rc_context():
        plt.rcdefaults()
        T = theme(name)
        T.style.apply_style()
        yield T


def render(draw, save):
    """Dibujar con cada estilo, guardar cada versión y devolver la figura de ML.

    ``draw(T)`` crea y devuelve la figura; ``save(fig, T)`` la guarda. Las
    figuras se cierran en pyplot para que el notebook no las muestre dos veces;
    la devuelta se puede seguir mostrando o guardando.
    """
    shown = None
    for name in THEME_NAMES:
        with use_theme(name) as T:
            fig = draw(T)
            save(fig, T)
            plt.close(fig)
            if name == SHOWN:
                shown = fig
    return shown


def display_figure(fig, width=None):
    """Mostrar una figura ya dibujada; con ``width`` se acota su tamaño en pantalla."""
    from io import BytesIO
    from IPython.display import Image as DisplayImage, display
    if width is None:
        display(fig)
        return
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    display(DisplayImage(data=buffer.getvalue(), width=width))
