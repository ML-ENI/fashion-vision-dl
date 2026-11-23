"""Shared chart style and readable labels for every project notebook.

The visual rules live in ``docs/visualization-rubric.md``. The user's report
preferences override its base typography and palette here: Latin Modern,
large text, white backgrounds, and the blue/orange sampled from their logo.
Use ``apply_style(n_categories=...)`` for figures needing additional colours.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from cycler import cycler
from matplotlib.dates import DateFormatter, ConciseDateFormatter, AutoDateFormatter
from matplotlib.ticker import FuncFormatter, PercentFormatter, ScalarFormatter
import numpy as np

PAPER, INK, MUTED, GRID = "#FFFFFF", "#17212B", "#667085", "#DDE3EA"
# RGB colours sampled from the user's ETSIT-UPM image.
BLUE, ORANGE = "#00458A", "#F18D01"
TEAL, CORAL, GOLD, VIOLET = "#16A085", "#E76F51", ORANGE, "#7357C8"
CYAN = "#0099CC"
BASE_PALETTE = [BLUE, ORANGE]
SPLIT_COLORS = {"train": "#6F4EAF", "validation": "#E17C05", "test": "#00A6D6"}
CATEGORY_PALETTE = [
    BLUE, ORANGE, "#00AA00", "#FF0011", "#7357C8",
    "#0099CC", "#884477", "#008866", "#DD7700", "#AA11DD",
    "#AA0033", "#999900", "#0055AA", "#CC7777", "#336600",
    "#CC66DD", "#993300", "#9988CC", "#FF2277", "#889955",
]
MODEL_COLORS = {
    "Ridge": BLUE,
    "Logistic regression": BLUE,
    "Decision tree": ORANGE,
    "Random Forest": "#00AA00",
    "Boosting · squared error": "#FF0011",
    "Boosting · absolute error": VIOLET,
    "Boosting · Poisson": CYAN,
}

FEATURE_LABELS = {
    "sales": "Current-week sales",
    **{f"lag_{lag}_sales": f"Sales {lag} week{'s' if lag > 1 else ''} ago"
       for lag in range(1, 13)},
    "rolling_4w_sales_mean": "4-week mean sales",
    "rolling_4w_sales_std": "4-week sales variability",
    "rolling_4w_active_weeks": "Active weeks in 4-week history",
    "rolling_12w_sales_mean": "12-week mean sales",
    "rolling_12w_sales_std": "12-week sales variability",
    "rolling_12w_active_weeks": "Active weeks in 12-week history",
    "sales_trend_4w": "4-week sales trend",
    "mean_price": "Current-week mean price",
    "rolling_4w_mean_price": "4-week mean price",
    "price_change_from_4w_mean": "Price change versus 4-week mean",
    "online_share": "Current-week online share",
    "rolling_4w_online_share": "4-week online share",
    "iso_week": "ISO week",
    "week_sin": "Seasonal week (sine)",
    "week_cos": "Seasonal week (cosine)",
    "product_type_name": "Product type",
    "product_group_name": "Product group",
    "index_group_name": "Index group",
    "section_name": "Section",
    "garment_group_name": "Garment group",
    "colour_group_name": "Colour group",
    "target_next_week_sales": "Next-week sales",
    "target_high_demand": "High next-week demand",
}


def feature_label(feature):
    return FEATURE_LABELS[feature]


TITLE_SIZE = 27
LABEL_SIZE = 21
TICK_SIZE = 17
LEGEND_SIZE = 16
SOURCE_SIZE = 12
FONT_FILE = (
    Path.home() / "AppData/Local/Microsoft/Windows/Fonts/lmroman10-regular.ttf"
)


def category_colors(count):
    """Blue/orange for up to two categories; distinct colours for larger sets.

    Keep the category order fixed across related figures. Beyond twenty
    categories, also distinguish series with markers or line styles.
    """
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError("The category count must be an integer.")
    if count < 0:
        raise ValueError("The category count must be non-negative.")
    return [CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)] for i in range(count)]


def _report_font():
    """Register regular/bold/italic Latin Modern faces when locally available."""
    if FONT_FILE.is_file():
        for face in ("regular", "bold", "italic", "bolditalic"):
            path = FONT_FILE.with_name(f"lmroman10-{face}.ttf")
            if path.is_file():
                font_manager.fontManager.addfont(str(path))
        return font_manager.FontProperties(fname=str(FONT_FILE)).get_name()
    available = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in ["Latin Modern Roman", "LM Roman 10",
                                  "DejaVu Serif", "Liberation Serif"]
                 if name in available), "DejaVu Serif")


def _localize_numeric_text(text):
    """Swap Python's numeric separators without touching non-numeric content."""
    return text.replace(",", "\N{UNIT SEPARATOR}").replace(".", ",").replace(
        "\N{UNIT SEPARATOR}", "."
    )


def format_number(value, decimals=None, signed=False):
    """Format a number with ``.`` for thousands and ``,`` for decimals."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        return str(value)
    if decimals is None:
        if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
            value = int(value)  # 3.0 reads as "3", not "3,0".
            rendered = format(value, ",")
        else:
            rendered = format(value, ",.3f").rstrip("0").rstrip(".")
    else:
        specification = f"{('+' if signed else '')},.{decimals}f"
        rendered = format(value, specification)
    return _localize_numeric_text(rendered)


def format_percent(value, decimals=1):
    """Format a ratio as a percentage using Spanish numeric separators."""
    return f"{format_number(100 * value, decimals)} %"


class _SpanishNumber:
    def __init__(self, value):
        self.value = value

    def __format__(self, specification):
        if specification.endswith("%"):
            decimals = specification.partition(".")[2].removesuffix("%")
            return format_percent(self.value, int(decimals) if decimals else 0)
        return _localize_numeric_text(format(self.value, specification))


def es(value):
    """Allow localized values inside f-strings while preserving their precision."""
    return _SpanishNumber(value)


def display_es(value):
    """Mostrar tablas con el formato común legible y números en español."""
    from IPython.display import display
    from src.visualization.tables import report_table
    if hasattr(value, 'style') or value.__class__.__name__ in {'Styler', 'Series'}:
        value = report_table(value)
    display(value)


def localize_axis_numbers(ax):
    """Apply Spanish separators to numeric and percentage ticks on one axis."""
    date_formatters = (DateFormatter, ConciseDateFormatter, AutoDateFormatter)
    for axis in (ax.xaxis, ax.yaxis):
        formatter = axis.get_major_formatter()
        if isinstance(formatter, PercentFormatter):
            maximum = formatter.xmax
            decimals = formatter.decimals
            axis.set_major_formatter(FuncFormatter(
                lambda value, _position, maximum=maximum, decimals=decimals:
                format_percent(value / maximum, 0 if decimals is None else decimals)
            ))
        elif isinstance(formatter, ScalarFormatter) and not isinstance(formatter, date_formatters):
            axis.set_major_formatter(FuncFormatter(lambda value, _position: format_number(value)))


def add_title(fig, title):
    """Main figure title; constrained layout keeps the same gap above every chart."""
    if fig.get_layout_engine() is None:
        fig.set_layout_engine("constrained")
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold", color=INK)


def add_source(fig, text):
    """Place the single source note at the bottom-left corner of the figure."""
    for ax in fig.axes:
        localize_axis_numbers(ax)
    engine = fig.get_layout_engine()
    if engine is not None and "rect" in engine.get():
        # Constrained layout would otherwise put the axis labels on the same line.
        left, bottom, width, height = engine.get()["rect"]
        reserved = 0.42 / fig.get_figheight()
        engine.set(rect=(left, bottom + reserved, width, height - reserved))
    fig.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=SOURCE_SIZE, color=MUTED)


def apply_style(n_categories=2):
    """Apply the report style; opt into extra colours for multi-category plots.

    ``apply_style()`` uses blue/orange. ``apply_style(n_categories=5)`` adds
    green, red and violet. Explicit per-artist styles take precedence.
    """
    colors = category_colors(n_categories) or BASE_PALETTE
    font = _report_font()
    plt.rcParams.update({
        "font.family": font, "font.size": 18,
        "font.weight": "normal", "text.usetex": False,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.labelsize": LABEL_SIZE, "axes.labelweight": "bold",
        "axes.labelpad": 12,
        "axes.titlesize": TITLE_SIZE, "axes.titleweight": "bold", "axes.titlepad": 16,
        "axes.titlelocation": "center",
        "axes.prop_cycle": cycler(color=colors),
        "figure.figsize": (11, 6.5), "figure.dpi": 110,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.14,
        "figure.constrained_layout.w_pad": 0.12,
        "figure.titlesize": TITLE_SIZE, "figure.titleweight": "bold",
        "axes.edgecolor": GRID, "grid.color": GRID, "grid.linewidth": .7,
        "axes.grid": True, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "lines.linewidth": 2, "lines.markersize": 8,
        "legend.fontsize": LEGEND_SIZE, "legend.title_fontsize": LEGEND_SIZE,
        "legend.facecolor": "white", "legend.frameon": True,
        "legend.edgecolor": GRID, "legend.fancybox": True, "legend.framealpha": 1,
        "figure.facecolor": PAPER, "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER, "savefig.transparent": False,
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": .25,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    try:
        # Each inline chart also carries an SVG copy, so VS Code can save it as .svg.
        from IPython import get_ipython
        from matplotlib_inline.config import InlineBackend
        if get_ipython() is not None:
            # Stored in the inline-backend config, so the kernel keeps it when it
            # re-applies its matplotlib setup on the first displayed chart.
            InlineBackend.instance().figure_formats = {"png", "svg"}
    except ImportError:
        pass
