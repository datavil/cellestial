from lets_plot import LetsPlot

from cellestial._version import __version__, versions
from cellestial.frames import build_frame
from cellestial.layers import arrow_axis, cluster_outlines
from cellestial.single import (
    bar,
    boxplot,
    boxplots,
    dimensional,
    dimensionals,
    dotplot,
    expression,
    expressions,
    heatmap,
    pca,
    pcas,
    plot,
    scatter,
    tsne,
    tsnes,
    umap,
    umaps,
    violin,
    violins,
    xyplot,
    xyplots,
)
from cellestial.util import get_mapping, retrieve
from cellestial.util import get_slice as slice
from cellestial.util.colors import (
    BLUE,
    CHERRY,
    LIGHT_GRAY,
    ORANGE,
    PINK,
    PURPLE,
    RED,
    SNOW,
    TEAL,
    show_colors,
)

setup_html = LetsPlot.setup_html

__all__ = [
    "BLUE",
    "CHERRY",
    "LIGHT_GRAY",
    "ORANGE",
    "PINK",
    "PURPLE",
    "RED",
    "SNOW",
    "TEAL",
    "__version__",
    "arrow_axis",
    "bar",
    "boxplot",
    "boxplots",
    "build_frame",
    "cluster_outlines",
    "dimensional",
    "dimensionals",
    "dotplot",
    "expression",
    "expressions",
    "get_mapping",
    "heatmap",
    "interactive",
    "pca",
    "pcas",
    "plot",
    "retrieve",
    "scatter",
    "setup_html",
    "show_colors",
    "slice",
    "tsne",
    "tsnes",
    "umap",
    "umaps",
    "versions",
    "violin",
    "violins",
    "xyplot",
    "xyplots",
]

try:
    from IPython.core.interactiveshell import InteractiveShell

    if InteractiveShell.initialized():
        from lets_plot import LetsPlot

        LetsPlot.setup_html()
except ImportError:
    pass
