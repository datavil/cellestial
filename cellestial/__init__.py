from lets_plot import LetsPlot

from cellestial import datasets
from cellestial._version import __version__, versions
from cellestial.frames import build_frame
from cellestial.layers import arrow_axis, borders, bracket, cluster_outlines, ondata_legend, stream
from cellestial.single import (
    bar,
    boxplot,
    boxplots,
    dimensional,
    dimensionals,
    dotplot,
    elbow,
    expression,
    expressions,
    heatmap,
    highest_expressed_genes,
    markers,
    matrixplot,
    pca,
    pcas,
    plot,
    ridge,
    ridges,
    scatter,
    stacked_violin,
    tsne,
    tsnes,
    umap,
    umaps,
    violin,
    violins,
    volcano,
    volcanos,
    xyplot,
    xyplots,
)
from cellestial.spatial import spatial, spatials
from cellestial.util import get_figure, get_figures, get_mapping, retrieve, save
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
    "borders",
    "boxplot",
    "boxplots",
    "bracket",
    "build_frame",
    "cluster_outlines",
    "datasets",
    "dimensional",
    "dimensionals",
    "dotplot",
    "elbow",
    "expression",
    "expressions",
    "get_figure",
    "get_figures",
    "get_mapping",
    "heatmap",
    "highest_expressed_genes",
    "markers",
    "matrixplot",
    "ondata_legend",
    "pca",
    "pcas",
    "plot",
    "retrieve",
    "ridge",
    "ridges",
    "save",
    "scatter",
    "setup_html",
    "show_colors",
    "spatial",
    "spatials",
    "stacked_violin",
    "stream",
    "tsne",
    "tsnes",
    "umap",
    "umaps",
    "versions",
    "violin",
    "violins",
    "volcano",
    "volcanos",
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
