from cellestial._version import __version__, versions
from cellestial.single import (
    boxplot,
    boxplots,
    dim,
    dimensional,
    dimensionals,
    dotplot,
    expression,
    expressions,
    pca,
    pcas,
    scatter,
    tsne,
    tsnes,
    umap,
    umaps,
    violin,
    violins,
)
from cellestial.util import _add_arrow_axis, retrieve, slice
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

__all__ = [
    "interactive",
    "_add_arrow_axis",
    "dimensional",
    "dimensionals",
    "dim",
    "umap",
    "umaps",
    "pca",
    "pcas",
    "tsne",
    "tsnes",
    "expression",
    "expressions",
    "violin",
    "violins",
    "boxplot",
    "boxplots",
    "dotplot",
    "versions",
    "TEAL",
    "RED",
    "CHERRY",
    "BLUE",
    "LIGHT_GRAY",
    "SNOW",
    "PURPLE",
    "PINK",
    "ORANGE",
    "show_colors",
    "scatter",
    "retrieve",
    "slice",
]
