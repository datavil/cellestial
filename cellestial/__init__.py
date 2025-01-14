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
    tsne,
    tsnes,
    umap,
    umaps,
    violin,
    violins,
)
from cellestial.util import _add_arrow_axis, interactive

# Hand-picked colors for cellestial
TEAL = "#219B9D"
RED = "#D2042D"
CHERRY = "#AF1740"
BLUE = "#377EB8"
LIGHT_GRAY = "#E6E6E6"


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
]
