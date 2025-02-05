from cellestial.single.core.dimensional import dimensional
from cellestial.single.core.distribution import boxplot, boxplots, violin, violins
from cellestial.single.core.grids import dimensionals, expressions, pcas, tsnes, umaps
from cellestial.single.core.subdimensional import expression, pca, tsne, umap

# alias
dim = dimensional
dims = dimensionals


__all__ = [
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
]
