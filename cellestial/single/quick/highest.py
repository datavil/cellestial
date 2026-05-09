from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_boxplot,
    ggplot,
)

from cellestial.frames import _highest_expressed_genes_frame
from cellestial.themes import _THEME_HIGHEST
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpec, PlotSpec


def highest_expressed_genes(
    data: AnnData,
    n: int = 20,
    *,
    mapping: FeatureSpec | None = None,
    threshold: float | None = None,
    observations_name: str = "Barcode",
    value_column: str = "value",
    variable_column: str = "variable",
    color: str = "#1f1f1f",
    size=0.5,
    outlier_size: float = 0.2,
    outlier_alpha: float = 0.5,
    fatten: float = 1,
    **geom_kwargs,
) -> PlotSpec:
    """
    Highest Expressed Genes Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    n : int, default=20
        Number of top expressed genes to display, ranked by mean percentage across all cells.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    value_column : str, default='value'
        The name to give to the value column after unpivoting.
    variable_column : str, default='variable'
        The name to give to the variable column after unpivoting.
    color : str, default='#1f1f1f'
        Border color of the boxplots.
    size : float, default=0.5
        Line size of the boxplots.
    outlier_size : float, default=0.2
        Size of the outlier points.
    outlier_alpha : float, default=0.5
        Transparency of the outlier points.
    fatten : float, default=1
        Factor to fatten the median line.
    **geom_kwargs
        Additional parameters for the `geom_boxplot` layer.
        For more information on geom_boxplot parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_boxplot.html

    Returns
    -------
    PlotSpec
        Highest expressed genes boxplot.

    Examples
    --------
    A simple Highest expressed genes boxplot.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        cl.highest_expressed_genes(data,n=20)

    Customize the plot.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        cl.highest_expressed_genes(
            data,
            n=10,
            outlier_size=0.1,
            outlier_alpha=0.2,
            outlier_shape=5)+scale_fill_viridis()


    """
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    # HANDLE: mapping
    defaults = aes(x=variable_column, y=value_column, fill=variable_column).as_dict()
    if mapping is not None:
        defaults.update(mapping.as_dict())
    mapping = aes(**defaults)

    # get the top n highest expressed genes by mean percentage across all cells
    frame = _highest_expressed_genes_frame(data, n=n)
    genes = frame.columns

    frame = frame.unpivot(value_name=value_column, variable_name=variable_column)
    # keep the order of genes
    frame = frame.sort(
        pl.col(variable_column).cast(pl.Enum(genes)),
        descending=True,
    )

    # FILTER: drop nulls and apply threshold if provided
    frame = frame.drop_nulls()
    frame = frame.filter(
        pl.col(value_column) >= threshold if threshold is not None else True,
    )

    # BUILD: boxplot
    bxplt = ggplot(frame)
    bxplt += geom_boxplot(
        mapping=mapping,
        color=color,
        size=size,
        outlier_size=outlier_size,
        outlier_alpha=outlier_alpha,
        fatten=fatten,
        **geom_kwargs,
    )

    return bxplt + _THEME_HIGHEST
