from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    coord_flip,
    element_blank,
    element_text,
    geom_boxplot,
    ggplot,
    scale_fill_hue,
    theme,
    ylab,
)
from polars import DataFrame
from scipy.sparse import issparse

from cellestial.frames.build import anndata_variable_columns

if TYPE_CHECKING:

    from lets_plot.plot.core import FeatureSpec, PlotSpec


_THEME_HIGHEST = (
    scale_fill_hue()
    + theme(
        text=element_text(family="Arial", color="#1f1f1f"),
        title=element_text(family="Arial", color="#1f1f1f"),
        legend_title=element_text(family="Arial", color="#1f1f1f", face="Bold"),
        axis_title_x=element_blank(),
        legend_position="none",
    )
    + ylab("Percentage of Total Counts")
    + coord_flip()
)


def _highest_expressed_genes_frame(
    data: AnnData,
    n: int = 10,
) -> DataFrame:
    """Get the top n highest expressed genes by mean percentage across all cells."""
    if isinstance(data, AnnData):
        X = data.X
        # normalize each cell to sum to 100 (percentage)
        if issparse(X):
            row_sums = np.array(X.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1
            X_normalized = X.multiply(100 / row_sums[:, np.newaxis])
            mean_percent = np.array(X_normalized.mean(axis=0)).ravel()
        else:
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_normalized = X / row_sums * 100
            mean_percent = X_normalized.mean(axis=0)

        top_idx = np.argsort(mean_percent)[::-1][:n]
        genes = data.var_names[top_idx].tolist()
        if issparse(X_normalized):
            X_normalized = X_normalized.tocsr()
        norm_data = AnnData(X=X_normalized, var=data.var, obs=data.obs)
        frame = pl.DataFrame(anndata_variable_columns(norm_data, keys=genes, column_names=[]))
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise TypeError(msg)

    return frame


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
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # HANDLE: mapping
    default_mapping = aes(variable_column, value_column, fill=variable_column)
    if mapping is not None:
        _mapping = default_mapping.as_dict()
        _mapping.update(mapping.as_dict())
        mapping = aes(**_mapping)
    else:
        mapping = default_mapping

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
