from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    scale_color_gradient,
    scale_fill_gradient,
)
from lets_plot.plot.core import PlotSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_DOTPLOT

if TYPE_CHECKING:
    from collections.abc import Sequence


def dotplot(
    data: AnnData,
    keys: Sequence[str],
    group_by: str,
    *,
    threshold: float = 0,
    variable_name: str = "gene",
    value_name: str = "expression",
    color_low: str = "#e6e6e6",
    color_high: str = "#D2042D",
    fill: bool = False,
    sort_by: str | Sequence[str] | None = None,
    sort_order: str = "descending",
    percentage_key: str = "pct_exp",
    mean_key: str = "avg_exp",
    show_tooltips: bool = True,
    interactive: bool = False,
    **geom_kwargs,
) -> PlotSpec:
    """
    Dotplot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : Sequence[str]
        The variable keys or names to include in the dotplot.
    group_by : str
        The key to group the data by.
    threshold : float, default=0
        The expression threshold to consider a gene as expressed.
    variable_name : str, default='gene'
        The name of the variable column in the long format.
    value_name : str, default="expression"
        The name of the value column in the long format.
    color_low : str, default='#e6e6e6'
        The low color for the gradient.
    color_high : str, default='#D2042D'
        The high color for the gradient.
    fill : bool, optional
        Whether to use fill aesthetic instead of color, by default False.
    sort_by : str | None
        The column to sort the results by, by default None.
    sort_order : str, default='descending'
        The sort order, either 'ascending' or 'descending'.
    percentage_key : str, default='pct_exp'
        The name of the percentage column.
    mean_key : str, default='avg_exp'
        The name of the mean expression column
    show_tooltips : bool, default=True
        Whether to show tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    **geom_kwargs : Any
        Additional keyword arguments for the geom_point layer.

    Returns
    -------
    PlotSpec
        Dotplot.
    """
    # HANDLE: Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)
    # BUILD: dataframe
    frame = build_frame(data=data, axis=0, variable_keys=keys)
    index_columns = [x for x in frame.columns if x not in keys]

    #  CRITICAL PARTS: Dataframe Operations
    # 1. Unpivot frame
    long_frame = frame.unpivot(
        on=keys,
        index=index_columns,
        variable_name=variable_name,
        value_name=value_name,
    )
    # 2. Aggregate and compute stats
    stats_frame = long_frame.group_by([group_by, variable_name]).agg(
        [
            pl.col(value_name).mean().alias(mean_key),
            (pl.col(value_name) > threshold).mean().mul(100).alias(percentage_key),
        ]
    )

    # HANDLE: Sorting
    # In case of pseudo-categorical integer group_by
    with contextlib.suppress(Exception):  # supress errors if sorting fails
        stats_frame = (
            stats_frame.with_columns(pl.col(group_by).cast(pl.String).cast(pl.Int64))
            .sort(group_by, descending=True)
            .with_columns(pl.col(group_by).cast(pl.String).cast(pl.Categorical))
        )
        # remove group_by from sort_by if present
        if isinstance(sort_by, str):
            sort_by = [sort_by]
        if sort_by is not None:
            sort_by = [s for s in sort_by if s != group_by]
    # perform sorting
    if sort_by is not None:
        stats_frame = stats_frame.sort(
            by=sort_by,
            descending=(sort_order == "descending"),
        )

    # BUILD: Dotplot
    if not fill:  # use color aesthetic
        dtplt = (
            ggplot(stats_frame, aes(x=variable_name, y=group_by))
            + geom_point(aes(size=percentage_key, color=mean_key), **geom_kwargs)
            + scale_color_gradient(low=color_low, high=color_high)
        )
    else:  # elif fill: use fill aesthetic
        dtplt = (
            ggplot(stats_frame, aes(x=variable_name, y=group_by))
            + geom_point(aes(size=percentage_key, fill=mean_key), **geom_kwargs)
            + scale_fill_gradient(low=color_low, high=color_high)
        )

    # ADD: layers
    dtplt += _THEME_DOTPLOT

    # HANDLE: interactive
    if interactive:
        dtplt += ggtb(size_zoomin=-1)

    return dtplt
