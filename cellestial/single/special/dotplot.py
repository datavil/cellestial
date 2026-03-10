from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    layer_tooltips,
    scale_color_gradient,
    scale_fill_gradient,
)
from lets_plot.plot.core import FeatureSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_DOTPLOT

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def dotplot(
    data: AnnData,
    keys: Sequence[str],
    group_by: str,
    *,
    threshold: float = 0,
    variables_name: str = "gene",
    value_name: str = "expression",
    color_low: str = "#e6e6e6",
    color_high: str = "#D2042D",
    fill: bool = False,
    sort_by: str | Sequence[str] | None = None,
    sort_order: Literal["ascending", "descending"] = "descending",
    percentage_key: str = "pct_exp",
    mean_key: str = "avg_exp",
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
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
    variables_name : str, default='gene'
        The name of the variable column in the long format.
    value_name : str, default='expression'
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
        The name of the mean expression column.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
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
    frame = build_frame(
        data=data,
        axis=0,
        variable_keys=keys,
    )
    index_columns = [x for x in frame.columns if x not in keys]

    # CRITICAL PARTS: Dataframe Operations
    # DataFrame to LazyFrame
    frame = frame.lazy()
    # 1. Unpivot frame
    frame = frame.unpivot(
        on=keys,
        index=index_columns,
        variable_name=variables_name,
        value_name=value_name,
    )
    # 2. Aggregate and compute stats
    frame = frame.group_by([group_by, variables_name]).agg(
        [
            pl.col(value_name).mean().alias(mean_key),
            (pl.col(value_name) > threshold).mean().mul(100).alias(percentage_key),
        ]
    )
    # LazyFrame to DataFrame
    frame = frame.collect()
    # HANDLE: Sorting
    # In case of pseudo-categorical integer group_by temporarily cast to int for proper sorting
    with contextlib.suppress(Exception):  # supress errors if sorting fails
        frame = (
            frame.with_columns(pl.col(group_by).cast(pl.String).cast(pl.Int64)).sort(
                group_by, descending=True
            )
            # .with_columns(pl.col(group_by).cast(pl.String).cast(pl.Categorical))
        )
    # perform sorting
    if sort_by is not None:
        frame = frame.sort(
            by=sort_by,
            descending=(sort_order == "descending"),
        )
    # Cast back to categorical
    if frame[group_by].dtype == pl.Int64:
        frame = frame.with_columns(pl.col(group_by).cast(pl.String).cast(pl.Categorical))

    # HANDLE: tooltips
    if tooltips is None:
        tooltips = [variables_name, group_by]
        tooltips_spec = layer_tooltips(tooltips)
    elif tooltips == "none" or isinstance(tooltips, str):
        tooltips_spec = tooltips
    elif isinstance(tooltips, Sequence):
        tooltips = list(tooltips)
        tooltips_spec = layer_tooltips(tooltips)
        if not set(tooltips).issubset(frame.columns):
            missing = set(tooltips) - set(frame.columns)
            msg = f"Some tooltip columns are not in the data: {missing}"
            raise ValueError(msg)
    elif isinstance(tooltips, FeatureSpec):
        tooltips_spec = tooltips

    # BUILD: Dotplot
    if not fill:  # use color aesthetic
        dtplt = (
            ggplot(frame, aes(x=variables_name, y=group_by))
            + geom_point(
                aes(size=percentage_key, color=mean_key), tooltips=tooltips_spec, **geom_kwargs
            )
            + scale_color_gradient(low=color_low, high=color_high)
        )
    else:  # elif fill: use fill aesthetic
        dtplt = (
            ggplot(frame, aes(x=variables_name, y=group_by))
            + geom_point(
                aes(size=percentage_key, fill=mean_key), tooltips=tooltips_spec, **geom_kwargs
            )
            + scale_fill_gradient(low=color_low, high=color_high)
        )

    # ADD: layers
    dtplt += _THEME_DOTPLOT

    # HANDLE: interactive
    if interactive:
        dtplt += ggtb(size_zoomin=-1)

    return dtplt
