from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_point,
    geom_segment,
    ggplot,
    ggtb,
    layer_tooltips,
    scale_color_gradient,
    scale_fill_gradient,
    scale_x_continuous,
    scale_y_continuous,
)
from lets_plot.plot.core import FeatureSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_DOTPLOT
from cellestial.util import _get_dendrogram, _get_dendrogram_segment_frame

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def dotplot(
    data: AnnData,
    keys: Sequence[str],
    group_by: str,
    *,
    mapping: FeatureSpec | None = None,
    threshold: float = 0,
    variable_column: str = "variable",
    color_low: str = "#e6e6e6",
    color_high: str = "#D2042D",
    sort_by: str | Sequence[str] | None = None,
    sort_order: Literal["ascending", "descending"] = "descending",
    percentage_key: str = "pct_exp",
    mean_key: str = "avg_exp",
    dendrogram: bool = False,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_kwargs: dict | None = None,
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
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the plot, the result of `aes()`.
    threshold : float, default=0
        The expression threshold to consider a gene as expressed.
    variable_column : str, default='variable'
        Name for the variable column after unpivoting.
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
    dendrogram : bool, default=False
        Whether to add a dendrogram for the ``group_by`` axis.
        Uses ``scanpy.tl.dendrogram`` if not already computed.
        When True, group order is determined by the dendrogram.
    dendrogram_color : str, default='black'
        Color of the dendrogram segments.
    dendrogram_size : float, default=0.5
        Size (thickness) of the dendrogram segments.
    dendrogram_kwargs : dict | None, default=None
        Additional parameters to pass to the dendrogram geom_segment.
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

    Examples
    --------
    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        markers = [
            "ITGB1",
            "SYNGR1",
            "HBA1",
            "TCF7L2",
            "HBM",
            "IGHD",
            "FCN1",
            "GYPA",
            "FCGR3A",
            "PAX5",
            "COL4A4",
            "HBB",
            "LYN",
            "PRDM1",
            "CD14",
            "IGHM",
            "CDK6",
            "MS4A1",
            "BLK",
            "IRF4",
            "BCL11A",
            "MKI67",
        ]

        (
            cl.dotplot(
                data,
                keys=markers,
                group_by="cell_type_lvl1",
                color_high="red",
                sort_by="avg_exp",
            )
            + scale_y_discrete(expand=[0.1, 0.1])
            + theme(axis_text_y=element_text(angle=90, size=12))
        )
    """
    # HANDLE: Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    mapping = mapping or aes()
    dendrogram_kwargs = dict(dendrogram_kwargs) if dendrogram_kwargs else {}

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
    value_name: str = "value"
    frame = frame.unpivot(
        on=keys,
        index=index_columns,
        variable_name=variable_column,
        value_name=value_name,
    )
    # 2. Aggregate and compute stats
    frame = frame.group_by([group_by, variable_column]).agg(
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



    # DETERMINE: y order of groups
    if dendrogram:
        y_order_groups, segments = _get_dendrogram(data, group_by)
    else:
        y_order_groups = (
            frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
        )
        segments = None

    # ASSIGN: numeric _x / _y positions
    x_keys = list(keys)
    n_x = len(x_keys)
    n_y = len(y_order_groups)
    x_pos = {k: i for i, k in enumerate(x_keys)}
    y_pos = {g: i for i, g in enumerate(y_order_groups)}
    frame = frame.with_columns(
        pl.col(variable_column).replace_strict(x_pos, return_dtype=pl.Float64).alias("_x"),
        pl.col(group_by)
        .cast(pl.String)
        .replace_strict(y_pos, return_dtype=pl.Float64)
        .alias("_y"),
    )

    # HANDLE: tooltips
    if tooltips is None:
        tooltips = [group_by, variable_column]
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
    use_fill = "fill" in mapping.as_dict()
    color_or_fill = "fill" if use_fill else "color"
    scale = scale_fill_gradient if use_fill else scale_color_gradient

    dtplt = (
        ggplot(frame)
        + geom_point(
            aes(x="_x", y="_y",size=percentage_key, **{color_or_fill: mean_key}, **mapping.as_dict()),
            tooltips=tooltips_spec,
            **geom_kwargs,
        )
        + scale(low=color_low, high=color_high)
    )

    # AXES: discrete labels via continuous breaks
    dtplt += scale_x_continuous(breaks=list(range(n_x)), labels=x_keys)
    dtplt += scale_y_continuous(breaks=list(range(n_y)), labels=y_order_groups)

    # DENDROGRAM (right side, along y-axis)
    if dendrogram:
        group_centers = [float(i) for i in range(n_y)]
        dendro_frame = _get_dendrogram_segment_frame(
            segments, n_x=n_x, n_groups=n_y, group_centers=group_centers
        )
        dtplt += geom_segment(
            data=dendro_frame,
            mapping=aes(x="x", y="y", xend="xend", yend="yend"),
            color=dendrogram_color,
            size=dendrogram_size,
            **dendrogram_kwargs,
        )

    # ADD: layers
    dtplt += _THEME_DOTPLOT

    # HANDLE: interactive
    if interactive:
        dtplt += ggtb(size_zoomin=-1)

    return dtplt
