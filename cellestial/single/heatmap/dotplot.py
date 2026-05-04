from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    element_line,
    geom_path,
    geom_point,
    geom_rect,
    ggplot,
    ggtb,
    scale_size,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from lets_plot.plot.core import FeatureSpec

from cellestial.frames import build_frame
from cellestial.single.heatmap._key_groups import _resolve_key_groups
from cellestial.themes import _THEME_DOTPLOT
from cellestial.util import (
    _color_gradient,
    _fill_gradient,
    _get_dendrogram,
    _get_dendrogram_path_frame,
    _resolve_tooltips,
    _validate_tooltips,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lets_plot.plot.core import PlotSpec


# AI-MODIFIED: Gemini 2.5 and Claude 4.6
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line, edge cases unverified
def dotplot(
    data: AnnData,
    keys: Sequence[str] | Mapping[str, Sequence[str]],
    group_by: str,
    *,
    mapping: FeatureSpec | None = None,
    threshold: float = 0,
    size_scale: float = 1.0,
    variable_column: str = "variable",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#D2042D",
    mid_point: Literal["mean", "median", "mid"] | float = "mid",
    sort_by: str | Sequence[str] | None = None,
    sort_order: Literal["ascending", "descending"] = "descending",
    percentage_key: str = "pct_exp",
    mean_key: str = "avg_exp",
    rectangle: bool = True,
    dendrogram: bool = False,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_kwargs: dict | None = None,
    rectangle_size: float = 0.8,
    rectangle_color: str = "#3f3f3f",
    rectangle_kwargs: dict | None = None,
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
    keys : Sequence[str] | Mapping[str, Sequence[str]]
        The variable keys to include in the dotplot. When a mapping is
        provided, each entry maps a group label to the keys belonging to that
        group; the keys are placed on the x-axis in mapping order. The same
        key cannot appear in more than one group.
    group_by : str
        The key to group the data by.
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the plot, the result of `aes()`.
    threshold : float, default=0
        The expression threshold to consider a gene as expressed.
    size_scale : float, default=1.0
        Scaling factor for the point sizes in the plot.
    point_size : float, default=1.0
        Scaling factor for the point sizes in the plot.
    variable_column : str, default='variable'
        Name for the variable column after unpivoting.
    color_low : str, default='#e6e6e6'
        Color for low values in the gradient.
    color_mid : str | None, default=None
        Color for mid values in the gradient.
    color_high : str, default='#D2042D'
        Color for high values in the gradient.
    mid_point : {'mean', 'median', 'mid'} | float, default='mid'
        Midpoint for the color gradient.
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
    rectangle : bool, default=True
        Whether to add a rectangle border around the data area
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
    rectangle_size : float, default=0.8
        Size (thickness) of the rectangle border.
    rectangle_color : str, default='#3f3f3f'
        Color of the rectangle border.
    rectangle_kwargs : dict | None, default=None
        Additional parameters to pass to the rectangle geom_rect.
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
    A simple dotplot.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

        dot = cl.dotplot(
            data,
            keys=markers,
            group_by="cell_type_lvl1",
        )
        dot

    Dotplot allows dendrograms among the groups.

    .. jupyter-execute::
        :emphasize-lines: 5

        dot = cl.dotplot(
            data,
            keys=markers,
            group_by="cell_type_lvl1",
            dendrogram=True,
        )
        dot


    Modify the dendrogram and rectangle borders.

    .. jupyter-execute::
        :emphasize-lines: 6-9

        dot = cl.dotplot(
            data,
            keys=markers,
            group_by="cell_type_lvl1",
            dendrogram=True,
            dendrogram_color="gray",
            dendrogram_size=1,
            rectangle_color="gray",
            rectangle_size=3,
        )
        dot

    """
    # HANDLE: Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    mapping = mapping or aes()

    # RESOLVE: dict ``keys`` into a flat list while preserving mapping order
    keys, _ = _resolve_key_groups(keys)

    # BUILD: dataframe
    frame = build_frame(
        data=data,
        axis=0,
        variable_keys=keys,
    )
    # DROP: rows with null group_by to avoid null labels downstream
    frame = frame.filter(pl.col(group_by).is_not_null())
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
        y_order_groups, paths = _get_dendrogram(data, group_by)
    else:
        y_order_groups = (
            frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
        )

    # ASSIGN: numeric _x / _y positions
    x_keys = list(keys)
    n_x = len(x_keys)
    n_y = len(y_order_groups)
    x_position = {k: i for i, k in enumerate(x_keys)}
    y_position = {g: i for i, g in enumerate(y_order_groups)}
    frame = frame.with_columns(
        pl.col(variable_column)
        .replace_strict(x_position, return_dtype=pl.Float64)
        .alias("position_x"),
        pl.col(group_by)
        .cast(pl.String)
        .replace_strict(y_position, return_dtype=pl.Float64)
        .alias("position_y"),
    )

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=[],
        defaults=[group_by, variable_column, mean_key, percentage_key],
    )
    _validate_tooltips(tooltips, frame)

    # BUILD: Dotplot
    use_fill = "fill" in mapping.as_dict()
    color_or_fill = "fill" if use_fill else "color"
    _gradient = _fill_gradient if use_fill else _color_gradient

    # DEFINE: mapping with defaults
    _mapping = {
        "x": "position_x",
        "y": "position_y",
        "size": percentage_key,
        color_or_fill: mean_key,
    }
    _mapping.update(mapping.as_dict())

    # Adjust point size range (keeps aes(size=...) mapping and its legend):
    n_max = max(n_x, n_y)
    size_max = 120 / n_max * size_scale
    _size_scale = scale_size(range=[size_max * 0.1, size_max])

    dtplt = (
        ggplot(frame)
        + geom_point(
            aes(**_mapping),
            tooltips=tooltips,
            **geom_kwargs,
        )
        + _gradient(
            frame[mean_key],
            color_low=color_low,
            color_mid=color_mid,
            color_high=color_high,
            mid_point=mid_point,
        )
        + _size_scale
    )

    # DENDROGRAM (right side, along y-axis), built first so we can derive tight x limits
    x_max_limit = n_x - 0.5
    if dendrogram:
        group_centers = [float(i) for i in range(n_y)]
        dendrogram_frame = _get_dendrogram_path_frame(
            paths, n_x=n_x, n_groups=n_y, group_centers=group_centers
        )
        x_max_limit = dendrogram_frame["x"].max()
        dtplt += geom_path(
            data=dendrogram_frame,
            mapping=aes(x="x", y="y", group="group"),
            color=dendrogram_color,
            size=dendrogram_size,
            **(dendrogram_kwargs or {}),
        )

    # AXES: discrete labels via continuous breaks; tight limits keep ticks against the rectangle
    y_max_limit = n_y - 0.5
    dtplt += scale_x_continuous(
        breaks=list(range(n_x)),
        labels=x_keys,
        limits=[-0.5, x_max_limit],
        expand=[0, 0],
    )
    dtplt += scale_y_continuous(
        breaks=list(range(n_y)),
        labels=y_order_groups,
        limits=[-0.5, y_max_limit],
        expand=[0, 0],
    )

    # BORDER: rectangle around data area only (keeps dendrogram outside the frame)
    if rectangle:
        dtplt += geom_rect(
            data={
                "xmin": [-0.5],
                "xmax": [n_x - 0.5],
                "ymin": [-0.5],
                "ymax": [n_y - 0.5],
            },
            mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
            color=rectangle_color,
            size=rectangle_size,
            fill="rgba(0,0,0,0)",
            inherit_aes=False,
            **(rectangle_kwargs or {}),
        )
    else:
        dtplt += theme(axis_line=element_line(color="#1f1f1f"))

    # ADD: layers
    dtplt += _THEME_DOTPLOT

    # HANDLE: interactive
    if interactive:
        dtplt += ggtb(size_zoomin=-1)

    return dtplt
