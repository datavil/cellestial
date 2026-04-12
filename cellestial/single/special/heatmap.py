from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from warnings import warn

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    element_blank,
    geom_raster,
    geom_segment,
    geom_tile,
    ggplot,
    ggtb,
    guides,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)

from cellestial.frames import build_frame
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import _fill_gradient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec, PlotSpec

_DENDROGRAM_RATIO = 0.15
_GROUP_BAR_RATIO = 0.02
_GROUP_BAR_GAP = 0.5


def _get_dendrogram(data: AnnData, group_by: str) -> tuple[list[str], pl.DataFrame]:
    """
    Get or compute the dendrogram for a group_by key and extract segments.

    Checks ``data.uns[f'dendrogram_{group_by}']``.
    For AnnData, if not present, runs ``scanpy.tl.dendrogram`` on  to compute it.
    Extracts the icoord/dcoord arrays into a Polars DataFrame of segments with normalized positions.
    """
    import numpy as np

    key = f"dendrogram_{group_by}"
    if key not in data.uns:
        import scanpy as sc

        sc.tl.dendrogram(data, groupby=group_by)

    dendro = data.uns[key]
    categories_ordered = list(dendro["categories_ordered"])
    dendro_info = dendro["dendrogram_info"]

    icoord = np.array(dendro_info["icoord"])
    dcoord = np.array(dendro_info["dcoord"])

    # scipy places leaves at 5, 15, 25, ... → normalize to 0, 1, 2, ...
    icoord = (icoord - 5) / 10

    max_height = dcoord.max()
    if max_height > 0:
        dcoord = dcoord / max_height

    x, xend, y, yend = [], [], [], []
    for i in range(len(icoord)):
        for j in range(3):
            x.append(float(icoord[i][j]))
            xend.append(float(icoord[i][j + 1]))
            y.append(float(dcoord[i][j]))
            yend.append(float(dcoord[i][j + 1]))

    segments = pl.DataFrame({"x": x, "xend": xend, "y": y, "yend": yend})
    return categories_ordered, segments


def _scale_values(frame: pl.DataFrame, *, value_column: str, partition_key: str) -> pl.DataFrame:
    """Min-max scale ``value_column`` within partitions defined by ``partition_key``."""
    v = pl.col(value_column)
    vmin = v.min().over(partition_key)
    vmax = v.max().over(partition_key)
    return frame.with_columns(((v - vmin) / (vmax - vmin)).alias(value_column))


def _assign_positions(
    frame: pl.DataFrame,
    *,
    aggregate: bool,
    group_by: str,
    observations_name: str,
    variable_column: str,
    x_keys: list[str],
    y_order_groups: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame | None, int, int, list[float]]:
    """
    Attach ``_x``/``_y`` to ``frame`` and compute layout metadata.

    Returns
    -------
    frame : pl.DataFrame
        Frame with ``_x`` and ``_y`` columns.
    cell_frame : pl.DataFrame | None
        Per-cell layout frame (None when aggregating).
    n_x : int
    n_y : int
    group_centers : list[float]
        Y center of each group, in ``y_order_groups`` order.
    """
    n_x = len(x_keys)
    x_pos = {k: i for i, k in enumerate(x_keys)}
    frame = frame.with_columns(
        pl.col(variable_column).replace_strict(x_pos, return_dtype=pl.Float64).alias("_x")
    )

    if aggregate:
        n_y = len(y_order_groups)
        y_pos = {g: i for i, g in enumerate(y_order_groups)}
        frame = frame.with_columns(
            pl.col(group_by)
            .cast(pl.String)
            .replace_strict(y_pos, return_dtype=pl.Float64)
            .alias("_y"),
        )
        return frame, None, n_x, n_y, [float(i) for i in range(n_y)]

    # non-aggregate: rescale per-cell _y to span [0, n_x] for square-ish aspect
    grp_idx = {g: i for i, g in enumerate(y_order_groups)}
    cell_frame = (
        frame.select(observations_name, group_by)
        .unique()
        .with_columns(
            pl.col(group_by)
            .cast(pl.String)
            .replace_strict(grp_idx, return_dtype=pl.Int64)
            .alias("_grp_idx")
        )
        .sort(["_grp_idx", observations_name])
    )
    n_y = cell_frame.height
    y_step = (n_x - 1) / max(n_y - 1, 1)
    cell_frame = cell_frame.with_columns(
        (pl.int_range(pl.len()).cast(pl.Float64) * y_step).alias("_y")
    )
    frame = frame.join(cell_frame.select(observations_name, "_y"), on=observations_name)
    centers_frame = (
        cell_frame.group_by(group_by, maintain_order=False)
        .agg(pl.col("_y").mean().alias("_center"), pl.col("_grp_idx").first())
        .sort("_grp_idx")
    )
    return frame, cell_frame, n_x, n_y, centers_frame["_center"].to_list()


def _get_group_bar_frame(
    cell_frame: pl.DataFrame, *, group_by: str, n_x: int
) -> tuple[pl.DataFrame, float]:
    """Build the per-group colored bar frame drawn to the left of the heatmap."""
    bar_width = max(1.0, n_x * _GROUP_BAR_RATIO)
    bar_xend = -_GROUP_BAR_GAP
    bar_x = bar_xend - bar_width
    bar_x_mid = (bar_x + bar_xend) / 2
    bar_frame = (
        cell_frame.group_by(group_by, maintain_order=False)
        .agg(
            pl.col("_y").min().alias("y_min"),
            pl.col("_y").max().alias("y_max"),
            pl.col("_grp_idx").first(),
        )
        .sort("_grp_idx")
        .with_columns(pl.lit(bar_x_mid).alias("x"))
    )
    return bar_frame, bar_x_mid


def _get_group_lines_frame(
    cell_frame: pl.DataFrame | None,
    *,
    aggregate: bool,
    group_by: str,
    n_x: int,
    n_y: int,
    n_groups: int,
) -> pl.DataFrame:
    """Build the horizontal separator-line frame between groups."""
    if aggregate:
        line_ys = [i + 0.5 for i in range(n_groups - 1)]
    else:
        boundaries = (
            cell_frame.group_by(group_by, maintain_order=False)
            .agg(pl.col("_y").max().alias("y_max"), pl.col("_grp_idx").first())
            .sort("_grp_idx")
            .head(n_groups - 1)["y_max"]
            .to_list()
        )
        half_step = (n_x - 1) / max(n_y - 1, 1) / 2
        line_ys = [y + half_step for y in boundaries]

    x_start = -0.5
    x_end = n_x - 0.5
    return pl.DataFrame(
        {
            "x": [x_start] * len(line_ys),
            "xend": [x_end] * len(line_ys),
            "y": line_ys,
            "yend": line_ys,
        }
    )


def _get_dendrogram_segment_frame(
    segments: pl.DataFrame, *, n_x: int, n_groups: int, group_centers: list[float]
) -> pl.DataFrame:
    """Map normalized dendrogram segments into heatmap coordinates (right side)."""
    import numpy as np

    y_dendrogram_size = n_x * _DENDROGRAM_RATIO
    leaf_xp = np.arange(n_groups, dtype=float)
    seg_y = np.interp(segments["x"].to_numpy(), leaf_xp, group_centers)
    seg_yend = np.interp(segments["xend"].to_numpy(), leaf_xp, group_centers)
    return pl.DataFrame(
        {
            "x": (n_x - 0.5) + segments["y"].to_numpy() * y_dendrogram_size,
            "xend": (n_x - 0.5) + segments["yend"].to_numpy() * y_dendrogram_size,
            "y": seg_y,
            "yend": seg_yend,
        }
    )


def heatmap(
    data: AnnData,
    group_by: str,
    keys: Sequence[str] | None = None,
    *,
    mapping: FeatureSpec | None = None,
    geom: Literal["raster", "tile"] = "raster",
    scale_axis: Literal[0, 1] | None = None,
    dendrogram: bool = False,
    aggregate: bool = True,
    group_bars: bool = True,
    group_bars_size: float = 6,
    group_bars_labels: bool = False,
    group_lines: bool = True,
    group_lines_color: str = "black",
    group_lines_size: float = 1.0,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    aggregate_kwargs: dict | None = None,
    group_lines_kwargs: dict | None = None,
    dendrogram_kwargs: dict | None = None,
    value_column: str = "value",
    variable_column: str = "variable",
    color_low: str = "#0000ff",
    color_mid: str = "#ffffff",
    color_high: str = "#ff0000",
    mid_point: Literal["mean", "median", "mid"] | float = "mid",
    axis: Literal[0, 1] | None = 0,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool | int = False,
    interactive: bool = False,
    **geom_kwargs,
) -> PlotSpec:
    """
    Heatmap.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    group_by : str
        The key to group the data by.
    keys : Sequence[str] | None, default=None
        Variable keys to include. If None, no additional keys are added.
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the plot, the result of `aes()`.
    geom : {'raster', 'tile'}, default='raster'
        The geom to use,. Use 'raster' for performance.
        Use 'tile' to enable tooltips.
    dendrogram : bool, default=False
        Whether to add a dendrogram for the ``group_by`` axis.
        Uses ``scanpy.tl.dendrogram`` if not already computed.
    aggregate : bool, default=True
        If True, aggregate values per group by mean so each row is a group.
        If False, plot one row per observation (i.e., cell).
    group_bars : bool, default=True
        Whether to draw colored vertical bars on the left marking group membership.
        Only used when ``aggregate=False``.
    group_bars_size : float, default=6
        Size (thickness) of the group color bars.
    group_bars_labels : bool, default=False
        Whether to show group names as labels along the y-axis.
        Removes the related legend.
        Only applies when ``group_bars=True`` and ``aggregate=False``.
    group_lines : bool, default=True
        Whether to draw horizontal lines within the heatmap separating groups.
    group_lines_color : str, default='black'
        Color of the group separator lines.
    group_lines_size : float, default=1.0
        Size (thickness) of the group separator lines.
    dendrogram_color : str, default='black'
        Color of the dendrogram segments.
    dendrogram_size : float, default=0.5
        Size (thickness) of the dendrogram segments.
    aggregate_kwargs : dict | None, default=None
        Additional parameters to pass to the main heatmap geom layer.
    group_lines_kwargs : dict | None, default=None
        Additional parameters to pass to the group separator lines geom_segment.
    dendrogram_kwargs : dict | None, default=None
        Additional parameters to pass to the dendrogram geom_segment.
    scale_axis : {0, 1} | None, default=None
        Whether to standardize a dimension between 0 and 1.
        Subtracts the minimum and divides by the maximum.
        If 0, standardize each variable (column).
        If 1, standardize each row (group when ``aggregate`` is True, observation otherwise).
    value_column : str, default='value'
        Name for the value column after unpivoting.
    variable_column : str, default='variable'
        Name for the variable column after unpivoting.
    color_low : str, default='#0000ff'
        Color for low values in the gradient.
    color_mid : str, default='#ffffff'
        Color for mid values in the gradient.
    color_high : str, default='#ff0000'
        Color for high values in the gradient.
    mid_point : {'mean', 'median', 'mid'} | float, default='mid'
        Midpoint for the color gradient.
    axis : {0,1} | None, default=0
        Axis of the data, 0 for observations and 1 for variables.
    observations_name : str, default='Barcode'
        The name of the observations column.
    variables_name : str, default='Variable'
        Name for the variables index column.
    include_dimensions : bool | int, default=False
        Whether to include dimensions in the DataFrame.
        Providing an integer will limit the number of dimensions to given number.
    interactive : bool, default=False
        Whether to make the plot interactive.
    **geom_kwargs
        Additional parameters for the heatmap geom layer.

    Returns
    -------
    PlotSpec
        Heatmap.

    Examples
    --------
    Heatmap aggregates per group_by, by default.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read("data/pbmc3k_pped.h5ad")

        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            dendrogram=True,
        )

    To enable tooltips, use ``geom='tile'``.

    .. jupyter-execute::

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            dendrogram=True,
            geom="tile",
            tooltips=["value"],
        )

    Values can be standardized per-row or per-column with ``scale_axis``.

    .. jupyter-execute::

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            dendrogram=True,
            scale_axis=1,
        )

    If ``aggregate=False``, each cell is plotted as a separate row.

    .. jupyter-execute::
        :emphasize-lines: 6

        htmp=cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            geom="raster",
            aggregate=False,
        ) + scale_fill_viridis()
        htmp


    Heatmap components (group separator lines, group bars, dendrogram) can be added or customized.

    .. jupyter-execute::
        :emphasize-lines: 7-13

        htmp=cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            geom="raster",
            aggregate=False,
            group_lines_size=0.5,
            group_lines_color="white",
            dendrogram=True,
            dendrogram_size="0.7",
            group_bars_labels=True,
            group_bars=True,
        ) + scale_fill_viridis()
        htmp

    """
    mapping = mapping or aes()
    aggregate_kwargs = dict(aggregate_kwargs) if aggregate_kwargs else {}
    group_lines_kwargs = dict(group_lines_kwargs) if group_lines_kwargs else {}
    dendrogram_kwargs = dict(dendrogram_kwargs) if dendrogram_kwargs else {}

    if "tooltips" in geom_kwargs and geom == "raster":
        warn(
            "\nWarning: tooltips are not supported for 'raster' geom and will be ignored."
            "\nUse 'tile' geom to enable tooltips.",
            stacklevel=1,
        )
        geom_kwargs.pop("tooltips")

    # BUILD: long-form dataframe
    frame = build_frame(
        data=data,
        variable_keys=keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
    )
    index_columns = [group_by] if aggregate else [observations_name, group_by]
    frame = frame.unpivot(
        on=keys,
        index=index_columns,
        variable_name=variable_column,
        value_name=value_column,
    )
    if aggregate:
        frame = frame.group_by(group_by, variable_column).agg(pl.col(value_column).mean())
    frame = frame.drop_nulls()

    # HANDLE: standard scaling
    if scale_axis is not None:
        partition_key = (
            variable_column if scale_axis == 0 else (group_by if aggregate else observations_name)
        )
        frame = _scale_values(frame, value_column=value_column, partition_key=partition_key)

    # DETERMINE: y order of groups
    if dendrogram:
        y_order_groups, segments = _get_dendrogram(data, group_by)
    else:
        y_order_groups = (
            frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
        )
        segments = None

    # ASSIGN: _x / _y positions and layout metadata
    x_keys = list(keys)
    frame, cell_frame, n_x, n_y, group_centers = _assign_positions(
        frame,
        aggregate=aggregate,
        group_by=group_by,
        observations_name=observations_name,
        variable_column=variable_column,
        x_keys=x_keys,
        y_order_groups=y_order_groups,
    )

    # BUILD: heatmap layer
    aes_main = aes(x="_x", y="_y", fill=value_column, **mapping.as_dict())
    geom_kwargs.pop("tooltips", None)
    geom_layer = (
        geom_raster(aes_main, **geom_kwargs)
        if geom == "raster"
        else geom_tile(aes_main, **geom_kwargs)
    )
    htmp = ggplot(frame) + geom_layer + _THEME_HEATMAP

    # X scale: variable labels
    htmp += scale_x_continuous(breaks=list(range(n_x)), labels=x_keys)

    # Y scale: groups for aggregate, group-center labels when titling bars, else hidden
    if aggregate:
        htmp += scale_y_continuous(breaks=list(range(n_y)), labels=y_order_groups)
    elif group_bars and group_bars_labels:
        htmp += scale_y_continuous(breaks=group_centers, labels=y_order_groups)
        htmp += guides(color="none")
    else:
        htmp += theme(axis_text_y=element_blank(), axis_ticks_y=element_blank())

    # GROUP color bar on left for non-aggregate
    if not aggregate and group_bars:
        group_bar_frame, _ = _get_group_bar_frame(cell_frame, group_by=group_by, n_x=n_x)
        htmp += geom_segment(
            data=group_bar_frame,
            mapping=aes(x="x", xend="x", y="y_min", yend="y_max", color=group_by),
            size=group_bars_size,
            **aggregate_kwargs,
        )

    # GROUP separator lines (horizontal, within heatmap)
    if group_lines and len(y_order_groups) > 1:
        lines_frame = _get_group_lines_frame(
            cell_frame,
            aggregate=aggregate,
            group_by=group_by,
            n_x=n_x,
            n_y=n_y,
            n_groups=len(y_order_groups),
        )
        htmp += geom_segment(
            data=lines_frame,
            mapping=aes(x="x", xend="xend", y="y", yend="yend"),
            color=group_lines_color,
            size=group_lines_size,
            **group_lines_kwargs,
        )

    # DENDROGRAM (right side, along y-axis)
    if dendrogram:
        dendro_frame = _get_dendrogram_segment_frame(
            segments, n_x=n_x, n_groups=len(y_order_groups), group_centers=group_centers
        )
        htmp += geom_segment(
            data=dendro_frame,
            mapping=aes(x="x", y="y", xend="xend", yend="yend"),
            color=dendrogram_color,
            size=dendrogram_size,
            **dendrogram_kwargs,
        )

    # FILL gradient
    htmp += _fill_gradient(
        frame[value_column],
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,
    )

    if interactive:
        htmp += ggtb(size_zoomin=-1)

    return htmp


imshow = heatmap
