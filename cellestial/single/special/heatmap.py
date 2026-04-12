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
    Get or compute the dendrogram for a groupby key and extract segments.

    Checks ``data.uns[f'dendrogram_{group_by}']``. If not present,
    runs ``scanpy.tl.dendrogram`` to compute it. Extracts the icoord/dcoord
    arrays into a Polars DataFrame of segments with normalized positions.

    Parameters
    ----------
    data : AnnData
        The AnnData object.
    group_by : str
        The groupby key.

    Returns
    -------
    categories_ordered : list[str]
        Group names in dendrogram leaf order.
    segments : pl.DataFrame
        Segment coordinates with columns ``x``, ``xend``, ``y``, ``yend``.
    """
    import numpy as np

    if isinstance(data, AnnData):
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

        # Normalize heights to [0, 1]
        max_height = dcoord.max()
        if max_height > 0:
            dcoord = dcoord / max_height

        x = []
        xend = []
        y = []
        yend = []
        for i in range(len(icoord)):
            for j in range(3):
                x.append(float(icoord[i][j]))
                xend.append(float(icoord[i][j + 1]))
                y.append(float(dcoord[i][j]))
                yend.append(float(dcoord[i][j + 1]))

        segments = pl.DataFrame({"x": x, "xend": xend, "y": y, "yend": yend})

    return categories_ordered, segments


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
    group_lines: bool = True,
    group_lines_color: str = "black",
    group_lines_size: float = 1.0,
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
        Which lets_plot geom to use. ``raster`` is much faster and is
        forced for non-aggregated heatmaps because tiles are
        prohibitively expensive at cell resolution. ``raster`` does not
        support tooltips.
    dendrogram : bool, default=False
        Whether to add a dendrogram for the ``group_by`` axis.
        Uses ``scanpy.tl.dendrogram`` if not already computed.
    aggregate : bool, default=True
        If True, aggregate values per group (mean) so each row is a group.
        If False, plot one row per observation (cell). Cells are sorted by
        group in dendrogram order (or input order). Y ticks/labels are
        hidden and a colored vertical bar on the left marks group
        membership.
    group_bars : bool, default=True
        Whether to draw colored vertical bars on the left marking group
        membership. Only used when ``aggregate=False``.
    group_lines : bool, default=True
        Whether to draw horizontal lines within the heatmap separating
        groups.
    group_lines_color : str, default='black'
        Color of the group separator lines.
    group_lines_size : float, default=1.0
        Size (thickness) of the group separator lines.
    scale_axis : {0, 1} | None, default=None
        Whether to standardize a dimension between 0 and 1.
        If 0, standardize each variable (column). If 1, standardize each
        row (group when ``aggregate`` is True, cell otherwise). For each
        entry, subtracts the minimum and divides by the maximum.
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
    """
    import numpy as np

    mapping = mapping or aes()

    if "tooltips" in geom_kwargs and geom == "raster":
        msg = "\nWarning: tooltips are not supported for 'raster' geom and will be ignored."
        msg += "\nUse 'tile' geom to enable tooltips."
        warn(msg, stacklevel=1)
        geom_kwargs.pop("tooltips")

    # BUILD: dataframe
    frame = build_frame(
        data=data,
        variable_keys=keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
    )

    # unpivot — keep group_by, plus observation id when not aggregating
    index_columns = [group_by] if aggregate else [observations_name, group_by]
    frame = frame.unpivot(
        on=keys,
        index=index_columns,
        variable_name=variable_column,
        value_name=value_column,
    )

    if aggregate:
        frame = frame.group_by(group_by, variable_column).agg(
            pl.col(value_column).mean()
        )
    frame = frame.drop_nulls()

    # HANDLE: standard scaling
    row_key = group_by if aggregate else observations_name
    if scale_axis == 0:
        frame = frame.with_columns(
            ((pl.col(value_column) - pl.col(value_column).min().over(variable_column))
            / (pl.col(value_column).max().over(variable_column)
               - pl.col(value_column).min().over(variable_column)))
            .alias(value_column)
        )
    elif scale_axis == 1:
        frame = frame.with_columns(
            ((pl.col(value_column) - pl.col(value_column).min().over(row_key))
            / (pl.col(value_column).max().over(row_key)
               - pl.col(value_column).min().over(row_key)))
            .alias(value_column)
        )

    # DETERMINE: y order of groups
    if dendrogram:
        y_order_groups, segments = _get_dendrogram(data, group_by)
    else:
        y_order_groups = (
            frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
        )
        segments = None

    # X positions (variables)
    x_keys = list(keys)
    n_x = len(x_keys)
    x_pos = {k: i for i, k in enumerate(x_keys)}
    frame = frame.with_columns(
        pl.col(variable_column).replace_strict(x_pos, return_dtype=pl.Float64).alias("_x")
    )

    # Y positions
    if aggregate:
        n_y = len(y_order_groups)
        y_pos = {g: i for i, g in enumerate(y_order_groups)}
        frame = frame.with_columns(
            pl.col(group_by)
            .cast(pl.String)
            .replace_strict(y_pos, return_dtype=pl.Float64)
            .alias("_y"),
        )
        group_centers = [float(i) for i in range(n_y)]
        cell_frame = None
    else:
        # sort cells by group order then by barcode for stable layout.
        # rescale _y to span [0, n_x] so the heatmap has a square-ish aspect
        # at default plot sizes (otherwise n_y >> n_x squashes tiles to <1px).
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
        # group centers in y_order order
        centers_frame = (
            cell_frame.group_by(group_by, maintain_order=False)
            .agg(pl.col("_y").mean().alias("_center"), pl.col("_grp_idx").first())
            .sort("_grp_idx")
        )
        group_centers = centers_frame["_center"].to_list()

    # BUILD: heatmap layer
    aes_main = aes(x="_x", y="_y", fill=value_column, **mapping.as_dict())
    geom_kwargs.pop("tooltips", None)  # raster doesn't support; tile gets default below

    if geom == "raster":
        htmp = ggplot(frame) + geom_raster(aes_main, **geom_kwargs) + _THEME_HEATMAP
    else:
        htmp = (
            ggplot(frame)
            + geom_tile(aes_main, **geom_kwargs)
            + _THEME_HEATMAP
        )

    # X scale: variable labels
    htmp += scale_x_continuous(breaks=list(range(n_x)), labels=x_keys)

    # Y scale: groups for aggregate, hidden for non-aggregate
    if aggregate:
        htmp += scale_y_continuous(breaks=list(range(n_y)), labels=y_order_groups)
    else:
        htmp += theme(axis_text_y=element_blank(), axis_ticks_y=element_blank())

    # GROUP color bar on left for non-aggregate
    if not aggregate and group_bars:
        bar_width = max(1.0, n_x * _GROUP_BAR_RATIO)
        bar_xend = -_GROUP_BAR_GAP
        bar_x = bar_xend - bar_width
        bar_x_mid = (bar_x + bar_xend) / 2
        group_bar_frame = (
            cell_frame.group_by(group_by, maintain_order=False)
            .agg(
                pl.col("_y").min().alias("y_min"),
                pl.col("_y").max().alias("y_max"),
                pl.col("_grp_idx").first(),
            )
            .sort("_grp_idx")
            .with_columns(pl.lit(bar_x_mid).alias("x"))
        )
        htmp += geom_segment(
            data=group_bar_frame,
            mapping=aes(x="x", xend="x", y="y_min", yend="y_max", color=group_by),
            size=6,
        )

    # GROUP separator lines (horizontal, within heatmap)
    if group_lines and len(y_order_groups) > 1:
        x_start = -0.5
        x_end = n_x - 0.5
        if aggregate:
            line_ys = [i + 0.5 for i in range(len(y_order_groups) - 1)]
        else:
            boundaries = (
                cell_frame.group_by(group_by, maintain_order=False)
                .agg(pl.col("_y").max().alias("y_max"), pl.col("_grp_idx").first())
                .sort("_grp_idx")
                .head(len(y_order_groups) - 1)["y_max"]
                .to_list()
            )
            half_step = (n_x - 1) / max(n_y - 1, 1) / 2
            line_ys = [y + half_step for y in boundaries]
        lines_frame = pl.DataFrame(
            {
                "x": [x_start] * len(line_ys),
                "xend": [x_end] * len(line_ys),
                "y": line_ys,
                "yend": line_ys,
            }
        )
        htmp += geom_segment(
            data=lines_frame,
            mapping=aes(x="x", xend="xend", y="y", yend="yend"),
            color=group_lines_color,
            size=group_lines_size,
        )

    # DENDROGRAM (right side, along y-axis)
    if dendrogram:
        n_groups = len(y_order_groups)
        y_dendrogram_size = n_x * _DENDROGRAM_RATIO
        leaf_xp = np.arange(n_groups, dtype=float)
        seg_y = np.interp(segments["x"].to_numpy(), leaf_xp, group_centers)
        seg_yend = np.interp(segments["xend"].to_numpy(), leaf_xp, group_centers)

        frame_y_dendrogram = pl.DataFrame(
            {
                "x": (n_x - 0.5) + segments["y"].to_numpy() * y_dendrogram_size,
                "xend": (n_x - 0.5) + segments["yend"].to_numpy() * y_dendrogram_size,
                "y": seg_y,
                "yend": seg_yend,
            }
        )
        htmp += geom_segment(
            data=frame_y_dendrogram,
            mapping=aes(x="x", y="y", xend="xend", yend="yend"),
            color="black",
            size=0.5,
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
