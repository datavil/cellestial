from __future__ import annotations

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
from cellestial.single.heatmap._key_groups import (
    _key_groups_bar_y,
    _key_groups_layers,
    _resolve_key_groups,
    _resolve_padding,
)
from cellestial.single.heatmap._rank_genes_groups import _resolve_rank_genes_groups_args
from cellestial.themes import _THEME_DOTPLOT
from cellestial.util import (
    _color_gradient,
    _fill_gradient,
    _get_dendrogram,
    _get_dendrogram_path_frame,
    _resolve_tooltips,
    _validate_tooltips,
    _warn,
)
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lets_plot.plot.core import PlotSpec


def dotplot(
    data: AnnData,
    keys: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    group_by: str | None = None,
    *,
    markers: bool | str = False,
    n_genes: int = 5,
    groups: Sequence[str] | None = None,
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
    dendrogram_key: str | None = None,
    dendrogram_kwargs: dict | None = None,
    rectangle_size: float = 0.8,
    rectangle_color: str = "#3f3f3f",
    rectangle_kwargs: dict | None = None,
    key_labels: bool = True,
    key_labels_text_size: float = 1.0,
    key_labels_bracket_size: float = 0.6,
    key_labels_text_color: str = "black",
    key_labels_bracket_color: str = "black",
    key_labels_width: float = 0.6,
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
    keys : Sequence[str] | Mapping[str, Sequence[str]] | None, default=None
        Variable keys to include, placed on the x-axis. A mapping assigns
        keys to group labels (no key in more than one group). Must be
        `None` when `markers` is set.
    group_by : str | None, default=None
        The key to group the data by. Inferred from a precomputed ranking
        when `markers` is set.
    markers : bool | str, default=False
        Derive `keys` from a precomputed ranking. Pass `True` to use the
        default ranking key, or a string to read a custom key (e.g.
        `"rank_genes_groups_wilcoxon"`).
    n_genes : int, default=5
        Number of top genes to take per group when `markers` is set.
    groups : Sequence[str] | None, default=None
        Subset of groups to include when `markers` is set;
        `None` keeps all groups in their stored order.
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
        Whether to add a dendrogram for the `group_by` axis.
        Uses `scanpy.tl.dendrogram` if not already computed.
        When True, group order is determined by the dendrogram.
    dendrogram_color : str, default='black'
        Color of the dendrogram segments.
    dendrogram_size : float, default=0.5
        Size (thickness) of the dendrogram segments.
    dendrogram_key : str | None, default=None
        Specific key holding the precomputed dendrogram.
        By default, `dendrogram_{group_by}` is used.
    dendrogram_kwargs : dict | None, default=None
        Additional parameters to pass to the dendrogram geom_segment.
    rectangle_size : float, default=0.8
        Size (thickness) of the rectangle border.
    rectangle_color : str, default='#3f3f3f'
        Color of the rectangle border.
    rectangle_kwargs : dict | None, default=None
        Additional parameters to pass to the rectangle geom_rect.
    key_labels : bool, default=True
        Whether to draw bracket labels above the plot when `keys` is a mapping.
    key_labels_text_size : float, default=1.0
        Scale multiplier on the auto-computed bracket label text size.
    key_labels_bracket_size : float, default=0.6
        Size (thickness) of the bracket lines.
    key_labels_text_color : str, default='black'
        Color of the bracket label text.
    key_labels_bracket_color : str, default='black'
        Color of the bracket lines.
    key_labels_width : float, default=0.6
        Bracket width (in column units) for a singleton group. Multi-key groups
        extend `key_labels_width / 2` past the first and last key on each side.
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

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If `markers` is enabled and the requested ranking result or
        group is missing.
    DuplicateKeysError
        If a mapping passed to `keys` assigns the same key to multiple groups.
    ValueError
        If `keys` and `group_by` are missing while `markers` is
        disabled.

    Examples
    --------
    A simple dotplot.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

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

    Plot the top genes from a precomputed ranking:

    .. jupyter-execute::

        sc.tl.rank_genes_groups(data, groupby="cell_type_lvl1")
        cl.dotplot(data, markers=True, n_genes=5)

    """
    # HANDLE: Data types
    if not isinstance(data, AnnData):
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    if markers:
        keys, group_by = _resolve_rank_genes_groups_args(
            data,
            rank_genes_groups=markers,
            n_genes=n_genes,
            groups=groups,
            keys=keys,
            group_by=group_by,
        )
    elif keys is None or group_by is None:
        msg = "`keys` and `group_by` are required (or enable `markers` to derive them)."
        raise ValueError(msg)

    mapping = mapping or aes()

    # RESOLVE: dict ``keys`` into a flat list while preserving mapping order
    keys, key_groups = _resolve_key_groups(keys, key_labels=key_labels)

    # BUILD: dataframe
    frame = build_frame(
        data=data,
        axis=0,
        variable_keys=keys,
    )
    # WARN: negative expression makes the percent-expressed (dot size) misleading
    overall_min = frame.select(pl.min_horizontal(pl.col(keys).min())).item()
    if overall_min is not None and overall_min < 0:
        _warn(
            "Expression matrix contains negative values, which suggests scaled data. "
            "Dot size (percentage of cells above `threshold`, default 0) assumes "
            "non-negative expression and may be misleading here; pass raw or "
            "log-normalized expression, or set `threshold` explicitly."
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
    # HANDLE: Sorting pseudo-categorical integer labels numerically when possible.
    numeric_group_by = "__cellestial_group_by_numeric"
    frame = frame.with_columns(
        pl.col(group_by).cast(pl.String).cast(pl.Int64, strict=False).alias(numeric_group_by)
    )
    if frame[numeric_group_by].null_count() == 0:
        frame = (
            frame.sort(numeric_group_by, descending=True)
            .drop(numeric_group_by)
            .with_columns(pl.col(group_by).cast(pl.String).cast(pl.Categorical))
        )
    else:
        frame = frame.drop(numeric_group_by)
    # perform sorting
    if sort_by is not None:
        frame = frame.sort(
            by=sort_by,
            descending=(sort_order == "descending"),
        )

    # DETERMINE: y order of groups
    if dendrogram:
        y_order_groups, paths = _get_dendrogram(data, group_by, use_key=dendrogram_key)
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
    data_top = n_y - 0.5
    y_max_limit = data_top
    key_groups_total_span: float | None = None
    if key_labels and key_groups is not None:
        # Extend the y limit upward to fit the bracket bar plus rotated label.
        data_range = data_top + 0.5
        key_groups_padding = _resolve_padding(
            key_groups,
            padding=None,
            data_range=data_range,
            size_unit="y",
            label_size_scale=key_labels_text_size,
        )
        key_groups_total_span = data_range + key_groups_padding
        y_max_limit = data_top + key_groups_padding
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

    # KEY-GROUP brackets above the data area when ``keys`` was a mapping.
    if key_labels and key_groups is not None:
        assert key_groups_total_span is not None
        bar_y = _key_groups_bar_y(data_top, total_span=key_groups_total_span)
        for layer in _key_groups_layers(
            key_groups,
            y=bar_y,
            total_span=key_groups_total_span,
            text_color=key_labels_text_color,
            bracket_color=key_labels_bracket_color,
            bracket_size=key_labels_bracket_size,
            width=key_labels_width,
            label_size_scale=key_labels_text_size,
            label_size_span=n_x,
            size_unit="x",
        ):
            dtplt += layer

    # BORDER: rectangle around data area only (keeps dendrogram outside the frame)
    if rectangle:
        dtplt += geom_rect(
            data={
                "xmin": [-0.5],
                "xmax": [n_x - 0.5],
                "ymin": [-0.5],
                "ymax": [data_top],
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
