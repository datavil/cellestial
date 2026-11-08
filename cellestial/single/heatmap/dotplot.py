from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "ggrust.plot.core", "mudata"]

from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from ggrust import (
    aes,
    element_line,
    element_rect,
    geom_point,
    ggplot,
    ggtb,
    guides,
    scale_size,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from ggrust.plot.core import FeatureSpec
from mudata import MuData

from cellestial.frames import build_frame
from cellestial.single.heatmap.utilities import (
    _key_groups_guide,
    _key_groups_theme,
    _resolve_key_groups,
    _resolve_rank_genes_groups_args,
)
from cellestial.themes import _THEME_DOTPLOT
from cellestial.util import (
    _color_gradient,
    _dendrogram_line_theme,
    _dendrogram_y_axis,
    _fill_gradient,
    _get_dendrogram_tree,
    _resolve_tooltips,
    _validate_tooltips,
    _warn,
)
from cellestial.util.dendrogram import _GEOM_LINE_UNIT
from cellestial.util.errors import _unsupported_data_type
from cellestial.util.utilities import _container_column, _modality_source

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ggrust.plot.core import PlotSpec


def dotplot(
    data: AnnData | MuData,
    keys: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    group_by: str | None = None,
    *,
    modality: str | None = None,
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
    midpoint: Literal["mean", "median", "mid"] | float = "mid",
    sort_by: str | Sequence[str] | None = None,
    sort_order: Literal["ascending", "descending"] = "descending",
    percentage_column: str = "pct_exp",
    mean_column: str = "avg_exp",
    rectangle: bool = True,
    dendrogram: bool = False,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_key: str | None = None,
    rectangle_size: float = 0.8,
    rectangle_color: str = "#3f3f3f",
    rectangle_kwargs: dict | None = None,
    key_labels: bool = True,
    key_labels_bracket_size: float = 0.6,
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
    data : AnnData | MuData
        The single-cell data object.
    keys : Sequence[str] | Mapping[str, Sequence[str]] | None, default=None
        Variable keys to include, placed on the x-axis. A mapping assigns
        keys to group labels (no key in more than one group). Must be
        `None` when `markers` is set.
    modality : str | None, default=None
        Which modality's stored analysis results to use when `markers` or
        `dendrogram` is enabled. Required for a multimodal object holding more
        than one modality, and not accepted otherwise.
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
    variable_column : str, default='variable'
        Name for the variable column after unpivoting.
    color_low : str, default='#e6e6e6'
        Color for low values in the gradient.
    color_mid : str | None, default=None
        Color for mid values in the gradient.
    color_high : str, default='#D2042D'
        Color for high values in the gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='mid'
        Midpoint for the color gradient.
    sort_by : str | Sequence[str] | None, default=None
        The column or columns to sort the results by.
    sort_order : str, default='descending'
        The sort order, either 'ascending' or 'descending'.
    percentage_column : str, default='pct_exp'
        The name of the percentage column.
    mean_column : str, default='avg_exp'
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
    rectangle_size : float, default=0.8
        Size (thickness) of the rectangle border.
    rectangle_color : str, default='#3f3f3f'
        Color of the rectangle border.
    rectangle_kwargs : dict | None, default=None
        Additional parameters for the border's `element_rect` (e.g. `linetype`).
    key_labels : bool, default=True
        Whether to draw bracket labels above the plot when `keys` is a mapping.
    key_labels_bracket_size : float, default=0.6
        Size (thickness) of the bracket lines.
    key_labels_bracket_color : str, default='black'
        Color of the bracket lines.
    key_labels_width : float, default=0.6
        Bracket width (in column units) for a singleton group. Multi-key groups
        extend `key_labels_width / 2` past the first and last key on each side.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
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

    Notes
    -----
    When `dendrogram=True` and the requested dendrogram is absent from
    `data.uns`, plot construction calls `scanpy.tl.dendrogram` and stores the
    result in `data.uns`. This mutates the input `AnnData`. Precompute the
    dendrogram under `dendrogram_key` to avoid mutation during plotting.

    Examples
    --------
    A simple dotplot.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

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
    if not isinstance(data, (AnnData, MuData)):
        raise _unsupported_data_type(data, AnnData, MuData)

    if markers:
        results, local_group_by = _modality_source(data, modality, group_by)
        keys, group_by = _resolve_rank_genes_groups_args(
            results,
            rank_genes_groups=markers,
            n_genes=n_genes,
            groups=groups,
            keys=keys,
            group_by=local_group_by,
        )
        group_by = _container_column(data, modality, group_by)
    elif keys is None or group_by is None:
        msg = "`keys` and `group_by` are required (or enable `markers` to derive them)."
        raise ValueError(msg)

    mapping = mapping or aes()

    # RESOLVE: dict `keys` into a flat list while preserving mapping order
    keys, key_groups = _resolve_key_groups(keys, key_labels=key_labels)

    # BUILD: dataframe. Only `group_by` is needed from the observation metadata;
    # the cell identifier is unused by the dotplot aggregation.
    frame = build_frame(
        data=data,
        axis=0,
        variable_keys=keys,
        observation_id_column=None,
        metadata_columns=[group_by],
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

    # CRITICAL PARTS: Dataframe Operations
    # 1. Aggregate per group while wide, so the unpivoted frame holds
    # groups x keys rows instead of cells x keys.
    # Cast to the dtype unpivot would supercast the keys to.
    value_dtype = frame.head(0).select(keys).unpivot(value_name=mean_column).schema[mean_column]
    finite_values = [pl.col(key).cast(value_dtype).filter(pl.col(key).is_finite()) for key in keys]
    percentage_names = [f"__cellestial_percentage_{index}" for index in range(len(keys))]
    frame = frame.group_by(group_by).agg(
        [values.mean().alias(key) for values, key in zip(finite_values, keys, strict=True)]
        + [
            (values > threshold).mean().mul(100).alias(name)
            for values, name in zip(finite_values, percentage_names, strict=True)
        ]
    )
    # 2. Unpivot means and percentages; both unpivots keep the same row order.
    means = frame.unpivot(
        on=keys, index=group_by, variable_name=variable_column, value_name=mean_column
    )
    percentages = frame.unpivot(on=percentage_names, index=group_by, value_name=percentage_column)
    # Keys with no finite value in a group get no row.
    frame = means.with_columns(percentages[percentage_column]).filter(
        pl.col(mean_column).is_not_null()
    )
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
        tree = _get_dendrogram_tree(
            *_modality_source(data, modality, group_by), use_key=dendrogram_key
        )
        y_order_groups = list(tree.ordered_labels)
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
        defaults=[group_by, variable_column, mean_column, percentage_column],
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
        "size": percentage_column,
        color_or_fill: mean_column,
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
            frame[mean_column],
            color_low=color_low,
            color_mid=color_mid,
            color_high=color_high,
            midpoint=midpoint,
        )
        + _size_scale
    )

    # AXES: discrete labels via continuous breaks; tight limits keep ticks against the rectangle
    data_top = n_y - 0.5
    dtplt += scale_x_continuous(
        breaks=list(range(n_x)),
        labels=x_keys,
        limits=[-0.5, n_x - 0.5],
        expand=[0, 0],
    )
    # DENDROGRAM (right side, along y-axis): the tree orders and labels the groups
    if dendrogram:
        dtplt += _dendrogram_y_axis(tree, limits=[-0.5, data_top])
    else:
        dtplt += scale_y_continuous(
            breaks=list(range(n_y)),
            labels=y_order_groups,
            limits=[-0.5, data_top],
            expand=[0, 0],
        )

    # KEY-GROUP brackets above the data area when `keys` was a mapping.
    if key_labels and key_groups is not None:
        dtplt += guides(x=_key_groups_guide(key_groups, width=key_labels_width))

    if not rectangle:
        dtplt += theme(axis_line=element_line(color="#1f1f1f"))

    # ADD: layers
    dtplt += _THEME_DOTPLOT
    # BORDER: the panel border boxes the data area only (brackets and tree sit outside it)
    if rectangle:
        dtplt += theme(
            panel_border=element_rect(
                color=rectangle_color,
                size=rectangle_size * _GEOM_LINE_UNIT,
                **(rectangle_kwargs or {}),
            )
        )
    if key_labels and key_groups is not None:
        dtplt += _key_groups_theme(color=key_labels_bracket_color, size=key_labels_bracket_size)
    if dendrogram:
        dtplt += _dendrogram_line_theme(color=dendrogram_color, size=dendrogram_size)

    # HANDLE: interactive
    if interactive:
        dtplt += ggtb(size_zoomin=-1)

    return dtplt
