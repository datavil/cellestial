from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "ggrust.plot.core", "mudata"]

from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from ggrust import (
    aes,
    element_line,
    element_rect,
    geom_polygon,
    ggplot,
    ggtb,
    guides,
    scale_x_continuous,
    scale_y_continuous,
    theme,
)
from ggrust.plot.core import FeatureSpec
from mudata import MuData

from cellestial.frames import build_frame
from cellestial.single.heatmap.utilities import (
    _compute_violin_polygons,
    _key_groups_guide,
    _key_groups_theme,
    _resolve_key_groups,
    _resolve_rank_genes_groups_args,
)
from cellestial.themes import _THEME_DOTPLOT
from cellestial.util import (
    _dendrogram_line_theme,
    _dendrogram_y_axis,
    _drop_nonfinite_rows,
    _fill_gradient,
    _get_dendrogram_tree,
    _resolve_tooltips,
    _validate_tooltips,
)
from cellestial.util.dendrogram import _GEOM_LINE_UNIT
from cellestial.util.errors import _unsupported_data_type
from cellestial.util.utilities import _container_column, _modality_source

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ggrust.plot.core import PlotSpec


def stacked_violin(
    data: AnnData | MuData,
    keys: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    group_by: str | None = None,
    *,
    modality: str | None = None,
    markers: bool | str = False,
    n_genes: int = 5,
    groups: Sequence[str] | None = None,
    mapping: FeatureSpec | None = None,
    threshold: float | None = None,
    scale: Literal["area", "count", "width"] = "width",
    width_scale: float = 0.85,
    height_scale: float = 0.85,
    n_points: int = 64,
    kde_max_samples: int | None = 1200,
    fill_statistic: Literal["median", "mean", "group", "variable"] | None = "median",
    size: float = 0.2,
    color_low: str = "#F5F5F5",
    color_mid: str | None = None,
    color_high: str = "#00008B",
    midpoint: Literal["mean", "median", "mid"] | float = "mid",
    geom_fill: str | None = None,
    geom_color: str | None = "#1f1f1f",
    dendrogram: bool = False,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_key: str | None = None,
    rectangle: bool = True,
    rectangle_size: float = 0.8,
    rectangle_color: str = "#3f3f3f",
    rectangle_kwargs: dict | None = None,
    key_labels: bool = True,
    key_labels_bracket_size: float = 0.6,
    key_labels_bracket_color: str = "black",
    key_labels_width: float = 0.6,
    aggregate_column: str = "expression",
    value_column: str = "value",
    variable_column: str = "variable",
    observation_id_column: str = "Barcode",
    variable_id_column: str = "Variable",
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    **geom_kwargs,
) -> PlotSpec:
    """
    Stacked Violin Plot.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    keys : Sequence[str] | Mapping[str, Sequence[str]] | None, default=None
        Variable keys laid out along the x-axis, one column of violins per
        key. A mapping assigns keys to group labels (no key in more than one
        group). Must be `None` when `markers` is set.
    modality : str | None, default=None
        Which modality's stored analysis results to use when `markers` or
        `dendrogram` is enabled. Required for a multimodal object holding more
        than one modality, and not accepted otherwise.
    group_by : str | None, default=None
        The key used to group observations along the y-axis. Inferred from a
        precomputed ranking when `markers` is set.
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
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    scale : {'area', 'count', 'width'}, default='width'
        Method for scaling violin widths.
        `'width'`, every violin has the same maximum width.
        `'count'`, width is proportional to the number of observations.
        `'area'`, widths preserve density area across groups within a variable.
    width_scale : float, default=0.85
        Maximum total width of a violin in x units (1 unit = one variable column).
    height_scale : float, default=0.85
        Total height of a violin in y units (1 unit = one group row).
    n_points : int, default=64
        Number of grid points for the kernel density estimate.
    kde_max_samples : int | None, default=1200
        If set, subsample each (variable, group) to at most this many cells
        before fitting the KDE. Color values (mean/median) and the "count"
        scale use the full sample size. Sampling is deterministic. `None`
        fits the KDE on every cell (slower at scale, no shape approximation).
    fill_statistic : {'median', 'mean', 'group', 'variable'} | None, default='median'
        Which value drives the fill aesthetic of each violin.
        `'median'` colors by median expression per (variable, group).
        `'mean'` colors by mean expression per (variable, group).
        `'group'` colors by `group_by` (categorical palette).
        `'variable'` colors by `variable_column` (categorical palette).
        `None` disables fill mapping (use `geom_fill` for a static fill).
    size : float, default=0.2
        Stroke size (edge width) of the violins.
    color_low : str, default='#F5F5F5'
        Color for low values in the gradient (used when `fill_statistic='mean'`).
    color_mid : str | None, default=None
        Color for mid values in the gradient.
    color_high : str, default='#00008B'
        Color for high values in the gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='mid'
        Midpoint for the color gradient.
    geom_fill : str | None, default=None
        Static fill color for all violins. Overrides any fill aesthetic.


    geom_color : str | None, default='#1f1f1f'
        Border color for all violins.
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
    rectangle : bool, default=True
        Whether to add a rectangle border around the data area.
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
    aggregate_column : str, default='expression'
        Name of the per-(variable, group) aggregate column attached to each violin
        (median or mean, selected by `fill_statistic`).
    value_column : str, default='value'
        Name for the value column after unpivoting.
    variable_column : str, default='variable'
        Name for the variable column after unpivoting.
    observation_id_column : str, default='Barcode'
        The name of the observations column.
    variable_id_column : str, default='Variable'
        Name for the variables index column.
    tooltips : {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    **geom_kwargs
        Additional parameters for the `geom_polygon` layer.
        For further detail on geom_polygon.
        https://lets-plot.org/python/pages/api/lets_plot.geom_polygon.html

    Returns
    -------
    PlotSpec
        Stacked violin plot.

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
    A simple stacked violin plot of marker genes across cell types.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

        cl.stacked_violin(
            data,
            keys=markers,
            group_by="cell_type_lvl1",
        )

    Reorder groups along the y-axis with a dendrogram.

    .. jupyter-execute::
        :emphasize-lines: 5

        cl.stacked_violin(
            data,
            keys=markers,
            group_by="cell_type_lvl1",
            dendrogram=True,
        )

    Color violins by `group_by` instead of the per-cell aggregate.

    .. jupyter-execute::
        :emphasize-lines: 5

        cl.stacked_violin(
            data,
            keys=markers,
            group_by="cell_type_lvl1",
            fill_statistic="group",
        )

    Plot the top genes from a precomputed ranking:

    .. jupyter-execute::

        sc.tl.rank_genes_groups(data, groupby="cell_type_lvl1")
        cl.stacked_violin(data, markers=True, n_genes=5)
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
    keys_list, key_groups = _resolve_key_groups(keys, key_labels=key_labels)

    # BUILD: dataframe. Only `group_by` is needed from the observation metadata
    # alongside the per-cell identifier used as the unpivot index.
    frame = build_frame(
        data=data,
        axis=0,
        variable_keys=keys_list,
        observation_id_column=observation_id_column,
        variable_id_column=variable_id_column,
        metadata_columns=[group_by],
    )
    # DROP: rows with null group_by to avoid null labels downstream
    frame = frame.filter(pl.col(group_by).is_not_null())

    # CRITICAL PARTS: Dataframe Operations
    # 1. Unpivot to long format
    frame = frame.unpivot(
        on=keys_list,
        index=[observation_id_column, group_by],
        variable_name=variable_column,
        value_name=value_column,
    )
    frame = _drop_nonfinite_rows(frame, [value_column])
    # 2. Apply threshold filter
    if threshold is not None:
        frame = frame.filter(pl.col(value_column) >= threshold)

    # DETERMINE: y order of groups (dendrogram or first-seen order)
    if dendrogram:
        tree = _get_dendrogram_tree(
            *_modality_source(data, modality, group_by), use_key=dendrogram_key
        )
        y_order_groups = list(tree.ordered_labels)
    else:
        y_order_groups = (
            frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
        )

    x_keys = keys_list
    n_x = len(x_keys)
    n_y = len(y_order_groups)

    # COMPUTE: KDE polygons, one violin per (variable, group)
    poly_frame = _compute_violin_polygons(
        frame,
        variable_column=variable_column,
        value_column=value_column,
        group_by=group_by,
        x_keys=x_keys,
        y_order_groups=y_order_groups,
        n_points=n_points,
        scale=scale,
        width_scale=width_scale,
        height_scale=height_scale,
        aggregate="mean" if fill_statistic == "mean" else "median",
        aggregate_column=aggregate_column,
        kde_max_samples=kde_max_samples,
    )

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=[],
        defaults=[variable_column, group_by, aggregate_column],
    )
    _validate_tooltips(tooltips, poly_frame)

    # DEFINE: mapping with defaults
    _mapping = {"x": "x", "y": "y", "group": "polygon_id"}
    if fill_statistic in ("mean", "median"):
        _mapping["fill"] = aggregate_column
    elif fill_statistic == "group":
        _mapping["fill"] = group_by
    elif fill_statistic == "variable":
        _mapping["fill"] = variable_column
    _mapping.update(mapping.as_dict())

    # BUILD: stacked violin
    plot = ggplot(poly_frame) + geom_polygon(
        aes(**_mapping),
        fill=geom_fill,
        color=geom_color,
        tooltips=tooltips,
        size=size,
        **geom_kwargs,
    )

    # ADD: continuous fill gradient when coloring by mean/median expression
    if fill_statistic in ("mean", "median") and poly_frame.height > 0:
        plot += _fill_gradient(
            poly_frame[aggregate_column],
            color_low=color_low,
            color_mid=color_mid,
            color_high=color_high,
            midpoint=midpoint,
        )

    # AXES: discrete labels via continuous breaks; tight limits keep ticks against the rectangle
    data_top = n_y - 0.5
    plot += scale_x_continuous(
        breaks=list(range(n_x)),
        labels=x_keys,
        limits=[-0.5, n_x - 0.5],
        expand=[0, 0],
    )
    # DENDROGRAM (right side, along y-axis): the tree orders and labels the groups
    if dendrogram:
        plot += _dendrogram_y_axis(tree, limits=[-0.5, data_top])
    else:
        plot += scale_y_continuous(
            breaks=list(range(n_y)),
            labels=y_order_groups,
            limits=[-0.5, data_top],
            expand=[0, 0],
        )

    # KEY-GROUP brackets above the data area when `keys` was a mapping.
    if key_labels and key_groups is not None:
        plot += guides(x=_key_groups_guide(key_groups, width=key_labels_width))

    if not rectangle:
        plot += theme(axis_line=element_line(color="#1f1f1f"))

    # ADD: theme
    plot += _THEME_DOTPLOT
    # BORDER: the panel border boxes the data area only (brackets and tree sit outside it)
    if rectangle:
        plot += theme(
            panel_border=element_rect(
                color=rectangle_color,
                size=rectangle_size * _GEOM_LINE_UNIT,
                **(rectangle_kwargs or {}),
            )
        )
    if key_labels and key_groups is not None:
        plot += _key_groups_theme(color=key_labels_bracket_color, size=key_labels_bracket_size)
    if dendrogram:
        plot += _dendrogram_line_theme(color=dendrogram_color, size=dendrogram_size)

    # HANDLE: interactive
    if interactive:
        plot += ggtb(size_zoomin=-1)

    return plot
