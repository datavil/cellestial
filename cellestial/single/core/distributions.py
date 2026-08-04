from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lets_plot import gggrid, ggtb
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.frames import build_frame
from cellestial.single.core.distribution import boxplot, histogram, violin
from cellestial.util import (
    _collect_aes_columns,
    _determine_axis,
    _resolve_tooltips,
    _share_axis,
    _share_ticks,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.subplots import SupPlotsSpec
    from mudata import MuData


def violins(
    data: AnnData | MuData,
    keys: Sequence[str],
    *,
    group_by: str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str | None = None,
    geom_color: str | None = None,
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
    point_mapping: FeatureSpec | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    show_points: bool = True,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    # multi plot args
    share_axis: bool = False,
    share_ticks: bool = False,
    layers: Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None = None,
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list[float] | None = None,
    heights: list[float] | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    guides: str = "auto",
    # other kwargs
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs,
) -> SupPlotsSpec:
    """
    Violin Plots.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : list[str] | tuple[str] | Sequence[str]
        The keys to get the values (numerical).
        e.g., ['total_counts', 'pct_counts_in_top_50_genes'] or a list of gene names.
    group_by : str | None, default=None
        Column to group observations on the x-axis.
        If not provided, falls back to `fill`, `color`, or the variable column.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups, keeping rows where `group_by` matches any
        of them. Categorical grouping columns only.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out rows where `group_by` matches any
        of them. Categorical grouping columns only.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        Shortcut for mapping=aes(color=...)
        e,g., 'cell_type' or 'leiden'.
        For a constant color on every geom, use `geom_color`.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        Shortcut for mapping=aes(fill=...)
        e,g., 'cell_type' or 'leiden'.
        For a constant fill on every geom, use `geom_fill`.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    geom_fill : str | None, default=None
        Fill color for all violins in the violin plot.
    geom_color : str | None, default=None
        Border color for all violins in the violin plot.
    point_color : str, default='#1f1f1f'
        Color for the points in the violin plot.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the violin plot.
    point_size : float, default=0.5
        Size for the points in the violin plot.
    point_geom : {'jitter','point','sina'}, default is 'jitter'
        Geom type of the points, default is geom_jitter.
    point_mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the points, the result of `aes()`.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    variable_column : str, default='variable'
        The name of the variable column in the dataframe.
    value_column : str, default='value'
        Additional layers to add to the plot.
        The name of the value column in the dataframe.
    share_ticks : bool, default=False
        Whether to share the ticks across all plots.
        If True, only X tick texts on bottom row and Y tick text on left column are shown.
    share_axis : bool, default=False
        Whether to share the axis across all plots.
        If True, only X axis on bottom row and Y axis on left column is shown.
    layers : Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None, default=None
        Additional layers to add to the plot.
    ncol : int, default=None
        Number of columns in grid. If not specified, shows plots horizontally, in one row.
    sharex, sharey : bool, default=None
        Controls sharing of axis limits between subplots in the grid.
        `all`/True - share limits between all subplots.
        `none`/False - do not share limits between subplots.
        `row` - share limits between subplots in the same row.
        `col` - share limits between subplots in the same column.
    widths : list[float], default=None
        Relative width of each column of grid, left to right.
    heights : list[float], default=None
        Relative height of each row of grid, top-down.
    hspace : float | None, default=None
        Cell horizontal spacing in px.
    vspace : float | None, default=None
        Cell vertical spacing in px.
    fit : bool | None, default=None
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool | None, default=None
        If True, align inner areas (i.e. “geom” bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot “inner areas” layouting.
    guides : str, default='auto'
        Specifies how guides (legends and colorbars) should be treated in the layout.
            - 'collect' collect guides from all subplots, removing duplicates.
            - 'keep' keep guides in their original subplots; do not collect at this level.
            - 'auto' allow guides to be collected if an upper-level layout uses guides='collect';

        otherwise, keep them in subplots.
        Duplicates are identified by comparing visual properties:
        For legends: title, labels, and all aesthetic values (colors, shapes, sizes, etc.).
        For colorbars: title, domain limits, breaks, and color gradient.

        For more information on gggrid parameters:
        https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **geom_kwargs
        Additional parameters for the `geom_violin` layer.
        For more information on geom_violin parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_violin.html

    Returns
    -------
    SupPlotsSpec
        Violin Plots.

    Examples
    --------
    Violin Plots.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.violins(
            data,
            ["n_genes_by_counts", "HLA-DRA", "log1p_total_counts_mt", "pct_counts_hb"],
            fill="cell_type_lvl1",
            layers=[scale_y_log2()],
            point_size=0.3,
            ncol=2,
        )
    """
    # BUILD: one shared (wide) frame for all keys, instead of rebuilding per key.
    # Axis is resolved once over all keys; mixed-axis key sets raise in `_determine_axis`.
    axis = (
        _determine_axis(data=data, keys=keys, companions=[group_by, fill, color])
        if axis is None
        else axis
    )
    if isinstance(add_keys, str):
        add_keys = [add_keys]
    variable_keys: list[str] = []
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[*keys, group_by, color, fill, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    _collect_aes_columns(
        data,
        keys=[],
        mapping=point_mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    # Tooltips are shared across subplots; pull their fields into the shared
    # frame so each subplot's unpivot index can keep them.
    _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[variable_column, value_column],
        metadata_columns=metadata_columns,
        axis=axis,
    )
    observation_column_name = None if tooltips == "none" else observations_name
    variable_column_name = None if tooltips == "none" else variables_name
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observation_column_name,
        variables_name=variable_column_name,
        metadata_columns=metadata_columns,
    )

    plots = []
    for i, key in enumerate(keys):
        plot = violin(
            data=data,
            key=key,
            frame=frame,
            group_by=group_by,
            groups=groups,
            drop=drop,
            mapping=mapping,
            axis=axis,
            color=color,
            fill=fill,
            threshold=threshold,
            add_keys=add_keys,
            tooltips=tooltips,
            geom_fill=geom_fill,
            geom_color=geom_color,
            point_color=point_color,
            point_alpha=point_alpha,
            point_size=point_size,
            point_geom=point_geom,
            point_mapping=point_mapping,
            observations_name=observations_name,
            variables_name=variables_name,
            show_points=show_points,
            value_column=value_column,
            variable_column=variable_column,
            point_kwargs=point_kwargs,
            **geom_kwargs,
        )
        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        if share_ticks:
            plot = _share_ticks(plot, i, keys, ncol)
        if share_axis:
            plot = _share_axis(plot, i, keys, ncol, "axis")
        plots.append(plot)

    dsts = gggrid(
        plots,
        ncol=ncol,  # ty:ignore[invalid-argument-type]
        sharex=sharex,  # ty:ignore[invalid-argument-type]
        sharey=sharey,  # ty:ignore[invalid-argument-type]
        widths=widths,  # ty:ignore[invalid-argument-type]
        heights=heights,  # ty:ignore[invalid-argument-type]
        hspace=hspace,  # ty:ignore[invalid-argument-type]
        vspace=vspace,  # ty:ignore[invalid-argument-type]
        fit=fit,  # ty:ignore[invalid-argument-type]
        align=align,  # ty:ignore[invalid-argument-type]
        guides=guides,
    )

    if interactive:
        dsts += ggtb(size_zoomin=-1)

    return dsts


def boxplots(
    data: AnnData | MuData,
    keys: Sequence[str],
    *,
    group_by: str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str | None = None,
    geom_color: str | None = None,
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
    point_mapping: FeatureSpec | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    show_points: bool = True,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    # multi plot args
    share_axis: bool = False,
    share_ticks: bool = False,
    layers: Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None = None,
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list[float] | None = None,
    heights: list[float] | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    guides: str = "auto",
    # other kwargs
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs,
) -> SupPlotsSpec:
    """
    Boxplots.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : list[str] | tuple[str] | Sequence[str]
        The keys to get the values (numerical).
        e.g., ['total_counts', 'pct_counts_in_top_50_genes'] or a list of gene names.
    group_by : str | None, default=None
        Column to group observations on the x-axis.
        If not provided, falls back to `fill`, `color`, or the variable column.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups, keeping rows where `group_by` matches any
        of them. Categorical grouping columns only.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out rows where `group_by` matches any
        of them. Categorical grouping columns only.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        Shortcut for mapping=aes(color=...)
        e,g., 'cell_type' or 'leiden'.
        For a constant color on every geom, use `geom_color`.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        Shortcut for mapping=aes(fill=...)
        e,g., 'cell_type' or 'leiden'.
        For a constant fill on every geom, use `geom_fill`.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    geom_fill : str | None, default=None
        Fill color for all boxplots in the boxplot.
    geom_color : str | None, default=None
        Border color for all boxplots in the boxplot.
    point_color : str, default='#1f1f1f'
        Color for the points in the boxplot.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the boxplot.
    point_size : float, default=0.5
        Size for the points in the boxplot.
    point_geom : {'jitter','point','sina'}, default is 'jitter'
        Geom type of the points, default is geom_jitter.
    point_mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the points, the result of `aes()`.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    variable_column : str, default='variable'
        The name of the variable column in the dataframe.
    value_column : str, default='value'
        The name of the value column in the dataframe.
    share_ticks : bool, default=False
        Whether to share the ticks across all plots.
        If True, only X tick texts on bottom row and Y tick text on left column are shown.
    share_axis : bool, default=False
        Whether to share the axis across all plots.
        If True, only X axis on bottom row and Y axis on left column is shown.
    layers : Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None, default=None
        Additional layers to add to the plot.
    ncol : int, default=None
        Number of columns in grid. If not specified, shows plots horizontally, in one row.
    sharex, sharey : bool, default=None
        Controls sharing of axis limits between subplots in the grid.
        `all`/True - share limits between all subplots.
        `none`/False - do not share limits between subplots.
        `row` - share limits between subplots in the same row.
        `col` - share limits between subplots in the same column.
    widths : list[float], default=None
        Relative width of each column of grid, left to right.
    heights : list[float], default=None
        Relative height of each row of grid, top-down.
    hspace : float | None, default=None
        Cell horizontal spacing in px.
    vspace : float | None, default=None
        Cell vertical spacing in px.
    fit : bool | None, default=None
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool | None, default=None
        If True, align inner areas (i.e. “geom” bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot “inner areas” layouting.
    guides : str, default='auto'
        Specifies how guides (legends and colorbars) should be treated in the layout.
            - 'collect' collect guides from all subplots, removing duplicates.
            - 'keep' keep guides in their original subplots; do not collect at this level.
            - 'auto' allow guides to be collected if an upper-level layout uses guides='collect';

        otherwise, keep them in subplots.
        Duplicates are identified by comparing visual properties:
        For legends: title, labels, and all aesthetic values (colors, shapes, sizes, etc.).
        For colorbars: title, domain limits, breaks, and color gradient.

        For more information on gggrid parameters:
        https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **geom_kwargs
        Additional parameters for the `geom_boxplot` layer.
        For more information on geom_boxplot parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_boxplot.html

    Returns
    -------
    SupPlotsSpec
        Boxplots.

    Examples
    --------
    Boxplots.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.boxplots(
            data,
            ["n_genes_by_counts", "HLA-DRA", "log1p_total_counts_mt", "pct_counts_hb"],
            fill="cell_type_lvl1",
            layers=[scale_y_log2()],
            point_size=0.3,
            ncol=2,
        )
    """
    # BUILD: one shared (wide) frame for all keys, instead of rebuilding per key.
    # Axis is resolved once over all keys; mixed-axis key sets raise in `_determine_axis`.
    axis = (
        _determine_axis(data=data, keys=keys, companions=[group_by, fill, color])
        if axis is None
        else axis
    )
    if isinstance(add_keys, str):
        add_keys = [add_keys]
    variable_keys: list[str] = []
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[*keys, group_by, color, fill, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    _collect_aes_columns(
        data,
        keys=[],
        mapping=point_mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    # Tooltips are shared across subplots; pull their fields into the shared
    # frame so each subplot's unpivot index can keep them.
    _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[variable_column, value_column],
        metadata_columns=metadata_columns,
        axis=axis,
    )
    observation_column_name = None if tooltips == "none" else observations_name
    variable_column_name = None if tooltips == "none" else variables_name
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observation_column_name,
        variables_name=variable_column_name,
        metadata_columns=metadata_columns,
    )

    plots = []
    for i, key in enumerate(keys):
        plot = boxplot(
            data=data,
            key=key,
            frame=frame,
            group_by=group_by,
            groups=groups,
            drop=drop,
            mapping=mapping,
            axis=axis,
            color=color,
            fill=fill,
            threshold=threshold,
            add_keys=add_keys,
            tooltips=tooltips,
            geom_fill=geom_fill,
            geom_color=geom_color,
            point_color=point_color,
            point_alpha=point_alpha,
            point_size=point_size,
            point_geom=point_geom,
            point_mapping=point_mapping,
            observations_name=observations_name,
            variables_name=variables_name,
            show_points=show_points,
            value_column=value_column,
            variable_column=variable_column,
            point_kwargs=point_kwargs,
            **geom_kwargs,
        )
        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        if share_ticks:
            plot = _share_ticks(plot, i, keys, ncol)
        if share_axis:
            plot = _share_axis(plot, i, keys, ncol, "axis")

        plots.append(plot)

    dsts = gggrid(
        plots,
        ncol=ncol,  # ty:ignore[invalid-argument-type]
        sharex=sharex,  # ty:ignore[invalid-argument-type]
        sharey=sharey,  # ty:ignore[invalid-argument-type]
        widths=widths,  # ty:ignore[invalid-argument-type]
        heights=heights,  # ty:ignore[invalid-argument-type]
        hspace=hspace,  # ty:ignore[invalid-argument-type]
        vspace=vspace,  # ty:ignore[invalid-argument-type]
        fit=fit,  # ty:ignore[invalid-argument-type]
        align=align,  # ty:ignore[invalid-argument-type]
        guides=guides,
    )

    if interactive:
        dsts += ggtb(size_zoomin=-1)

    return dsts


def histograms(
    data: AnnData | MuData,
    keys: Sequence[str],
    *,
    group_by: str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    bins: int | None = None,
    binwidth: float | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str | None = None,
    geom_color: str | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    # multi plot args
    share_axis: bool = False,
    share_ticks: bool = False,
    layers: Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None = None,
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list[float] | None = None,
    heights: list[float] | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    guides: str = "auto",
    **geom_kwargs,
) -> SupPlotsSpec:
    """
    Histograms.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : list[str] | tuple[str] | Sequence[str]
        The keys to get the values (numerical).
        e.g., ['total_counts', 'pct_counts_in_top_50_genes'] or a list of gene names.
    group_by : str | None, default=None
        Column to filter observations on. Only rows with a non-null value are kept.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups, keeping rows where `group_by` matches any
        of them. Categorical grouping columns only.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out rows where `group_by` matches any
        of them. Categorical grouping columns only.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the histogram (categorical).
        Shortcut for mapping=aes(color=...)
        e,g., 'cell_type' or 'leiden'.
        For a constant color on every geom, use `geom_color`.
    fill : str | None, default=None
        Fill aesthetic to split the histogram (categorical).
        Shortcut for mapping=aes(fill=...)
        e,g., 'cell_type' or 'leiden'.
        For a constant fill on every geom, use `geom_fill`.
    bins : int | None, default=None
        Number of bins. Overridden by `binwidth` if both are provided.
    binwidth : float | None, default=None
        Width of each bin. Takes precedence over `bins`.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    geom_fill : str | None, default=None
        Fill color for all bars in the histogram.
    geom_color : str | None, default=None
        Border color for all bars in the histogram.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    interactive : bool, default=False
        Whether to make the plot interactive.
    variable_column : str, default='variable'
        The name of the variable column in the dataframe.
    value_column : str, default='value'
        The name of the value column in the dataframe.
    share_ticks : bool, default=False
        Whether to share the ticks across all plots.
        If True, only X tick texts on bottom row and Y tick text on left column are shown.
    share_axis : bool, default=False
        Whether to share the axis across all plots.
        If True, only X axis on bottom row and Y axis on left column is shown.
    layers : Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None, default=None
        Additional layers to add to the plot.
    ncol : int, default=None
        Number of columns in grid. If not specified, shows plots horizontally, in one row.
    sharex, sharey : bool, default=None
        Controls sharing of axis limits between subplots in the grid.
        `all`/True - share limits between all subplots.
        `none`/False - do not share limits between subplots.
        `row` - share limits between subplots in the same row.
        `col` - share limits between subplots in the same column.
    widths : list[float], default=None
        Relative width of each column of grid, left to right.
    heights : list[float], default=None
        Relative height of each row of grid, top-down.
    hspace : float | None, default=None
        Cell horizontal spacing in px.
    vspace : float | None, default=None
        Cell vertical spacing in px.
    fit : bool | None, default=None
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool | None, default=None
        If True, align inner areas (i.e. "geom" bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot "inner areas" layouting.
    guides : str, default='auto'
        Specifies how guides (legends and colorbars) should be treated in the layout.
            - 'collect' collect guides from all subplots, removing duplicates.
            - 'keep' keep guides in their original subplots; do not collect at this level.
            - 'auto' allow guides to be collected if an upper-level layout uses guides='collect';

        otherwise, keep them in subplots.

        For more information on gggrid parameters:
        https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    **geom_kwargs
        Additional parameters for the `geom_histogram` layer.
        For more information on geom_histogram parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_histogram.html

    Returns
    -------
    SupPlotsSpec
        Histograms.

    Examples
    --------
    Histograms.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.histograms(
            data,
            ["n_genes_by_counts", "log1p_total_counts_mt"],
            fill="cell_type_lvl1",
            bins=40,
            ncol=2,
            guides="collect",
        )
    """
    # BUILD: one shared (wide) frame for all keys, instead of rebuilding per key.
    # Axis is resolved once over all keys; mixed-axis key sets raise in `_determine_axis`.
    axis = (
        _determine_axis(data=data, keys=keys, companions=[group_by, fill, color])
        if axis is None
        else axis
    )
    if isinstance(add_keys, str):
        add_keys = [add_keys]
    variable_keys: list[str] = []
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[*keys, group_by, color, fill, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    # Tooltips are shared across subplots; pull their fields into the shared
    # frame so each subplot's unpivot index can keep them.
    _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[variable_column, value_column],
        metadata_columns=metadata_columns,
        axis=axis,
    )
    observation_column_name = None if tooltips == "none" else observations_name
    variable_column_name = None if tooltips == "none" else variables_name
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observation_column_name,
        variables_name=variable_column_name,
        metadata_columns=metadata_columns,
    )

    plots = []
    for i, key in enumerate(keys):
        plot = histogram(
            data=data,
            key=key,
            frame=frame,
            group_by=group_by,
            groups=groups,
            drop=drop,
            mapping=mapping,
            axis=axis,
            color=color,
            fill=fill,
            bins=bins,
            binwidth=binwidth,
            threshold=threshold,
            add_keys=add_keys,
            tooltips=tooltips,
            geom_fill=geom_fill,
            geom_color=geom_color,
            observations_name=observations_name,
            variables_name=variables_name,
            value_column=value_column,
            variable_column=variable_column,
            **geom_kwargs,
        )
        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        if share_ticks:
            plot = _share_ticks(plot, i, keys, ncol)
        if share_axis:
            plot = _share_axis(plot, i, keys, ncol, "axis")

        plots.append(plot)

    dsts = gggrid(
        plots,
        ncol=ncol,  # ty:ignore[invalid-argument-type]
        sharex=sharex,  # ty:ignore[invalid-argument-type]
        sharey=sharey,  # ty:ignore[invalid-argument-type]
        widths=widths,  # ty:ignore[invalid-argument-type]
        heights=heights,  # ty:ignore[invalid-argument-type]
        hspace=hspace,  # ty:ignore[invalid-argument-type]
        vspace=vspace,  # ty:ignore[invalid-argument-type]
        fit=fit,  # ty:ignore[invalid-argument-type]
        align=align,  # ty:ignore[invalid-argument-type]
        guides=guides,
    )

    if interactive:
        dsts += ggtb(size_zoomin=-1)

    return dsts
