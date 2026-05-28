from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    element_blank,
    geom_path,
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
from cellestial.single.heatmap.utilities import (
    _assign_positions,
    _bin_within_groups,
    _get_group_bar_frame,
    _get_group_lines_frame,
    _key_groups_bar_y,
    _key_groups_layers,
    _resolve_key_groups,
    _resolve_padding,
    _resolve_rank_genes_groups_args,
    _scale_values,
)
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import (
    _fill_gradient,
    _get_dendrogram,
    _get_dendrogram_path_frame,
    _warn,
)
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lets_plot.plot.core import FeatureSpec, PlotSpec


def heatmap(
    data: AnnData,
    keys: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    group_by: str | None = None,
    *,
    markers: bool | str = False,
    n_genes: int = 5,
    groups: Sequence[str] | None = None,
    mapping: FeatureSpec | None = None,
    geom: Literal["raster", "tile"] = "raster",
    scale_axis: Literal[0, 1] | None = None,
    dendrogram: bool = False,
    aggregate: bool = False,
    group_bars: bool = True,
    group_bars_size: float = 6,
    group_bars_labels: bool = True,
    group_lines: bool = True,
    group_lines_color: str = "white",
    group_lines_size: float = 0.6,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_key: str | None = None,
    group_lines_kwargs: dict | None = None,
    dendrogram_kwargs: dict | None = None,
    key_labels: bool = True,
    key_labels_text_size: float = 1.0,
    key_labels_bracket_size: float = 0.6,
    key_labels_text_color: str = "black",
    key_labels_bracket_color: str = "black",
    key_labels_width: float = 0.6,
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
    max_rows: int | None = 1000,
    **geom_kwargs,
) -> PlotSpec:
    """
    Heatmap.

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
    geom : {'raster', 'tile'}, default='raster'
        The geom to use,. Use 'raster' for performance.
        Use 'tile' to enable tooltips.
    dendrogram : bool, default=False
        Whether to add a dendrogram for the `group_by` axis.
        Uses `scanpy.tl.dendrogram` if not already computed.
    aggregate : bool, default=False
        If False, plot one row per observation (i.e., cell).
        If True, aggregate values per group by mean so each row is a group.
        For the aggregated view, prefer :func:`matrixplot`.
    group_bars : bool, default=True
        Whether to draw colored vertical bars on the left marking group membership.
        Only used when `aggregate=False`.
    group_bars_size : float, default=6
        Size (thickness) of the group color bars.
    group_bars_labels : bool, default=True
        Whether to show group names as labels along the y-axis.
        Removes the related legend.
        Only applies when `group_bars=True` and `aggregate=False`.
    group_lines : bool, default=True
        Whether to draw horizontal lines within the heatmap separating groups.
    group_lines_color : str, default='white'
        Color of the group separator lines.
    group_lines_size : float, default=0.6
        Size (thickness) of the group separator lines.
    dendrogram_color : str, default='black'
        Color of the dendrogram segments.
    dendrogram_size : float, default=0.5
        Size (thickness) of the dendrogram segments.
    dendrogram_key : str | None, default=None
        Specific key holding the precomputed dendrogram.
        By default, `dendrogram_{group_by}` is used.
    group_lines_kwargs : dict | None, default=None
        Additional parameters to pass to the group separator lines geom_segment.
    dendrogram_kwargs : dict | None, default=None
        Additional parameters to pass to the dendrogram geom_segment.
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
    scale_axis : {0, 1} | None, default=None
        Whether to standardize a dimension between 0 and 1.
        Uses min-max scaling; constant partitions are set to 0.
        If 0, standardize each variable (column).
        If 1, standardize each row (group when `aggregate` is True, observation otherwise).
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
    max_rows : int | None, default=1000
        When `aggregate=False`, cap the number of plotted rows by averaging
        contiguous observations within each group into virtual bins.
        Bypasses the long-form data payload sent to the renderer when row
        counts exceed display resolution. Set to `None` to disable.
    **geom_kwargs
        Additional parameters for the heatmap geom layer.

    Returns
    -------
    PlotSpec
        Heatmap.

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
    Heatmap plots one row per observation (cell), grouped by `group_by`.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            dendrogram=True,
        ) + scale_fill_viridis()

    To enable tooltips, use `geom='tile'`.

    .. jupyter-execute::

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            dendrogram=True,
            geom="tile",
            tooltips=["value"],
        ) + scale_fill_viridis()

    Values can be standardized per-row or per-column with `scale_axis`.

    .. jupyter-execute::

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            dendrogram=True,
            scale_axis=1,
        ) + scale_fill_viridis()

    Heatmap components (group separator lines, group bars, dendrogram) can be added or customized.

    .. jupyter-execute::
        :emphasize-lines: 6-12

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            geom="raster",
            group_lines_size=0.5,
            group_lines_color="white",
            dendrogram=True,
            dendrogram_size="0.7",
            group_bars_labels=True,
            group_bars=True,
        ) + scale_fill_viridis()

    For the aggregated view (one row per group), use :func:`matrixplot`
    or pass `aggregate=True`.

    .. jupyter-execute::
        :emphasize-lines: 5

        cl.heatmap(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            aggregate=True,
            dendrogram=True,
        )

    To plot the top genes from a precomputed ranking, set
    `markers` and `n_genes` (`keys` and `group_by` are
    inferred):

    .. jupyter-execute::

        import scanpy as sc

        sc.tl.rank_genes_groups(data, groupby="cell_type_lvl1")
        cl.heatmap(data, markers=True, n_genes=5, aggregate=True)

    """
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

    if "tooltips" in geom_kwargs and geom == "raster":
        _warn(
            "\nWarning: tooltips are not supported for 'raster' geom and will be ignored."
            "\nUse 'tile' geom to enable tooltips."
        )
        geom_kwargs.pop("tooltips")

    # RESOLVE: dict ``keys`` into a flat list while preserving mapping order
    keys, key_groups = _resolve_key_groups(keys, key_labels=key_labels)

    # BUILD: long-form dataframe.
    # `group_by` (and the cell identifier, when not aggregating) are needed from
    # the observation metadata. On the variable axis `keys` are metadata columns
    # rather than genes pulled from X, so they must be materialised explicitly.
    observation_column_name = None if aggregate else observations_name
    metadata_columns = [group_by] if group_by is not None else []
    if axis == 1:
        metadata_columns = list(dict.fromkeys([*keys, *metadata_columns]))
    frame = build_frame(
        data=data,
        variable_keys=keys,
        axis=axis,
        observations_name=observation_column_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
        metadata_columns=metadata_columns,
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
        y_order_groups, paths = _get_dendrogram(data, group_by, use_key=dendrogram_key)
    else:
        y_order_groups = (
            frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
        )
        paths = None

    # BIN: collapse cells into at most ``max_rows`` virtual rows (within group)
    if not aggregate and max_rows is not None:
        frame = _bin_within_groups(
            frame,
            observations_name=observations_name,
            group_by=group_by,
            variable_column=variable_column,
            value_column=value_column,
            y_order_groups=y_order_groups,
            max_rows=max_rows,
        )

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
    aes_main = aes(x="position_x", y="position_y", fill=value_column, **mapping.as_dict())
    geom_layer = (
        geom_raster(aes_main, **geom_kwargs)
        if geom == "raster"
        else geom_tile(aes_main, **geom_kwargs)
    )
    htmp = ggplot(frame) + geom_layer + _THEME_HEATMAP

    # X scale: variable labels
    htmp += scale_x_continuous(breaks=list(range(n_x)), labels=x_keys)

    # Compute top edge of data and the y limit, extending the latter when
    # key-group brackets are drawn so rotated labels are not clipped.
    if aggregate:
        data_top = n_y - 0.5
    else:
        # ``position_y`` ranges from 0 to ``n_x - 1``; cells extend half a row above.
        half_step = (n_x - 1) / max(n_y - 1, 1) / 2
        data_top = (n_x - 1) + half_step
    y_max_limit = data_top
    key_groups_total_span: float | None = None
    if key_labels and key_groups is not None:
        data_range = data_top + 0.5
        key_groups_padding = _resolve_padding(
            key_groups,
            padding=None,
            data_range=data_range,
            scale=1.0,
            label_size_scale=key_labels_text_size,
        )
        key_groups_total_span = data_range + key_groups_padding
        y_max_limit = data_top + key_groups_padding

    # Y scale: groups for aggregate, group-center labels when titling bars, else hidden
    if aggregate:
        htmp += scale_y_continuous(
            breaks=list(range(n_y)),
            labels=y_order_groups,
            limits=[-0.5, y_max_limit],
            expand=[0, 0],
        )
    elif group_bars and group_bars_labels:
        htmp += scale_y_continuous(
            breaks=group_centers,
            labels=y_order_groups,
            limits=[-0.5, y_max_limit],
            expand=[0, 0],
        )
        htmp += guides(color="none")
    else:
        htmp += scale_y_continuous(limits=[-0.5, y_max_limit], expand=[0, 0])
        htmp += theme(axis_text_y=element_blank(), axis_ticks_y=element_blank())

    # GROUP color bar on left for non-aggregate
    if not aggregate and group_bars:
        assert cell_frame is not None
        group_bar_frame = _get_group_bar_frame(cell_frame, group_by=group_by, n_x=n_x)
        htmp += geom_segment(
            data=group_bar_frame,
            mapping=aes(x="x", xend="x", y="y_min", yend="y_max", color=group_by),
            size=group_bars_size,
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
            **(group_lines_kwargs or {}),
        )

    # DENDROGRAM (right side, along y-axis)
    if dendrogram:
        assert paths is not None
        dendro_frame = _get_dendrogram_path_frame(
            paths, n_x=n_x, n_groups=len(y_order_groups), group_centers=group_centers
        )
        htmp += geom_path(
            data=dendro_frame,
            mapping=aes(x="x", y="y", group="group"),
            color=dendrogram_color,
            size=dendrogram_size,
            **(dendrogram_kwargs or {}),
        )

    # FILL gradient
    htmp += _fill_gradient(
        frame[value_column],
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,
    )

    # KEY-GROUP brackets above the plot when ``keys`` was a mapping.
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
        ):
            htmp += layer

    if interactive:
        htmp += ggtb(size_zoomin=-1)

    return htmp


def matrixplot(
    data: AnnData,
    keys: Sequence[str] | Mapping[str, Sequence[str]] | None = None,
    group_by: str | None = None,
    *,
    markers: bool | str = False,
    n_genes: int = 5,
    groups: Sequence[str] | None = None,
    mapping: FeatureSpec | None = None,
    geom: Literal["raster", "tile"] = "raster",
    scale_axis: Literal[0, 1] | None = None,
    dendrogram: bool = False,
    group_lines: bool = True,
    group_lines_color: str = "white",
    group_lines_size: float = 0.6,
    dendrogram_color: str = "black",
    dendrogram_size: float = 0.5,
    dendrogram_key: str | None = None,
    group_lines_kwargs: dict | None = None,
    dendrogram_kwargs: dict | None = None,
    key_labels: bool = True,
    key_labels_text_size: float = 1.0,
    key_labels_bracket_size: float = 0.6,
    key_labels_text_color: str = "black",
    key_labels_bracket_color: str = "black",
    key_labels_width: float = 0.6,
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
    Matrix plot.

    Parameters
    ----------
    data : AnnData
        The single-cell data object.
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
    geom : {'raster', 'tile'}, default='raster'
        The geom to use. Use 'raster' for performance.
        Use 'tile' to enable tooltips.
    scale_axis : {0, 1} | None, default=None
        Whether to standardize a dimension between 0 and 1.
        Uses min-max scaling; constant partitions are set to 0.
        If 0, standardize each variable (column).
        If 1, standardize each group (row).
    dendrogram : bool, default=False
        Whether to add a dendrogram for the `group_by` axis.
        Uses the precomputed dendrogram, computing one if not already present.
    group_lines : bool, default=True
        Whether to draw horizontal lines within the plot separating groups.
    group_lines_color : str, default='white'
        Color of the group separator lines.
    group_lines_size : float, default=0.6
        Size (thickness) of the group separator lines.
    dendrogram_color : str, default='black'
        Color of the dendrogram segments.
    dendrogram_size : float, default=0.5
        Size (thickness) of the dendrogram segments.
    dendrogram_key : str | None, default=None
        Specific key holding the precomputed dendrogram.
        By default, `dendrogram_{group_by}` is used.
    group_lines_kwargs : dict | None, default=None
        Additional parameters to pass to the group separator lines geom_segment.
    dendrogram_kwargs : dict | None, default=None
        Additional parameters to pass to the dendrogram geom_segment.
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
        Matrix plot.

    Notes
    -----
    Equivalent to `heatmap` with `aggregate=True`.

    Examples
    --------
    Matrix plot of marker expression aggregated per cell type.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        markers = ["C1QA", "PSAP", "CD79A", "CD79B", "CST3", "LYZ"]

        cl.matrixplot(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            dendrogram=True,
        )

    Standardize per-variable so each marker spans 0-1 across groups.

    .. jupyter-execute::

        cl.matrixplot(
            data,
            group_by="cell_type_lvl1",
            keys=markers,
            scale_axis=0,
        )

    Plot the top genes from a precomputed ranking:

    .. jupyter-execute::

        sc.tl.rank_genes_groups(data, groupby="cell_type_lvl1")
        cl.matrixplot(data, markers=True, n_genes=5)
    """
    return heatmap(
        data,
        keys=keys,
        group_by=group_by,
        markers=markers,
        n_genes=n_genes,
        groups=groups,
        mapping=mapping,
        geom=geom,
        scale_axis=scale_axis,
        dendrogram=dendrogram,
        key_labels=key_labels,
        key_labels_text_size=key_labels_text_size,
        key_labels_bracket_size=key_labels_bracket_size,
        key_labels_text_color=key_labels_text_color,
        key_labels_bracket_color=key_labels_bracket_color,
        key_labels_width=key_labels_width,
        aggregate=True,
        group_lines=group_lines,
        group_lines_color=group_lines_color,
        group_lines_size=group_lines_size,
        dendrogram_color=dendrogram_color,
        dendrogram_size=dendrogram_size,
        dendrogram_key=dendrogram_key,
        group_lines_kwargs=group_lines_kwargs,
        dendrogram_kwargs=dendrogram_kwargs,
        value_column=value_column,
        variable_column=variable_column,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
        interactive=interactive,
        **geom_kwargs,
    )
