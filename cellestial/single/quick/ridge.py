from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_area_ridges,
    gggrid,
    ggplot,
    ggtb,
    scale_fill_hue,
)
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.frames import build_frame
from cellestial.util import (
    _collect_aes_columns,
    _determine_axis,
    _drop_nonfinite_rows,
    _reject_sequence_key,
    _resolve_tooltips,
    _validate_tooltips,
    _warn,
)
from cellestial.util.errors import _unsupported_data_type

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import PlotSpec
    from lets_plot.plot.subplots import SupPlotsSpec
    from polars import DataFrame


def ridge(
    data: AnnData,
    key: str,
    group_by: str,
    *,
    frame: DataFrame | None = None,
    scale: float = 2.0,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    interactive: bool = False,
    **geom_kwargs,
) -> PlotSpec:
    """
    Ridge Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str
        The key to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    group_by : str
        The key to group the ridges by (categorical).
        e.g., 'cell_type' or 'leiden'.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the `key` and `group_by` columns.
    scale : float, default=2.0
        Scaling factor for the height of the ridges.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups, keeping rows where `group_by` matches any
        of them. Categorical grouping columns only.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out rows where `group_by` matches any
        of them. Categorical grouping columns only.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    interactive : bool, default=False
        Whether to make the plot interactive.
    **geom_kwargs
        Additional parameters for the `geom_area_ridges` layer.
        For more information on geom_area_ridges parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_area_ridges.html

    Returns
    -------
    PlotSpec
        Ridge plot.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.

    Examples
    --------

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = cl.datasets.pbmc3k()

        ridge = (
            cl.ridge(
                data,
                key="B2M",
                group_by="cell_type_lvl1",
            )
        )

        ridge

    Customize the geom.

    .. jupyter-execute::
        :emphasize-lines: 6-7

        ridge = (
            cl.ridge(
                data,
                key="B2M",
                group_by="cell_type_lvl1",
                alpha=0.7,
                color="#1f1f1f",
            )
        )

        ridge
    """
    # Handling Data types
    if not isinstance(data, AnnData):
        raise _unsupported_data_type(data, AnnData)

    _reject_sequence_key(key, singular="ridge", plural="ridges")

    # HANDLE: mapping
    mapping = mapping or aes()

    keys = [key]

    # determine index to unpivot
    index = [group_by]
    if add_keys is not None:
        if isinstance(add_keys, str):
            add_keys = [add_keys]
        index.extend(add_keys)

    # DETERMINE: axis if not provided
    axis = _determine_axis(data=data, keys=keys) if axis is None else axis

    # BUILD: the DataFrame (variable_keys is still needed for tooltip resolution)
    variable_keys: list[str] = []
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[*keys, group_by, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    # Resolve tooltips before building so tooltip fields reach `metadata_columns`.
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[group_by, key],
        metadata_columns=metadata_columns,
        axis=axis,
    )
    if frame is None:
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

    # FILTER: keep finite values and apply threshold
    frame = _drop_nonfinite_rows(frame, [key]).filter(pl.col(group_by).is_not_null())
    frame = frame.filter(
        pl.col(key) >= threshold if threshold is not None else True,
    )

    # HANDLE: groups filter (categorical-only) on the grouping column
    if groups is not None:
        if isinstance(groups, str):
            groups = [groups]
        if frame[group_by].dtype == pl.Categorical:
            frame = frame.filter(pl.col(group_by).is_in(list(groups)))
        else:
            msg = f"group_by `{group_by}` is not categorical, `groups` filter ignored"
            _warn(msg)

    # HANDLE: drop filter (categorical-only) on the grouping column
    if drop is not None:
        if isinstance(drop, str):
            drop = [drop]
        if frame[group_by].dtype == pl.Categorical:
            frame = frame.filter(~pl.col(group_by).is_in(list(drop)).fill_null(False))
        else:
            msg = f"group_by `{group_by}` is not categorical, `drop` filter ignored"
            _warn(msg)

    # VALIDATE: tooltips were resolved before the frame build above.
    _validate_tooltips(tooltips, frame)

    # BUILD: the plot
    rdg = ggplot(data=frame)

    rdg += geom_area_ridges(
        mapping=aes(
            x=key,
            y=group_by,
            fill=group_by,
            **mapping.as_dict(),
        ),
        scale=scale,
        tooltips=tooltips,
        **geom_kwargs,
    )

    # handle interactive
    if interactive:
        rdg += ggtb(size_zoomin=-1)

    return rdg + scale_fill_hue()


def ridges(
    data: AnnData,
    keys: Sequence[str],
    group_by: str,
    *,
    scale: float = 2.0,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    interactive: bool = False,
    # multi plot args
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
    Ridge Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    keys : Sequence[str]
        The keys to get the values (numerical).
        e.g., ['total_counts', 'pct_counts_in_top_50_genes'] or a list of gene names.
    group_by : str
        The key to group the ridges by (categorical).
        e.g., 'cell_type' or 'leiden'.
    scale : float, default=2.0
        Scaling factor for the height of the ridges.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    threshold : float | None, default=None
        If provided, filters out rows where the value column is below the threshold.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups, keeping rows where `group_by` matches any
        of them. Categorical grouping columns only.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out rows where `group_by` matches any
        of them. Categorical grouping columns only.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    interactive : bool, default=False
        Whether to make the plot interactive.
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
    **geom_kwargs
        Additional parameters for the `geom_area_ridges` layer.
        For more information on geom_area_ridges parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_area_ridges.html

    Returns
    -------
    SupPlotsSpec
        Ridge plot.

    Examples
    --------

    .. jupyter-execute::

        import numpy as np
        import scanpy as sc

        import cellestial as cl

        data = cl.datasets.pbmc3k()
        # get the top 8 genes with the highest mean expression
        gene_means = np.asarray(data.X.mean(axis=0)).flatten()
        top8_genes = data.var_names[np.argsort(gene_means)[::-1][:8]].tolist()

        cl.ridges(
            data,
            keys=top8_genes,
            group_by="cell_type_lvl1",
            alpha=0.7,
            scale=2,
            ncol=2,
            color="#3f3f3f",
            guides="collect",
        )
    """
    # BUILD: one shared frame for all keys, instead of rebuilding per key.
    # Axis is resolved once over all keys; mixed-axis key sets raise in `_determine_axis`.
    axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    if isinstance(add_keys, str):
        add_keys = [add_keys]
    variable_keys: list[str] = []
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[*keys, group_by, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    # Tooltips are shared across subplots; pull their fields into the shared
    # frame so each subplot's tooltip validation can find them.
    _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[group_by, *keys],
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
    for key in keys:
        plot = ridge(
            data=data,
            key=key,
            group_by=group_by,
            groups=groups,
            drop=drop,
            frame=frame,
            scale=scale,
            mapping=mapping,
            axis=axis,
            threshold=threshold,
            add_keys=add_keys,
            tooltips=tooltips,
            observations_name=observations_name,
            variables_name=variables_name,
            interactive=interactive,
            **geom_kwargs,
        )
        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer

        plots.append(plot)

    rdgs = gggrid(
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
        rdgs += ggtb(size_zoomin=-1)

    return rdgs
