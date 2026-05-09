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
    _determine_axis,
    _resolve_tooltips,
    _select_variable_keys,
    _validate_tooltips,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import PlotSpec
    from lets_plot.plot.subplots import SupPlotsSpec

def ridge(
    data: AnnData,
    key: str,
    *,
    group_by: str | None = None,
    scale: float = 2.0,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
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

    Examples
    --------

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad('data/pbmc3k_pped.h5ad')

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
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

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

    # BUILD: the DataFrame
    variable_keys = _select_variable_keys(data=data, keys=keys)
    variable_keys.extend(_select_variable_keys(data=data, keys=add_keys))
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
    )

    # FILTER: drop nulls and apply threshold
    frame = frame.drop_nulls(subset=[key])
    frame = frame.filter(
        pl.col(key) >= threshold if threshold is not None else True,
    )

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[group_by, key],
    )
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
    *,
    group_by: str | None = None,
    scale: float = 2.0,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    threshold: float | None = None,
    add_keys: Sequence[str] | str | None = None,
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
    widths: list | None = None,
    heights: list | None = None,
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
    fit : bool, default=True
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool, default=False
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

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")
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
    plots = []
    for key in keys:
        plot = ridge(
            data=data,
            key=key,
            group_by=group_by,
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
        ncol=ncol, # ty:ignore[invalid-argument-type]
        sharex=sharex, # ty:ignore[invalid-argument-type]
        sharey=sharey, # ty:ignore[invalid-argument-type]
        widths=widths, # ty:ignore[invalid-argument-type]
        heights=heights, # ty:ignore[invalid-argument-type]
        hspace=hspace, # ty:ignore[invalid-argument-type]
        vspace=vspace, # ty:ignore[invalid-argument-type]
        fit=fit, # ty:ignore[invalid-argument-type]
        align=align, # ty:ignore[invalid-argument-type]
        guides=guides,
    )

    if interactive:
        rdgs += ggtb(size_zoomin=-1)

    return rdgs
