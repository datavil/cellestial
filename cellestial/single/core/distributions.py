from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from anndata import AnnData
from lets_plot import gggrid
from lets_plot.plot.core import FeatureSpec, LayerSpec
from lets_plot.plot.subplots import SupPlotsSpec

from cellestial.single.core.distribution import boxplot, violin


def violins(
    data: AnnData,
    keys: list[str] | tuple[str] | Sequence[str],
    *,
    axis: Literal[0,1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    geom_fill: str = "#FF00FF",
    geom_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom : Literal["jitter","point","sina"] = "jitter",
    barcode_name: str = "Barcode",
    variable_name: str = "Variable",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: Sequence[str] | str | None = None,
    custom_tooltips: Sequence[str] | str | None = None,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    layers: list | tuple | Sequence | FeatureSpec | LayerSpec | None = None,
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
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    geom_fill : str, default="#FF00FF"
        Fill color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str, default="#2f2f2f"
        Border color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default="#1f1f1f"
        Color for the points in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the violin plot.
    point_size : float, default=0.5
        Size for the points in the violin plot.
    point_geom : Literal["jitter","point","sina"], default is "jitter",
        Geom type of the points, default is geom_jitter.
    trim : bool, default=False
        Whether to trim the violin plot.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variable_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : Sequence[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : Sequence[str] | str | None, default=None
        Custom tooltips to show.
    interactive : bool, default=False
        Whether to make the plot interactive.
    layers : list | tuple | Sequence | FeatureSpec | LayerSpec | None, default=None
        Additional layers to add to the plot.
    variable_column : str, default="variable"
        The name of the variable column in the dataframe.
    value_column : str, default="value"
        The name of the value column in the dataframe.
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
    hspace : float | None = None
        Cell horizontal spacing in px.
    vspace : float | None = None
        Cell vertical spacing in px.
    fit : bool, default=True
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool, default=False
        If True, align inner areas (i.e. “geom” bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot “inner areas” layouting.
    guides : str, default="auto"
        Specifies how guides (legends and colorbars) should be treated in the layout.
        - 'collect'- collect guides from all subplots, removing duplicates.
        - 'keep' - keep guides in their original subplots; do not collect at this level.
        - 'auto' - allow guides to be collected if an upper-level layout uses guides='collect';
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
    """
    plots = []
    for key in keys:
        dst = violin(
            data=data,
            key=key,
            axis=axis,
            color=color,
            fill=fill,
            geom_fill=geom_fill,
            geom_color=geom_color,
            point_color=point_color,
            point_alpha=point_alpha,
            point_size=point_size,
            point_geom=point_geom,
            barcode_name=barcode_name,
            variable_name=variable_name,
            show_tooltips=show_tooltips,
            show_points=show_points,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            interactive=interactive,
            value_column=value_column,
            variable_column=variable_column,
            point_kwargs=point_kwargs,
            **geom_kwargs,
        )
        # handle the layers
        if layers is not None:
            if not isinstance(layers, Sequence):
                layers = [layers]
            for layer in list(layers):
                dst += layer

        plots.append(dst)

    dsts = gggrid(
        plots,
        ncol=ncol,
        sharex=sharex,
        sharey=sharey,
        widths=widths,
        heights=heights,
        hspace=hspace,
        vspace=vspace,
        fit=fit,
        align=align,
        guides=guides,
    )

    return dsts


def boxplots(
    data: AnnData,
    keys: list[str] | tuple[str] | Sequence[str],
    *,
    axis: Literal[0,1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    geom_fill: str = "#FF00FF",
    geom_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom : Literal["jitter","point","sina"] = "jitter",
    barcode_name: str = "Barcode",
    variable_name: str = "Variable",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: Sequence[str] | str | None = None,
    custom_tooltips: Sequence[str] | str | None = None,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    layers: list | tuple | Sequence | FeatureSpec | LayerSpec | None = None,
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
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    geom_fill : str, default="#FF00FF"
        Fill color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str, default="#2f2f2f"
        Border color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default="#1f1f1f"
        Color for the points in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the boxplot.
    point_size : float, default=0.5
        Size for the points in the boxplot.
    point_geom : Literal["jitter","point","sina"], default is "jitter",
        Geom type of the points, default is geom_jitter.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variable_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : Sequence[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : Sequence[str] | str | None, default=None
        Custom tooltips to show.
    interactive : bool, default=False
        Whether to make the plot interactive.
    layers : list | tuple | Sequence | FeatureSpec | LayerSpec | None, default=None
        Additional layers to add to the plot.
    variable_column : str, default="variable"
        The name of the variable column in the dataframe.
    value_column : str, default="value"
        The name of the value column in the dataframe.
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
    hspace : float | None = None
        Cell horizontal spacing in px.
    vspace : float | None = None
        Cell vertical spacing in px.
    fit : bool, default=True
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool, default=False
        If True, align inner areas (i.e. “geom” bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot “inner areas” layouting.
    guides : str, default="auto"
        Specifies how guides (legends and colorbars) should be treated in the layout.
        - 'collect'- collect guides from all subplots, removing duplicates.
        - 'keep' - keep guides in their original subplots; do not collect at this level.
        - 'auto' - allow guides to be collected if an upper-level layout uses guides='collect';
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
    SupPlotsSpec | PlotSpec
        Boxplots.
    """
    plots = []
    for key in keys:
        dst = boxplot(
            data=data,
            key=key,
            axis=axis,
            color=color,
            fill=fill,
            geom_fill=geom_fill,
            geom_color=geom_color,
            point_color=point_color,
            point_alpha=point_alpha,
            point_size=point_size,
            point_geom=point_geom,
            barcode_name=barcode_name,
            variable_name=variable_name,
            show_tooltips=show_tooltips,
            show_points=show_points,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            interactive=interactive,
            value_column=value_column,
            variable_column=variable_column,
            point_kwargs=point_kwargs,
            **geom_kwargs,
        )
        # handle the layers
        if layers is not None:
            if not isinstance(layers, Sequence):
                layers = [layers]
            for layer in list(layers):
                dst += layer

        plots.append(dst)

    dsts = gggrid(
        plots,
        ncol=ncol,
        sharex=sharex,
        sharey=sharey,
        widths=widths,
        heights=heights,
        hspace=hspace,
        vspace=vspace,
        fit=fit,
        align=align,
        guides=guides,
    )

    return dsts
