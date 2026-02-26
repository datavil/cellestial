from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from lets_plot import gggrid
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.single.core.distribution import boxplot, violin
from cellestial.util import _share_axis, _share_ticks

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.subplots import SupPlotsSpec


def violins(
    data: AnnData,
    keys: Sequence[str],
    *,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str = "#FF00FF",
    geom_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
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
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: Literal['none'] | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
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
    observations_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    share_ticks : bool, default=True
        Whether to share the labels across all plots.
        If True, only X tick texts on bottom row and Y tick text on left column are shown.
    share_axis : bool, default=False
        Whether to share the axis across all plots.
        If True, only X axis on bottom row and Y axis on left column is shown.
    layers : Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None, default=None
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
    for i,key in enumerate(keys):
        plot = violin(
            data=data,
            key=key,
            mapping=mapping,
            axis=axis,
            color=color,
            fill=fill,
            add_keys=add_keys,
            tooltips=tooltips,
            geom_fill=geom_fill,
            geom_color=geom_color,
            point_color=point_color,
            point_alpha=point_alpha,
            point_size=point_size,
            point_geom=point_geom,
            observations_name=observations_name,
            variables_name=variables_name,
            show_points=show_points,
            interactive=interactive,
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

    return gggrid(
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

def boxplots(
    data: AnnData,
    keys: Sequence[str],
    *,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str = "#FF00FF",
    geom_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
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
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: Literal['none'] | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
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
    observations_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default="Gene"
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    share_ticks : bool, default=True
        Whether to share the labels across all plots.
        If True, only X tick texts on bottom row and Y tick text on left column are shown.
    share_axis : bool, default=False
        Whether to share the axis across all plots.
        If True, only X axis on bottom row and Y axis on left column is shown.
    layers : Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None, default=None
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
    for i,key in enumerate(keys):
        plot = boxplot(
            data=data,
            key=key,
            mapping=mapping,
            axis=axis,
            color=color,
            fill=fill,
            add_keys=add_keys,
            tooltips=tooltips,
            geom_fill=geom_fill,
            geom_color=geom_color,
            point_color=point_color,
            point_alpha=point_alpha,
            point_size=point_size,
            point_geom=point_geom,
            observations_name=observations_name,
            variables_name=variables_name,
            show_points=show_points,
            interactive=interactive,
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

    return gggrid(
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

