from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

# Data retrieval
from anndata import AnnData
from lets_plot import gggrid

from cellestial.single.basic.scatter import scatter
from cellestial.util.errors import ConfilictingLengthError

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpec, LayerSpec
    from lets_plot.plot.subplots import SupPlotsSpec


def scatters(
    data: AnnData,
    x: str | Sequence[str],
    y: str | Sequence[str],
    *,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    size: str | None = None,
    shape: str | None = None,
    point_color: str | None = None,
    point_fill: str | None = None,
    point_size: str | None = None,
    point_shape: str | None = None,
    interactive: bool = False,
    barcode_name: str = "Barcode",
    variable_name: str = "Varible",
    include_dimensions: bool = False,
    show_tooltips: bool = True,
    add_tooltips: Sequence[str] | str | None = None,
    custom_tooltips: Sequence[str] | str | None = None,
    tooltips_title: str | None = None,
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
    **point_kwargs,
) -> SupPlotsSpec:
    """
    Scatter Plots.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    x : str
        The key(s) for the x-axis.
    y : str
        The key(s) for the y-axis.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic for the geom_point.
    fill : str | None, default=None
        Fill aesthetic for the geom_point.
    size : str | None, default=None
        Size aesthetic for the geom_point.
    shape : str | None, default=None
        Shape aesthetic for the geom_point.
    point_color : str | None, default=None
        Color for all the points.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_fill : str | None, default=None
        Fill color for all the points.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_size : str | None, default=None
        Size for all the points.
    point_shape : str | None, default=None
        Shape of all the points, an integer from 0 to 25.
        For more information see:
        https://lets-plot.org/python/pages/aesthetics.html#point-shapes
    interactive : bool, default=False
        Whether to make the plot interactive.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variable_name : str, default="Variable"
        The name to give to variable index column in the dataframe.
    include_dimensions : bool, default=False
        Whether to include dimensions in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    add_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Custom tooltips, will overwrite the base_tooltips.
    tooltips_title : str | None, default=None
        Title for the tooltips.
    layers : Sequence[FeatureSpec|LayerSpec] | FeatureSpec | LayerSpec | None, default=None,
        Layers to add to all the plots in the grid.
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

    For more information on gggrid parameters:
    https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    **point_kwargs : dict[str, Any]
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Scatter plot.
    """
    # str to list for x and y
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]

    # check for broadcasting
    if len(x) != len(y):
        if len(x) == 1:
            x = x * len(y)
        elif len(y) == 1:
            y = y * len(x)
        else:
            msg = f"Length of x ({len(x)}) and y ({len(y)}) must be the same, or one of them must be of length 1."
            raise ConfilictingLengthError(msg)

    # build plots
    plots = []
    for xi, yi in zip(x, y):
        scttr = scatter(
            data,
            x=xi,
            y=yi,
            axis=axis,
            color=color,
            fill=fill,
            size=size,
            shape=shape,
            point_color=point_color,
            point_fill=point_fill,
            point_size=point_size,
            point_shape=point_shape,
            interactive=interactive,
            barcode_name=barcode_name,
            variable_name=variable_name,
            include_dimensions=include_dimensions,
            show_tooltips=show_tooltips,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            tooltips_title=tooltips_title,
            **point_kwargs,
        )
        # handle the layers
        if layers is not None:
            if not isinstance(layers, Sequence):
                layers = [layers]
            for layer in list(layers):
                scttr += layer
        plots.append(scttr)

    scttrs = gggrid(
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
    )

    return scttrs
