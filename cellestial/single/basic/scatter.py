from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Literal

# Data retrieval
from anndata import AnnData
from lets_plot import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    guide_legend,
    guides,
    labs,
    layer_tooltips,
)
from lets_plot.plot.core import PlotSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_SCATTER
from cellestial.util import (
    _decide_tooltips,
    _determine_axis,
    _select_variable_keys,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import PlotSpec


def scatter(
    data: AnnData,
    x: str,
    y: str,
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
    **point_kwargs,
) -> PlotSpec:
    """
    Scatter Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    x : str
        The key for the x-axis.
    y : str
        The key for the y-axis.
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
    **point_kwargs : dict[str, Any]
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Scatter plot.

    """
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # handle point_kwargs
    if point_kwargs is None:
        point_kwargs = {}
    else:
        if "tooltips" in point_kwargs:
            msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
            raise KeyError(msg)

    # Handle tooltips
    base_tooltips = [x, y]
    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )

    # dimensional_keys = [key for key in [x, y] if key.startswith("X_")]
    keys = [
        key
        for key in [x, y, color, fill, size, shape]
        if key is not None and not key.startswith("X_")
    ]
    variable_keys = _select_variable_keys(data=data, keys=keys)

    # BUILD: the dataframe
    axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=barcode_name,
        variables_name=variable_name,
        include_dimensions=include_dimensions,
    )

    # scatter kwargs
    point_kwargs["size"] = point_size
    point_kwargs["color"] = point_color
    point_kwargs["fill"] = point_fill
    point_kwargs["shape"] = point_shape

    # BUILD: the scatterplot
    scttr = (
        ggplot(data=frame)
        + geom_point(
            aes(x=x, y=y, color=color, size=size, shape=shape, fill=fill),
            tooltips=layer_tooltips(tooltips),
            **point_kwargs,
        )
        + labs(x=x, y=y)
        + _THEME_SCATTER
    )
    # handle legend wrapping (Deprecated as of lets_plot=4.8.0)
    # if color is not None:
    #     n_distinct = frame.select(color).unique().height
    #     if n_distinct > 10:
    #         ncol = ceil(n_distinct / 10)
    #         scttr += guides(color=guide_legend(ncol=ncol))
    # if fill is not None:
    #     n_distinct = frame.select(fill).unique().height
    #     if n_distinct > 10:
    #         ncol = ceil(n_distinct / 10)
    #         scttr += guides(fill=guide_legend(ncol=ncol))
    # handle interactive
    if interactive:
        scttr += ggtb(size_zoomin=-1)

    return scttr
