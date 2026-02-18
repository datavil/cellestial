from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from anndata import AnnData
from lets_plot import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    labs,
    layer_tooltips,
)
from lets_plot.plot.core import FeatureSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_SCATTER
from cellestial.util import (
    _determine_axis,
    _select_variable_keys,
)

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def xyplot(
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
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool | int = False,
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
    tooltips: Literal['none'] | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observations_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default="Variable"
        The name to give to variable index column in the dataframe.
    include_dimensions : bool | int, default=False
        Whether to include dimensions in the DataFrame.
        Providing an integer will limit the number of dimensions to given number.
    **point_kwargs
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

    # Determine variable keys
    keys = [
        key
        for key in [x, y, color, fill, size, shape]
        if key is not None and not key.startswith("X_")
    ]
    variable_keys = _select_variable_keys(data=data, keys=keys)

    # HANDLE: tooltips
    if tooltips is None:
        tooltips = [x, y]
        tooltips_spec = layer_tooltips(tooltips)
    elif tooltips == "none" or isinstance(tooltips, str):
        tooltips_spec = tooltips
    elif isinstance(tooltips, Sequence):
        tooltips = list(tooltips)
        tooltips_spec = layer_tooltips(tooltips)
        tooltips_variables = _select_variable_keys(data, tooltips)
        if not set(tooltips_variables).issubset(variable_keys):
            variable_keys.extend(tooltips_variables)
    elif isinstance(tooltips, FeatureSpec):
        tooltips_spec = tooltips

    # BUILD: the dataframe
    axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
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
            tooltips=tooltips_spec,
            **point_kwargs,
        )
        + labs(x=x, y=y)
        + _THEME_SCATTER
    )
    if interactive:
        scttr += ggtb(size_zoomin=-1)

    return scttr
