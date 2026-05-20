from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from lets_plot import (
    aes,
    arrow,
    element_blank,
    element_text,
    geom_blank,
    geom_path,
    geom_segment,
    theme,
)

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import get_mapping
from cellestial.util.errors import MissingAestheticError

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpecArray, PlotSpec
    from polars import DataFrame

# fraction of each axis arm drawn as the arrowhead-bearing tip segment;
# lets_plot caps the arrowhead to ~half the segment length, so the tip must be
# long enough to render a full-size head
_TIP_FRACTION = 0.3


def _axis_arrow_layers(
    frame: DataFrame,
    *,
    x: str,
    y: str,
    length: float,
    size: float,
    color: str,
    angle: float,
    arrow_kwargs: dict | None = None,
) -> FeatureSpecArray:
    """
    Build the arrow-axis as an L-shaped path plus two arrowhead tip segments.

    The semi-square L is one continuous `geom_path` that stops short of each
    open end so the corner joins cleanly with no overlap. Each tip is completed
    by a `geom_segment` carrying the arrowhead, meeting the path end-to-end.

    Parameters
    ----------
    frame : `polars.DataFrame`
        DataFrame holding the `x` and `y` columns.
    x : str
        Name of the x axis column.
    y : str
        Name of the y axis column.
    length : float
        Length of each axis arm as a fraction of its data range.
    size : float
        Line width of the path and tip segments.
    color : str
        Color of the path and tip segments.
    angle : float
        Angle of the arrowhead in degrees.
    arrow_kwargs : dict | None
        Additional parameters for the `arrow` layer.

    Returns
    -------
    `FeatureSpecArray`
        The path and tip-segment layers.
    """
    arrow_kwargs = arrow_kwargs or {}

    x_min = frame[x].min()
    x_max = frame[x].max()
    y_min = frame[y].min()
    y_max = frame[y].max()

    # find total difference between the max and min for both axis
    x_diff = x_max - x_min
    y_diff = y_max - y_min

    # find the ends of the arrows
    xend = x_min + length * x_diff
    yend = y_min + length * y_diff

    # adjust bottom ends of arrows
    adjust_rate = 0.025
    x0 = x_min - x_diff * adjust_rate
    y0 = y_min - y_diff * adjust_rate

    # tip segments carry the arrowheads; the path stops short of each tip
    tip_x = (xend - x0) * _TIP_FRACTION
    tip_y = (yend - y0) * _TIP_FRACTION

    # semi-square L (stopping short of both tips), drawn as one connected path
    path_frame = pl.DataFrame(
        {"x": [xend - tip_x, x0, x0], "y": [y0, y0, yend - tip_y]}
    )

    layers = geom_path(
        data=path_frame,
        mapping=aes(x="x", y="y"),
        color=color,
        size=size,
    )
    # X axis tip
    layers += geom_segment(
        x=xend - tip_x,
        y=y0,
        xend=xend,
        yend=y0,
        color=color,
        size=size,
        arrow=arrow(angle, **arrow_kwargs),
    )
    # Y axis tip
    layers += geom_segment(
        x=x0,
        y=yend - tip_y,
        xend=x0,
        yend=yend,
        color=color,
        size=size,
        arrow=arrow(angle, **arrow_kwargs),
    )
    return layers


def _modify_axis(
    frame: DataFrame,
    *,
    x: str,
    y: str,
    axis_type: Literal["axis", "arrow"] | None,
    arrow_size: float,
    arrow_color: str,
    arrow_angle: float,
    arrow_length: float,
):
    """
    Adds arrows as the X and Y axis to the plot.

    Parameters
    ----------
    frame : `polars.DataFrame`
        DataFrame copied from the single cell data.
    x : str
        Name of the x axis column.
    y : str
        Name of the y axis column.
    axis_type : {'axis', 'arrow'} | None
        Whether to use regular axis or arrows as the axis.
    arrow_size : float
        Width of the axis lines.
    arrow_color : str
        Color of the arrow.
    arrow_angle : float
        Angle of the arrow head in degrees.
    arrow_length : float
        Span of each axis line as a fraction of its data range (0.25 covers 25%).

    Returns
    -------
    `FeatureSpec` or `FeatureSpecArray`
        Theme feature specification.

    for more information on the arrow parameters, see:
    https://lets-plot.org/python/pages/api/lets_plot.arrow.html
    """
    if axis_type is None:
        return theme(
            # remove axis elements
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line=element_blank(),
        )

    elif axis_type == "axis":
        return geom_blank()

    elif axis_type == "arrow":
        new_layer = theme(
            # remove axis elements
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line=element_blank(),
            # position axis titles according to arrow size
            axis_title_x=element_text(hjust=arrow_length / 2.5),  # better than 2
            axis_title_y=element_text(hjust=arrow_length / 2.5),
        )
        new_layer += _axis_arrow_layers(
            frame,
            x=x,
            y=y,
            length=arrow_length,
            size=arrow_size,
            color=arrow_color,
            angle=arrow_angle,
        )
    else:
        msg = f"expected 'axis' or 'arrow' for 'axis_type' argument, but received {axis_type}"
        raise ValueError(msg)

    return new_layer


def arrow_axis(
    *,
    plot: PlotSpec | None = None,
    size: float = 1,
    length: float = 0.25,
    angle: float = 10,
    color: str = "#3f3f3f",
    **arrow_kwargs,
) -> DeferredLayer:
    """
    Layer of arrows as the X and Y axis to the plot.

    Parameters
    ----------
    plot : PlotSpec | None, default=None
        If provided, the arrow axis is built from this plot's data and aesthetics
        regardless of which plot the resulting layer is added to. When `None`,
        the layer is deferred and introspects the plot it is added to via `+`.
    size : float
        Width of the axis lines.
    color : str
        Color of the arrow.
    angle : float
        Angle of the arrow head in degrees.
    length : float
        Span of each axis line as a fraction of its data range (0.25 covers 25%).
    **arrow_kwargs
        Additional parameters for the `arrow` layer.
        For more information on arrow parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.arrow.html

    Returns
    -------
    DeferredLayer

    Raises
    ------
    MissingAestheticError
        If `x` or `y` cannot be inferred from the receiving plot.

    Examples
    --------
    Without arrow.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        umap = cl.umap(data,"HBA2")
        umap

    Adding the arrow axis.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        umap = cl.umap(data,"HBA2")
        umap + cl.arrow_axis()

    Arrow customization.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        umap = cl.umap(data,"HBA2")
        umap + cl.arrow_axis(length=0.20,color="dark_violet")

    """
    explicit_plot = plot

    def _build(receiving_plot: PlotSpec) -> FeatureSpecArray:
        source = explicit_plot if explicit_plot is not None else receiving_plot
        frame = source.get_plot_shared_data()
        mapping = get_mapping(source, index=0)
        x = mapping.get("x")
        y = mapping.get("y")
        if x is None:
            msg = "`x` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)
        if y is None:
            msg = "`y` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)
        new_layer = theme(
            # remove axis elements
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line=element_blank(),
            # position axis titles according to arrow size
            axis_title_x=element_text(hjust=length / 2.5),  # better than 2
            axis_title_y=element_text(hjust=length / 2.5),
        )
        new_layer += _axis_arrow_layers(
            frame,
            x=x,
            y=y,
            length=length,
            size=size,
            color=color,
            angle=angle,
            arrow_kwargs=arrow_kwargs,
        )
        return new_layer

    return DeferredLayer(_build)
