from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lets_plot import (
    arrow,
    element_blank,
    element_text,
    geom_blank,
    geom_segment,
    theme,
)

from cellestial.util import get_mapping

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpecArray, PlotSpec
    from polars import DataFrame


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
        Size of the arrow.
    arrow_color : str
        Color of the arrow.
    arrow_angle : float
        Angle of the arrow head in degrees.
    arrow_length : float
        Length of the arrow head (px).

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
        x_max = frame[x].max()
        x_min = frame[x].min()
        y_max = frame[y].max()
        y_min = frame[y].min()

        # find total difference between the max and min for both axis
        x_diff = x_max - x_min  # ty:ignore[unsupported-operator]
        y_diff = y_max - y_min  # ty:ignore[unsupported-operator]

        # find the ends of the arrows
        # ensure the arrow length is the same for both axis
        xend = x_min + arrow_length * x_diff
        yend = y_min + arrow_length * y_diff

        # adjust bottom ends of arrows
        adjust_rate = 0.025
        x0 = x_min - x_diff * adjust_rate
        y0 = y_min - y_diff * adjust_rate

        # X axis
        new_layer += geom_segment(
            x=x0,
            y=y0,
            xend=xend,
            yend=y0,
            color=arrow_color,
            size=arrow_size,
            arrow=arrow(arrow_angle),
        )
        # Y axis
        new_layer += geom_segment(
            x=x0,
            y=y0,
            xend=x0,
            yend=yend,
            color=arrow_color,
            size=arrow_size,
            arrow=arrow(arrow_angle),
        )
    else:
        msg = f"expected 'axis' or 'arrow' for 'axis_type' argument, but received {axis_type}"
        raise ValueError(msg)

    return new_layer


def arrow_axis(
    plot: PlotSpec,
    *,
    size: float = 1,
    length: float = 0.25,
    angle: float = 10,
    color: str = "#3f3f3f",
    **arrow_kwargs,
) -> FeatureSpecArray:
    """
    Layer of arrows as the X and Y axis to the plot.

    Parameters
    ----------
    plot : PlotSpec
        The plot to which the layer will be added. Used to extract data and aesthetics.
    size : float
        Size of the arrow.
    color : str
        Color of the arrow.
    angle : float
        Angle of the arrow head in degrees.
    length : float
        Length of the arrow head (px).
    **arrow_kwargs
        Additional parameters for the `arrow` layer.
        For more information on arrow parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.arrow.html

    Returns
    -------
    FeatureSpecArray

    Examples
    --------
    Without arrow.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        umap = cl.umap(data,"HBA2")
        umap

    Adding the arrow axis.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        umap = cl.umap(data,"HBA2")
        umap + cl.arrow_axis(umap)

    Arrow customization.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        umap = cl.umap(data,"HBA2")
        umap + cl.arrow_axis(umap,length=0.20,color="dark_violet")

    """
    frame = plot.get_plot_shared_data()
    mapping = get_mapping(plot, index=0)
    x: str = mapping.get("x")
    y: str = mapping.get("y")
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
    x_max = frame[x].max()
    x_min = frame[x].min()
    y_max = frame[y].max()
    y_min = frame[y].min()

    # find total difference between the max and min for both axis
    x_diff = x_max - x_min  # ty:ignore[unsupported-operator]
    y_diff = y_max - y_min  # ty:ignore[unsupported-operator]

    # find the ends of the arrows
    xend = x_min + length * x_diff
    yend = y_min + length * y_diff

    # adjust bottom ends of arrows
    adjust_rate = 0.025
    x0 = x_min - x_diff * adjust_rate
    y0 = y_min - y_diff * adjust_rate

    # X axis
    new_layer += geom_segment(
        x=x0,
        y=y0,
        xend=xend,
        yend=y0,
        color=color,
        size=size,
        arrow=arrow(angle, **arrow_kwargs),
    )
    # Y axis
    new_layer += geom_segment(
        x=x0,
        y=y0,
        xend=x0,
        yend=yend,
        color=color,
        size=size,
        arrow=arrow(angle, **arrow_kwargs),
    )

    return new_layer
