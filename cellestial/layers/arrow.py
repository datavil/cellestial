from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from typing import TYPE_CHECKING, Literal

from ggrust import (
    arrow,
    element_blank,
    element_line,
    element_text,
    geom_blank,
    theme,
)

if TYPE_CHECKING:
    from ggrust.plot.core import FeatureSpec


def _arrow_theme(
    *,
    length: float,
    size: float,
    color: str,
    angle: float,
    arrow_kwargs: dict | None = None,
) -> FeatureSpec:
    """
    Theme drawing the X and Y axis lines as arrows from the bottom-left corner.

    Parameters
    ----------
    length : float
        Span of each axis line as a fraction of the panel, in (0, 1].
    size : float
        Width of the axis lines.
    color : str
        Color of the axis lines and arrowheads.
    angle : float
        Angle of the arrowhead in degrees.
    arrow_kwargs : dict | None
        Additional parameters for the `arrow`.

    Returns
    -------
    `FeatureSpec`
        The theme replacing the axis text, ticks, and lines.

    Notes
    -----
    The arrows are axis lines drawn by the plot layout, so they do not depend on
    the data range and stay fixed when an interactive plot is zoomed. The axis
    titles sit at the start of each axis, next to the arrows.
    """
    line = element_line(
        size=size,
        color=color,
        extent=length,
        arrow=arrow(angle=angle, **(arrow_kwargs or {})),
    )
    return theme(
        # remove axis elements
        axis_text_x=element_blank(),
        axis_text_y=element_blank(),
        axis_ticks_y=element_blank(),
        axis_ticks_x=element_blank(),
        # both lines explicitly: a theme blanking `axis_line_y` would win over `axis_line`
        axis_line_x=line,
        axis_line_y=line,
        axis_title_x=element_text(hjust=0),
        axis_title_y=element_text(hjust=0),
    )


def _modify_axis(
    *,
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
    axis_type : {'axis', 'arrow'} | None
        Whether to use regular axis or arrows as the axis.
    arrow_size : float
        Width of the axis lines.
    arrow_color : str
        Color of the arrow.
    arrow_angle : float
        Angle of the arrow head in degrees.
    arrow_length : float
        Span of each axis line as a fraction of the panel (0.25 covers 25%).

    Returns
    -------
    `FeatureSpec`
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
        return _arrow_theme(
            length=arrow_length,
            size=arrow_size,
            color=arrow_color,
            angle=arrow_angle,
        )
    else:
        msg = f"expected 'axis' or 'arrow' for 'axis_type' argument, but received {axis_type}"
        raise ValueError(msg)


def arrow_axis(
    *,
    size: float = 1,
    length: float = 0.25,
    angle: float = 10,
    color: str = "#3f3f3f",
    **arrow_kwargs,
) -> FeatureSpec:
    """
    Layer of arrows as the X and Y axis to the plot.

    Parameters
    ----------
    size : float
        Width of the axis lines.
    color : str
        Color of the arrow.
    angle : float
        Angle of the arrow head in degrees.
    length : float
        Span of each axis line as a fraction of the panel (0.25 covers 25%).
    **arrow_kwargs
        Additional parameters for the `arrow`, such as `type` or `ends`.
        For more information on arrow parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.arrow.html

    Returns
    -------
    FeatureSpec

    Notes
    -----
    Draws the axis lines themselves as arrows from the bottom-left corner of the
    panel, with the axis titles next to them, so the arrows do not depend on the
    plotted data.

    Examples
    --------
    Without arrow.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        umap = cl.umap(data,"HBA2")
        umap

    Adding the arrow axis.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        umap = cl.umap(data,"HBA2")
        umap + cl.arrow_axis()

    Arrow customization.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        umap = cl.umap(data,"HBA2")
        umap + cl.arrow_axis(length=0.20,color="dark_violet")

    """
    return _arrow_theme(
        length=length, size=size, color=color, angle=angle, arrow_kwargs=arrow_kwargs
    )
