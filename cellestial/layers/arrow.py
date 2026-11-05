from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from typing import TYPE_CHECKING, Literal

from ggrust import arrow, element_blank, element_line, element_text, geom_blank, theme

if TYPE_CHECKING:
    from ggrust.plot.core import FeatureSpec

# geom line widths are drawn at `size * 2.2` px, theme lines at `size` px;
# scaling keeps `size` in geom units so the arrows match the former layer version
_GEOM_LINE_UNIT = 2.2


def _arrow_theme(
    *,
    length: float,
    size: float,
    color: str,
    angle: float,
    linetype: int | str | None = None,
    arrow_kwargs: dict | None = None,
) -> FeatureSpec:
    """
    Build the arrow axis as theme axis lines with arrowheads.

    Parameters
    ----------
    length : float
        Span of each axis line as a fraction of the panel (0.25 covers 25%).
    size : float
        Width of the axis lines, in geom line-width units.
    color : str
        Color of the axis lines.
    angle : float
        Angle of the arrowhead in degrees.
    linetype : int | str | None, default=None
        Line type of the axis lines, by number or name.
    arrow_kwargs : dict | None, default=None
        Additional parameters for `arrow`.

    Returns
    -------
    `FeatureSpec`
        Theme feature specification.
    """
    line = element_line(
        color=color,
        size=size * _GEOM_LINE_UNIT,
        linetype=linetype,
        arrow=arrow(angle, **(arrow_kwargs or {})),
        extent=length,
    )
    return theme(
        # remove axis elements
        axis_text_x=element_blank(),
        axis_text_y=element_blank(),
        axis_ticks_y=element_blank(),
        axis_ticks_x=element_blank(),
        # the default theme blanks `axis_line_y`, so set both lines explicitly
        axis_line_x=line,
        axis_line_y=line,
        # position axis titles according to arrow size
        axis_title_x=element_text(hjust=length / 2.5),  # better than 2
        axis_title_y=element_text(hjust=length / 2.5),
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
    `FeatureSpec` or `FeatureSpecArray`
        Theme feature specification.
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
    linetype: int | str | None = None,
    arrow_kwargs: dict | None = None,
) -> FeatureSpec:
    """
    Theme with arrows as the X and Y axis of the plot.

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
    linetype : int | str | None, default=None
        Line type of the axis lines, by number or name. E.g "dashed".
    arrow_kwargs : dict | None, default=None
        Additional parameters for `arrow`. E.g {"type": "closed"}.

    Returns
    -------
    `FeatureSpec`
        Theme feature specification.

    Notes
    -----
    Uses ggrust arrow axes: `element_line(arrow=..., extent=...)` on the axis lines.

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
        length=length,
        size=size,
        color=color,
        angle=angle,
        linetype=linetype,
        arrow_kwargs=arrow_kwargs,
    )
