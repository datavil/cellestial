from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lets_plot import GGBunch, ggsave

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec
    from lets_plot.plot.subplots import SupPlotsSpec


def save(
    plot: PlotSpec | SupPlotsSpec | GGBunch,
    filename: str,
    *,
    path: str = ".",
    iframe: bool = True,
    scale: float = 1.0,
    w: int | float | None = None,
    h: int | float | None = None,
    unit: Literal["in", "cm", "mm", "px"] = "in",
    dpi: int = 300,
) -> str:
    """
    Convenience function for exporting plots.

    Parameters
    ----------
    plot : `PlotSpec` | `SupPlotsSpec` | `GGBunch`
        Plot to export.
    filename : str
        Name of the file.
        It must end with a file extension corresponding
        to one of the supported formats: SVG, HTML (or HTM), PNG, PDF (requires the pillow library).
    path : str, default='.'
        Path to the folder to save image files in.
        By default, it is current working directory.
    iframe : bool, default=True
        Whether to wrap the HTML page into an iFrame.
        Only applicable when exporting to HTML.
        Some browsers may not display some UTF-8 characters correctly when setting iframe=True
    scale : float, default=2.0
        Scaling factor for raster output.
        Only applicable when exporting to PNG or PDF.
    w : float, default=None
        Width of the output image in units.
        Only applicable when exporting to SVG, PNG, or PDF.
    h : float, default=None
        Height of the output image in units.
        Only applicable when exporting to SVG, PNG, or PDF.
    unit : {'in', 'cm', 'mm', 'px'}, default='in'
        Unit of the output image. One of: 'in', 'cm', 'mm' or 'px'.
        Only applicable when exporting to SVG, PNG, or PDF.
    dpi : int, default=300
        Resolution in dots per inch.
        Only applicable when exporting to PNG or PDF.
        The default value depends on the unit:

        - for 'px' it is 96 (output image will have the same pixel size as `w`, and `h` values)
        - for physical units ('in', 'cm', 'mm') it is 300.

    Returns
    -------
    str
        Absolute pathname of the created file.

    Notes
    -----
    Supported formats: PNG, SVG, PDF, HTML.
    Wraps `ggsave`; docstrings adapted from the lets_plot ggsave function.
    See https://lets-plot.org/python/pages/api/lets_plot.ggsave.html.
    """
    return ggsave(
        plot=plot,
        filename=filename,
        path=path,
        iframe=iframe,
        scale=scale,
        w=w,
        h=h,
        unit=unit,
        dpi=dpi,
    )
