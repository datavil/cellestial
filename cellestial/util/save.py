from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from typing import TYPE_CHECKING, Any

from ggrust import ggsave

if TYPE_CHECKING:
    from ggrust.plot.core import PlotSpec
    from ggrust.plot.subplots import SupPlotsSpec


def save(
    plot: PlotSpec | SupPlotsSpec,
    filename: str,
    *,
    scale: float = 2.0,
    **ggsave_kwargs: Any,
) -> str:
    """
    Save a plot to a file, with a sharper default raster scale than `ggsave`.

    Parameters
    ----------
    plot : `PlotSpec` | `SupPlotsSpec`
        Plot to save, from a single or a plural plot function.
    filename : str
        File name with an extension that picks the format: '.png', '.svg', '.pdf',
        '.html', '.htm' or '.ggrs'.
    scale : float, default=2.0
        Pixel scale for PNG and PDF output. The default 2.0 doubles the pixel density
        so text and points stay crisp on high resolution screens and in slides.
    **ggsave_kwargs : Any
        Passed unchanged to `ggsave` and follow its defaults. Common ones:

        - `path` : folder to save into; the current working directory when omitted.
        - `w`, `h` : output width and height (SVG, PNG, PDF); give both.
        - `unit` : unit of `w` and `h`, one of 'in', 'cm', 'mm', 'px'; inches when omitted.
        - `dpi` : resolution for PNG and PDF.
        - `iframe` : HTML only, wrap the page in an iframe (default True).

    Returns
    -------
    str
        Absolute path of the written file.

    Raises
    ------
    TypeError
        If `ggsave_kwargs` holds a name `ggsave` does not accept.
    ValueError
        Raised by `ggsave` (as ggrust's `ExportError`) when `plot` is not a plot, or
        when `filename` has no name, no extension or an unsupported extension.

    Examples
    --------
    >>> import tempfile
    >>> import cellestial as cl
    >>> adata = cl.datasets.pbmc3k(cache_directory="data")
    >>> plot = cl.umap(adata, "leiden")
    >>> folder = tempfile.mkdtemp()
    >>> output = cl.save(plot, "umap.png", path=folder)
    >>> output = cl.save(plot, "umap.pdf", path=folder, w=6, h=4, unit="in", dpi=300)

    Notes
    -----
    Only `scale` differs from `ggsave`; every other setting, including `dpi` and
    `unit`, uses the `ggsave` default unless you pass it.
    """
    return ggsave(plot=plot, filename=filename, scale=scale, **ggsave_kwargs)
