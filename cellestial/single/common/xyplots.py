from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lets_plot import gggrid
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.single.common.xyplot import xyplot
from cellestial.util.errors import ConfilictingLengthError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.subplots import SupPlotsSpec


def xyplots(
    data: AnnData,
    x: str | Sequence[str],
    y: str | Sequence[str],
    *,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool | int = False,
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
    guides: str = "auto",
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
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    tooltips: Literal['none'] | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    include_dimensions : bool | int, default=False
        Whether to include dimensions in the DataFrame.
        Providing an integer will limit the number of dimensions to given number.
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
    guides : str, default='auto'
        Specifies how guides (legends and colorbars) should be treated in the layout.
            - 'collect' collect guides from all subplots, removing duplicates.
            - 'keep' keep guides in their original subplots; do not collect at this level.
            - 'auto' allow guides to be collected if an upper-level layout uses guides='collect';

        otherwise, keep them in subplots.
        Duplicates are identified by comparing visual properties:
        For legends: title, labels, and all aesthetic values (colors, shapes, sizes, etc.).
        For colorbars: title, domain limits, breaks, and color gradient.

        For more information on gggrid parameters:
        https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    **point_kwargs
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
            x = list(x) * len(y)
        elif len(y) == 1:
            y = list(y) * len(x)
        else:
            msg = f"Length of x ({len(x)}) and y ({len(y)}) must be the same, or one of them must be of length 1."
            raise ConfilictingLengthError(msg)

    # build plots
    plots = []
    for xi, yi in zip(x, y, strict=True):
        plot = xyplot(
            data,
            x=xi,
            y=yi,
            mapping=mapping,
            axis=axis,
            tooltips=tooltips,
            interactive=interactive,
            observations_name=observations_name,
            variables_name=variables_name,
            include_dimensions=include_dimensions,
            **point_kwargs,
        )
        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        plots.append(plot)

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
        guides=guides,
    )

    return scttrs
