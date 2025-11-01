from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

# Data retrieval
from anndata import AnnData
from lets_plot import gggrid

from cellestial.single.basic.scatter import scatter
from cellestial.util.errors import ConfilictingLengthError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lets_plot.plot.core import FeatureSpec, LayerSpec
    from lets_plot.plot.subplots import SupPlotsSpec

def scatters(
    data: AnnData,
    x: str| Sequence[str],
    y: str | Sequence[str],
    *,
    axis: Literal[0,1] | None = None,
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
    add_tooltips: Iterable[str] | str | None = None,
    custom_tooltips: Iterable[str] | str | None = None,
    tooltips_title: str | None = None,
    # multi plot args
    share_labels: bool = True,
    share_axis: bool = False,
    layers: list | tuple | Sequence | FeatureSpec | LayerSpec | None = None,
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
)-> SupPlotsSpec:
    # str to list for x and y
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]

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
    for xi,yi in zip(x,y):
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

