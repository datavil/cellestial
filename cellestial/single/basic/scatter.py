from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

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

from cellestial.frames import _axis_data, _construct_cell_frame, _construct_var_frame
from cellestial.themes import _THEME_SCATTER
from cellestial.util import _build_tooltips, _decide_tooltips

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lets_plot.plot.core import PlotSpec


def scatter(
    data: AnnData,
    x: str,
    y: str,
    *,
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
    var_name: str = "Gene",
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | str | None = None,
    tooltips_title: str | None = None,
    **point_kwargs,
) -> PlotSpec:
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

    # handle tooltips
    axis = _axis_data(data=data, key=x)
    identifier = barcode_name if axis == 0 else var_name
    base_tooltips = [x, y, identifier]
    if color is not None:
        base_tooltips.append(color)
    if fill is not None:
        base_tooltips.append(fill)
    if size is not None:
        base_tooltips.append(size)
    if shape is not None:
        base_tooltips.append(shape)

    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )
    tooltips_object = _build_tooltips(
        tooltips=tooltips,
        title=tooltips_title,
    )

    # construct the frame
    all_keys = [t for t in base_tooltips if t != identifier]  # base_tooltips minus the identifier
    if tooltips != "none":
        for tooltip in tooltips:
            if tooltip not in all_keys and tooltip != identifier:
                print(tooltip)
                all_keys.append(tooltip)

    if axis == 0:  # for obs and var_names
        frame = _construct_cell_frame(
            data=data,
            keys=all_keys,
            xy=None,
            barcode_name=barcode_name,
        )
    elif axis == 1:  # for var
        frame = _construct_var_frame(
            data=data,
            keys=all_keys,
            var_name=var_name,
        )

    # scatter kwargs
    if size is not None:
        point_kwargs["size"] = point_size
    if color is not None:
        point_kwargs["color"] = point_color
    if fill is not None:
        point_kwargs["fill"] = point_fill
    if shape is not None:
        point_kwargs["shape"] = point_shape

    # create the scatterplot
    scttr = (
        ggplot(data=frame)
        + geom_point(
            aes(x=x, y=y, color=color, size=size, shape=shape, fill=fill),
            tooltips=tooltips_object,
            **point_kwargs,
        )
        + labs(x=x, y=y)
        + _THEME_SCATTER
    )
    # handle legend wrapping
    if color is not None:
        n_distinct = frame.select(color).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            scttr += guides(color=guide_legend(ncol=ncol))
    if fill is not None:
        n_distinct = frame.select(fill).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            scttr += guides(fill=guide_legend(ncol=ncol))
    # handle interactive
    if interactive:
        scttr += ggtb()

    return scttr


def test_scatter():
    import scanpy as sc

    data = sc.read("data/pbmc3k_pped.h5ad")
    plot = scatter(
        data,
        "n_genes",
        "pct_counts_in_top_50_genes",
        add_tooltips=["sample"],
        aes_color="leiden",
        interactive=True,
        size=0.6,
    )
    plot.to_html("testplots/test_scatter.html")


if __name__ == "__main__":
    test_scatter()
