from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

# Core scverse libraries
import polars as pl

# Data retrieval
from anndata import AnnData
from lets_plot import (
    LetsPlot,
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

from cellestial.themes import _THEME_SCATTER
from cellestial.util import _decide_tooltips

LetsPlot.setup_html()

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
    aes_color: str | None = None,
    aes_fill: str | None = None,
    aes_size: str | None = None,
    aes_shape: str | None = None,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    interactive: bool = False,
    **point_kwargs,
) -> PlotSpec:
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    if x in data.obs.columns:
        part = data.obs
        part_name = "obs"
    elif x in data.var.columns:
        part = data.var
        part_name = "var"
    else:
        msg = f"`{x}` is not present in `obs` nor `var` columns"
        raise ValueError(msg)

    if y not in part.columns:
        msg = f"`{y}` is not present at the {part_name} dataframe"
        raise ValueError(msg)

    # create the dataframe
    index_name = "CellID" if part_name == "obs" else "GeneID"
    frame = pl.from_pandas(part, include_index=True).rename({"None": index_name})

    # handle tooltips
    base_tooltips = [x, y, index_name]
    base_tooltips.append(aes_color) if aes_color is not None else None
    base_tooltips.append(aes_fill) if aes_fill is not None else None
    base_tooltips.append(aes_size) if aes_size is not None else None
    base_tooltips.append(aes_shape) if aes_shape is not None else None
    # decide on the tooltips
    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )
    # scatter kwargs
    if size is not None:
        point_kwargs["size"] = size
    if color is not None:
        point_kwargs["color"] = color
    if fill is not None:
        point_kwargs["fill"] = fill
    if shape is not None:
        point_kwargs["shape"] = shape

    # create the scatterplot
    scttr = (
        ggplot(data=frame)
        + geom_point(
            aes(x=x, y=y, color=aes_color, size=aes_size, shape=aes_shape, fill=aes_fill),
            tooltips=layer_tooltips(tooltips),
            **point_kwargs,
        )
        + labs(x=x, y=y)
        + _THEME_SCATTER
    )
    # handle legend wrapping
    if aes_color is not None:
        n_distinct = frame.select(aes_color).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            scttr += guides(color=guide_legend(ncol=ncol))
    if aes_fill is not None:
        n_distinct = frame.select(aes_fill).unique().height
        print(n_distinct)
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
