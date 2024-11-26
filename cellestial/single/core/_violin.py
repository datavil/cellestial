from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

# Core scverse libraries
import polars as pl

# Data retrieval
import scanpy as sc
from lets_plot import LetsPlot, aes, geom_jitter, geom_violin, gggrid, ggplot, layer_tooltips
from lets_plot.plot.core import PlotSpec
from scanpy import AnnData

from cellestial.themes import _THEME_VIOLIN
from cellestial.util import interactive

LetsPlot.setup_html()

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


@interactive
def violin(
    data: AnnData,
    key: str,
    *,
    violin_fill: str = "#FF00FF",
    violin_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    trim: bool = False,
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    layers: list[str] | tuple[str] | Iterable[str] | None = None,
)-> PlotSpec:
    # check if data is an AnnData object
    if not isinstance(data, sc.AnnData):
        msg = "data must be an AnnData object"
        raise TypeError(msg)
    else:
        frame = pl.from_pandas(data.obs, include_index=True).rename({"None": "CellID"})
    # check if key is in the columns
    if key not in frame.columns:
        msg = f"key must be a column in the AnnData object, but {key} is not in the columns"
        raise KeyError(msg)

    # handle tooltips
    base_tooltips = ["CellID", key]
    if not show_tooltips:
        tooltips = "none"  # for letsplot, this removes the tooltips
    else:
        if isinstance(custom_tooltips, Iterable):
            tooltips = list(custom_tooltips)
        elif isinstance(add_tooltips, Iterable):
            tooltips = base_tooltips + list(add_tooltips)
        else:
            tooltips = base_tooltips

    vln = (
        ggplot(data=frame)
        + geom_violin(
            data=frame,
            mapping=aes(y=key),
            fill=violin_fill,
            color=violin_color,
            trim=trim,
            tooltips=layer_tooltips([key]),
        )
        + _THEME_VIOLIN
    )
    if show_points:
        vln += geom_jitter(
            data=frame,
            mapping=aes(y=key),
            color=point_color,
            alpha=point_alpha,
            size=point_size,
            tooltips=layer_tooltips(tooltips),
        )
    if layers is not None:
        for layer in layers:
            vln += layer

    return vln

@interactive
def violins(data, keys: list, interactive=False, multi_panel=True, **kwargs):
    if multi_panel:
        plots = list()
        for key in keys:
            plots.append(violin(data, key=key, **kwargs))

        vlns = gggrid(plots)
    else:
        frame = pl.from_pandas(data.obs[keys], include_index=True).rename({"None": "CellID"})
        frame = frame.unpivot(index="CellID", variable_name="observations", value_name="value")
        vlns = (
            ggplot(data=frame)
            + geom_violin(aes(x="observations", y="value", fill="observations"))
            + _THEME_VIOLIN
        )
    return vlns
