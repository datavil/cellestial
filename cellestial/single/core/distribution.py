from __future__ import annotations

from collections.abc import Iterable
from math import ceil
from typing import TYPE_CHECKING

# Core scverse libraries
import polars as pl

# Data retrieval
import scanpy as sc
from lets_plot import (
    LetsPlot,
    aes,
    geom_boxplot,
    geom_jitter,
    geom_violin,
    gggrid,
    ggplot,
    ggsize,
    ggtb,
    guide_legend,
    guides,
    layer_tooltips,
)
from lets_plot.plot.core import PlotSpec
from scanpy import AnnData

from cellestial.themes import _THEME_BOXPLOT, _THEME_VIOLIN

LetsPlot.setup_html()

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def violin(
    data: AnnData,
    key: str,
    *,
    color: str | None = None,
    fill: str | None = None,
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
    interactive: bool = False,
) -> PlotSpec:
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

    # handle fill and color
    violin_fill = None if fill is not None else violin_fill
    violin_color = None if color is not None else violin_color
    # handle violimn tooltips
    violin_tooltips = [key]
    violin_tooltips.append(color) if color is not None else None
    violin_tooltips.append(fill) if fill is not None else None
    # generate the plot
    vln = (
        ggplot(data=frame)
        + geom_violin(
            data=frame,
            mapping=aes(y=key, color=color, fill=fill),
            fill=violin_fill,
            color=violin_color,
            trim=trim,
            tooltips=layer_tooltips(violin_tooltips),
        )
        + _THEME_VIOLIN
    )
    # handle the point (jitter)
    if show_points:
        vln += geom_jitter(
            data=frame,
            mapping=aes(y=key),
            color=point_color,
            alpha=point_alpha,
            size=point_size,
            tooltips=layer_tooltips(tooltips),
        )

    # wrap the legend
    if fill is not None:
        n_distinct = frame.select(fill).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            vln = vln + guides(fill=guide_legend(ncol=ncol))
    if color is not None:
        n_distinct = frame.select(color).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            vln = vln + guides(color=guide_legend(ncol=ncol))

    # handle interactive
    if interactive:
        vln += ggtb()

    return vln


def violins(
    data,
    keys: list[str] | tuple[str] | Iterable[str],
    *,
    color: str | None = None,
    fill: str | None = None,
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
    interactive: bool = False,
    multi_panel: bool = True,
    **grid_kwargs,
):
    if multi_panel:  # standard grid plotting
        plots = []
        for key in keys:
            vln = violin(
                data,
                key=key,
                color=color,
                fill=fill,
                violin_fill=violin_fill,
                violin_color=violin_color,
                point_color=point_color,
                point_alpha=point_alpha,
                point_size=point_size,
                trim=trim,
                show_tooltips=show_tooltips,
                show_points=show_points,
                add_tooltips=add_tooltips,
                custom_tooltips=custom_tooltips,
                interactive=interactive,
            )
            # handle the layers
            if layers is not None:
                for layer in layers:
                    vln += layer

            plots.append(vln)

        vlns = gggrid(plots, ncol=grid_kwargs.get("ncol"))

    else:  # unpivot the data so that it can be plotted in a single (combined) panel
        frame = pl.from_pandas(data.obs[keys], include_index=True).rename({"None": "CellID"})
        frame = frame.unpivot(index="CellID", variable_name="observations", value_name="value")
        vlns = (
            ggplot(data=frame)
            + geom_violin(aes(x="observations", y="value", fill="observations"))
            + _THEME_VIOLIN
            + ggsize(800, 400)
        )

        # handle the layers
        if layers is not None:
            for layer in layers:
                vlns += layer

    # handle interactive
    if interactive:
        vlns += ggtb()

    return vlns


def boxplot(
    data: AnnData,
    key: str,
    *,
    color: str | None = None,
    fill: str | None = None,
    boxplot_fill: str = "#FF00FF",
    boxplot_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    trim: bool = False,
    show_tooltips: bool = True,
    show_points: bool = False,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    interactive: bool = False,
    boxplot_kwargs: dict | None = None,
) -> PlotSpec:
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

    # handle fill and color
    boxplot_fill = None if fill is not None else boxplot_fill
    boxplot_color = None if color is not None else boxplot_color
    # handle box tooltips
    boxplot_tooltips = [key]
    boxplot_tooltips.append(color) if color is not None else None
    boxplot_tooltips.append(fill) if fill is not None else None

    # handle boxplot_kwargs
    if boxplot_kwargs is None:
        boxplot_kwargs = {}

    # generate the plot
    bxplt = (
        ggplot(data=frame)
        + geom_boxplot(
            data=frame,
            mapping=aes(y=key, color=color, fill=fill),
            fill=boxplot_fill,
            color=boxplot_color,
            trim=trim,
            tooltips=layer_tooltips(boxplot_tooltips),
            **boxplot_kwargs,
        )
        + _THEME_BOXPLOT
    )
    # handle the point (jitter)
    if show_points:
        bxplt += geom_jitter(
            data=frame,
            mapping=aes(y=key),
            color=point_color,
            alpha=point_alpha,
            size=point_size,
            tooltips=layer_tooltips(tooltips),
        )

    # wrap the legend
    if fill is not None:
        n_distinct = frame.select(fill).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            bxplt = bxplt + guides(fill=guide_legend(ncol=ncol))
    if color is not None:
        n_distinct = frame.select(color).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            bxplt = bxplt + guides(color=guide_legend(ncol=ncol))

    # handle interactive
    if interactive:
        bxplt += ggtb()

    return bxplt


def boxplots(
    data,
    keys: list[str] | tuple[str] | Iterable[str],
    *,
    color: str | None = None,
    fill: str | None = None,
    boxplot_fill: str = "#FF00FF",
    boxplot_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    trim: bool = False,
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    layers: list[str] | tuple[str] | Iterable[str] | None = None,
    interactive: bool = False,
    multi_panel: bool = True,
    **grid_kwargs,
):
    if multi_panel:  # standard grid plotting
        plots = []
        for key in keys:
            bxplt = boxplot(
                data,
                key=key,
                color=color,
                fill=fill,
                boxplot_fill=boxplot_fill,
                boxplot_color=boxplot_color,
                point_color=point_color,
                point_alpha=point_alpha,
                point_size=point_size,
                trim=trim,
                show_tooltips=show_tooltips,
                show_points=show_points,
                add_tooltips=add_tooltips,
                custom_tooltips=custom_tooltips,
                interactive=interactive,
            )
            # handle the layers
            if layers is not None:
                for layer in layers:
                    bxplt += layer

            plots.append(bxplt)

        bxplts = gggrid(plots, ncol=grid_kwargs.get("ncol"))

    else:  # unpivot the data so that it can be plotted in a single (combined) panel
        frame = pl.from_pandas(data.obs[keys], include_index=True).rename({"None": "CellID"})
        frame = frame.unpivot(index="CellID", variable_name="observations", value_name="value")
        bxplts = (
            ggplot(data=frame)
            + geom_boxplot(aes(x="observations", y="value", fill="observations"))
            + _THEME_BOXPLOT
            + ggsize(800, 400)
        )

        # handle the layers
        if layers is not None:
            for layer in layers:
                bxplts += layer

    # handle interactive
    if interactive:
        bxplts += ggtb()

    return bxplts
