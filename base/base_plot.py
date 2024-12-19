from __future__ import annotations

from collections.abc import Iterable
from typing import Any, List, Optional, Union

from lets_plot import PlotSpec, gggrid, ggplot
from scanpy import AnnData


def base_plot(
    data: AnnData,
    key: str,
    *,
    interactive: bool = False,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    **plot_kwargs: dict[str, Any],
) -> PlotSpec:
    plot = ggplot()
    return plot


def base_multi_plot(
    data: AnnData,
    keys: list[str],
    *,
    interactive: bool = False,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list | None = None,
    heights: list | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    **plot_kwargs: dict[str, Any],
) -> PlotSpec:
    plots = []
    for key in keys:
        plot = base_plot(
            data,
            key,
            interactive=interactive,
            show_tooltips=show_tooltips,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            **plot_kwargs,
        )
        plots.append(plot)

    return gggrid(
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
