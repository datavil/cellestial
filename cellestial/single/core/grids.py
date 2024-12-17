from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Any, Iterable, Literal

from lets_plot import element_blank, gggrid, theme

from cellestial.single.core.dimensional import dimensional, expression
from cellestial.single.core.subdimensional import pca, tsne, umap

if TYPE_CHECKING:
    from lets_plot.plot.subplots import SupPlotsSpec
    from scanpy import AnnData


def _share_labels(plot, i: int, keys: list[str], ncol: int):
    total = len(keys)
    nrow = ceil(total / ncol)
    left_places = [i for i in range(total) if i % ncol == 0]
    bottom_places = [i for i in range(total) if i >= ncol * (nrow - 1)]
    if len(bottom_places) < ncol:
        penultimate_row = list(range((nrow - 2) * ncol, (nrow - 1) * ncol))
        bottom_places.extend(penultimate_row)
    if i not in bottom_places:  # remove x axis title except for bottom row
        plot = plot + theme(axis_title_x=element_blank())
    if i not in left_places:  # remove y axis title except for left column
        plot = plot + theme(axis_title_y=element_blank())

    return plot


def _share_axis(plot, i: int, keys: list[str], ncol: int, axis_type: Literal["axis", "arrow"]):
    total = len(keys)
    nrow = ceil(total / ncol)
    left_places = [i for i in range(total) if i % ncol == 0]
    bottom_places = [i for i in range(total) if i >= ncol * (nrow - 1)]
    if len(bottom_places) < ncol:
        penultimate_row = list(range((nrow - 2) * ncol, (nrow - 1) * ncol))
        bottom_places.extend(penultimate_row)

    if axis_type == "axis":
        if i not in bottom_places:  # remove x axis title except for bottom row
            plot = plot + theme(
                # remove axis elements
                axis_text_x=element_blank(),
                axis_text_y=element_blank(),
                axis_ticks_y=element_blank(),
                axis_ticks_x=element_blank(),
                axis_line=element_blank(),
            )
        if i not in left_places:  # remove y axis title except for left column
            plot = plot + theme(
                # remove axis elements
                axis_text_x=element_blank(),
                axis_text_y=element_blank(),
                axis_ticks_y=element_blank(),
                axis_ticks_x=element_blank(),
                axis_line=element_blank(),
            )
    elif axis_type == "arrow":
        pass
    else:
        msg = f"expected 'axis' or 'arrow' for 'axis_type' argument, but received {axis_type}"
        raise ValueError(msg)

    return plot


def dimensionals(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] = ("leiden",),
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    size: float = 0.8,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    share_labels: bool = True,
    share_axis: bool = True,
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    layers: list | tuple | Iterable | None = None,
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
    **point_kwargs: dict[str, Any],
) -> SupPlotsSpec:
    plots = []

    for i, key in enumerate(keys):
        plot = dimensional(
            data=data,
            key=key,
            dimensions=dimensions,
            size=size,
            interactive=interactive,
            cluster_name=cluster_name,
            barcode_name=barcode_name,
            color_low=color_low,
            color_high=color_high,
            axis_type=axis_type,
            arrow_length=arrow_length,
            arrow_size=arrow_size,
            arrow_color=arrow_color,
            arrow_angle=arrow_angle,
            **point_kwargs,
        )

        if layers is not None:
            for layer in layers:
                plot += layer
        if share_labels:
            plot = _share_labels(plot, i, keys, ncol)
        if share_axis:
            if axis_type is not None:
                plot = _share_axis(plot, i, keys, ncol, axis_type)

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


def umaps(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] = ("leiden",),
    *,
    size: float = 0.8,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    share_labels: bool = True,
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    layers: list | tuple | Iterable | None = None,
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
    **point_kwargs: dict[str, Any],
) -> SupPlotsSpec:
    plots = []

    for i, key in enumerate(keys):
        plot = umap(
            data=data,
            key=key,
            size=size,
            interactive=interactive,
            cluster_name=cluster_name,
            barcode_name=barcode_name,
            color_low=color_low,
            color_high=color_high,
            axis_type=axis_type,
            arrow_length=arrow_length,
            arrow_size=arrow_size,
            arrow_color=arrow_color,
            arrow_angle=arrow_angle,
            **point_kwargs,
        )

        if layers is not None:
            for layer in layers:
                plot += layer
        if share_labels:
            plot = _share_labels(plot, i, keys, ncol)

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


def tsnes(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] = ("leiden",),
    *,
    size: float = 0.8,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    share_labels: bool = True,
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    layers: list | tuple | Iterable | None = None,
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
    **point_kwargs: dict[str, Any],
) -> SupPlotsSpec:
    plots = []

    for i, key in enumerate(keys):
        plot = tsne(
            data=data,
            key=key,
            size=size,
            interactive=interactive,
            cluster_name=cluster_name,
            barcode_name=barcode_name,
            color_low=color_low,
            color_high=color_high,
            axis_type=axis_type,
            arrow_length=arrow_length,
            arrow_size=arrow_size,
            arrow_color=arrow_color,
            arrow_angle=arrow_angle,
            **point_kwargs,
        )

        if layers is not None:
            for layer in layers:
                plot += layer

        if share_labels:
            plot = _share_labels(plot, i, keys, ncol)

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


def pcas(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] = ("leiden",),
    *,
    size: float = 0.8,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    share_labels: bool = True,
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    layers: list | tuple | Iterable | None = None,
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
    **point_kwargs: dict[str, Any],
) -> SupPlotsSpec:
    plots = []

    for i, key in enumerate(keys):
        plot = pca(
            data=data,
            key=key,
            size=size,
            interactive=interactive,
            cluster_name=cluster_name,
            barcode_name=barcode_name,
            color_low=color_low,
            color_high=color_high,
            axis_type=axis_type,
            arrow_length=arrow_length,
            arrow_size=arrow_size,
            arrow_color=arrow_color,
            arrow_angle=arrow_angle,
            **point_kwargs,
        )

        if layers is not None:
            for layer in layers:
                plot += layer
        if share_labels:
            plot = _share_labels(plot, i, keys, ncol)

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


def expressions(
    data: AnnData,
    genes: list[str] | tuple[str] | Iterable[str] = ("leiden",),
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    size: float = 0.8,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    cluster_type: Literal["leiden", "louvain"] | None = None,
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    share_labels: bool = True,
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    layers: list | tuple | Iterable | None = None,
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
) -> SupPlotsSpec:
    plots = []

    for i, gene in enumerate(genes):
        plot = expression(
            data=data,
            gene=gene,
            dimensions=dimensions,
            size=size,
            interactive=interactive,
            cluster_name=cluster_name,
            cluster_type=cluster_type,
            barcode_name=barcode_name,
            color_low=color_low,
            color_high=color_high,
            axis_type=axis_type,
            arrow_length=arrow_length,
            arrow_size=arrow_size,
            arrow_color=arrow_color,
            arrow_angle=arrow_angle,
        )

        if layers is not None:
            for layer in layers:
                plot += layer
        if share_labels:
            plot = _share_labels(plot, i, genes, ncol)

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
