from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal

from lets_plot import gggrid

from cellestial.single.core.dimensional import dimensional, expression
from cellestial.single.core.subdimensional import pca, tsne, umap

if TYPE_CHECKING:
    from lets_plot.plot.subplots import SupPlotsSpec
    from scanpy import AnnData


def dimensionals(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] | str = ("leiden",),
    ncol: int | None = None,
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    size: float = 0.8,
    point_shape: int = 16,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 3,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 20,
    layers: list | tuple | Iterable | None = None,
) -> SupPlotsSpec:
    grid = list()

    for key in keys:
        plot = dimensional(
            data=data,
            key=key,
            dimensions=dimensions,
            size=size,
            point_shape=point_shape,
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
        )

        if layers is not None:
            for layer in layers:
                plot += layer

        grid.append(plot)

    return gggrid(grid, ncol=ncol)


def umaps(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] | str = ("leiden",),
    ncol: int | None = None,
    *,
    size: float = 0.8,
    point_shape: int = 16,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 3,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 20,
    layers: list | tuple | Iterable | None = None,
    **kwargs
) -> SupPlotsSpec:
    grid = list()

    for key in keys:
        plot = umap(
            data=data,
            key=key,
            size=size,
            point_shape=point_shape,
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
        )

        if layers is not None:
            for layer in layers:
                plot += layer

        grid.append(plot)

    return gggrid(grid, ncol=ncol, **kwargs)

def tsnes(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] | str = ("leiden",),
    ncol: int | None = None,
    *,
    size: float = 0.8,
    point_shape: int = 16,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 3,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 20,
    layers: list | tuple | Iterable | None = None,
) -> SupPlotsSpec:
    grid = list()

    for key in keys:
        plot = tsne(
            data=data,
            key=key,
            size=size,
            point_shape=point_shape,
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
        )

        if layers is not None:
            for layer in layers:
                plot += layer

        grid.append(plot)

    return gggrid(grid, ncol=ncol)


def pcas(
    data: AnnData,
    keys: list[str] | tuple[str] | Iterable[str] | str = ("leiden",),
    ncol: int | None = None,
    *,
    size: float = 0.8,
    point_shape: int = 16,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 3,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 20,
    layers: list | tuple | Iterable | None = None,
) -> SupPlotsSpec:
    grid = list()

    for key in keys:
        plot = pca(
            data=data,
            key=key,
            size=size,
            point_shape=point_shape,
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
        )

        if layers is not None:
            for layer in layers:
                plot += layer
        grid.append(plot)

    return gggrid(grid, ncol=ncol)


def expressions(
    data: AnnData,
    genes: list[str] | tuple[str] | Iterable[str] | str = ("leiden",),
    ncol: int | None = None,
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    size: float = 0.8,
    point_shape: int = 16,
    interactive: bool = False,  # used by interactive decorator
    cluster_name: str = "Cluster",
    cluster_type: Literal["leiden", "louvain"] | None = None,
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 3,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 20,
    layers: list | tuple | Iterable | None = None,
) -> SupPlotsSpec:
    grid = list()

    for gene in genes:
        plot = expression(
            data=data,
            gene=gene,
            dimensions=dimensions,
            size=size,
            point_shape=point_shape,
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

        grid.append(plot)

    return gggrid(grid, ncol=ncol)
