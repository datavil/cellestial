from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from cellestial.single.core.dimensional import dimensional

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec
    from scanpy import AnnData


def umap(
    data: AnnData,
    key: Literal["leiden", "louvain"] | str = "leiden",
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
) -> PlotSpec:
    return dimensional(
        data=data,
        key=key,
        dimensions="umap",
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


def tsne(
    data: AnnData,
    key: Literal["leiden", "louvain"] | str = "leiden",
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
) -> PlotSpec:
    return dimensional(
        data=data,
        key=key,
        dimensions="tsne",
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


def pca(
    data: AnnData,
    key: Literal["leiden", "louvain"] | str = "leiden",
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
) -> PlotSpec:
    return dimensional(
        data=data,
        key=key,
        dimensions="pca",
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
