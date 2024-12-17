from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

from cellestial.single.core.dimensional import dimensional

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec
    from scanpy import AnnData


def umap(
    data: AnnData,
    key: Literal["leiden", "louvain"] | str = "leiden",
    *,
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "CellID",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    **point_kwargs: dict[str, Any],
) -> PlotSpec:
    return dimensional(
        data=data,
        key=key,
        dimensions="umap",
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
        show_tooltips=show_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        **point_kwargs,
    )


def tsne(
    data: AnnData,
    key: Literal["leiden", "louvain"] | str = "leiden",
    *,
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "CellID",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    **point_kwargs: dict[str, Any],
) -> PlotSpec:
    return dimensional(
        data=data,
        key=key,
        dimensions="tsne",
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
        show_tooltips=show_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        **point_kwargs,
    )


def pca(
    data: AnnData,
    key: Literal["leiden", "louvain"] | str = "leiden",
    *,
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "CellID",
    color_low: str = "#e6e6e6",
    color_high: str = "#377eb8",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    **point_kwargs: dict[str, Any],
) -> PlotSpec:
    return dimensional(
        data=data,
        key=key,
        dimensions="pca",
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
        show_tooltips=show_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        **point_kwargs,
    )
