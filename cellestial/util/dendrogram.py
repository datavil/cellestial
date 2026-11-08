from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "ggrust.plot.core"]

from typing import Literal

from anndata import AnnData
from ggrust import (
    ClusterTree,
    coord_cartesian,
    coord_fixed,
    element_line,
    guide_axis_dendro,
    scale_y_dendro,
    theme,
)
from ggrust.plot.core import FeatureSpec, FeatureSpecArray

from cellestial.util.errors import KeyNotFoundError

# geom line widths are drawn at `size * 2.2` px, theme lines at `size` px;
# scaling keeps `dendrogram_size` in the geom units of the former path layer.
_GEOM_LINE_UNIT = 2.2


def _dendrogram_info(
    data: AnnData,
    group_by: str,
    *,
    use_key: str | None = None,
) -> dict:
    """Get or compute the scanpy dendrogram of `group_by` stored in `data.uns`."""
    if not isinstance(data, AnnData):
        msg = f"Unsupported data type: `{type(data)}`"
        raise TypeError(msg)
    key = use_key if use_key is not None else f"dendrogram_{group_by}"
    if key not in data.uns:
        if group_by not in data.obs.columns:
            # Reached when a multimodal object groups by a container-level
            # column that the selected modality does not carry.
            msg = (
                f"Cannot compute a dendrogram for `{group_by}`: it is not an "
                "observation column of the data the dendrogram is computed from.\n"
                "Precompute the dendrogram, or group by a column that data carries."
            )
            raise KeyNotFoundError(msg)
        import scanpy as sc

        sc.tl.dendrogram(data, groupby=group_by, key_added=key)
    return data.uns[key]


def _get_dendrogram_tree(
    data: AnnData,
    group_by: str,
    *,
    use_key: str | None = None,
) -> ClusterTree:
    """
    Get or compute the dendrogram of `group_by` as a ggrust tree.

    Leaves are labelled by category; the tree's order is scanpy's
    `categories_ordered`.
    """
    dendro = _dendrogram_info(data, group_by, use_key=use_key)
    # Linkage leaf `i` is category `i`; recover the categories from the stored order.
    labels = [""] * len(dendro["categories_ordered"])
    for index, category in zip(
        dendro["categories_idx_ordered"], dendro["categories_ordered"], strict=True
    ):
        labels[index] = str(category)
    return ClusterTree.from_linkage(dendro["linkage"], labels)


def _dendrogram_y_axis(
    tree: ClusterTree,
    *,
    limits: list[float],
    fixed: bool = False,
    leaf_levels: dict[str, list[str]] | None = None,
    breaks: list[str] | None = None,
    labels: list[str] | None = None,
) -> FeatureSpecArray:
    """
    Order the y-axis by `tree` and draw it right of the panel, as scanpy does.

    Layers keep numeric y positions: level `i` of the axis sits at `i`. Without
    `leaf_levels` the levels are the tree's leaves; with it, each leaf covers
    its run of levels (the rows of a group). `limits` replace the continuous
    scale's limits (the tree fixes the levels); `fixed` keeps the square cells
    `geom_raster` plots default to. `breaks` and `labels` pick and name levels.
    """
    coord = coord_fixed(ylim=limits) if fixed else coord_cartesian(ylim=limits)
    scale = scale_y_dendro(
        tree,
        breaks=breaks,
        labels=labels,
        expand=[0, 0],
        guide=guide_axis_dendro(side="opposite"),
        leaf_levels=leaf_levels,
    )
    return scale + coord


def _dendrogram_line_theme(
    *, color: str, size: float, axis: Literal["x", "y"] = "y"
) -> FeatureSpec:
    """Style the `axis` tree, drawn with its tick line; zero-length ticks stay hidden."""
    line = element_line(color=color, size=size * _GEOM_LINE_UNIT)
    if axis == "x":
        return theme(axis_ticks_x=line, axis_ticks_length_x=0)
    return theme(axis_ticks_y=line, axis_ticks_length_y=0)
