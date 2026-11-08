from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "ggrust.plot.core", "numpy"]

import numpy as np
import polars as pl
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
    tree: ClusterTree, *, limits: list[float], fixed: bool = False
) -> FeatureSpecArray:
    """
    Order the y-axis by `tree` and draw it right of the panel, as scanpy does.

    Layers keep numeric y positions: leaf `i` of the tree's order sits at `i`.
    `limits` replace the continuous scale's limits (the tree fixes the levels);
    `fixed` keeps the square cells `geom_raster` plots default to.
    """
    coord = coord_fixed(ylim=limits) if fixed else coord_cartesian(ylim=limits)
    return scale_y_dendro(tree, expand=[0, 0], guide=guide_axis_dendro(side="opposite")) + coord


def _dendrogram_line_theme(*, color: str, size: float) -> FeatureSpec:
    """Style the y-axis tree, drawn with the y tick line; zero-length ticks stay hidden."""
    return theme(
        axis_ticks_y=element_line(color=color, size=size * _GEOM_LINE_UNIT),
        axis_ticks_length_y=0,
    )


def _get_dendrogram(
    data: AnnData,
    group_by: str,
    *,
    use_key: str | None = None,
) -> tuple[list[str], pl.DataFrame]:
    """Get or compute the dendrogram for `group_by` and extract paths."""
    if isinstance(data, AnnData):
        dendro = _dendrogram_info(data, group_by, use_key=use_key)
        categories_ordered = list(dendro["categories_ordered"])
        dendro_info = dendro["dendrogram_info"]

        icoord = np.array(dendro_info["icoord"])
        dcoord = np.array(dendro_info["dcoord"])

        # scipy places leaves at 5, 15, 25, ... → normalize to 0, 1, 2, ...
        icoord = (icoord - 5) / 10

        max_height = dcoord.max()
        if max_height > 0:
            dcoord = dcoord / max_height

        path_x, path_y, path_group = [], [], []
        for i in range(len(icoord)):
            for j in range(icoord.shape[1]):
                path_x.append(float(icoord[i][j]))
                path_y.append(float(dcoord[i][j]))
                path_group.append(i)
        paths = pl.DataFrame({"x": path_x, "y": path_y, "group": path_group})
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise TypeError(msg)

    return categories_ordered, paths


def _get_dendrogram_path_frame(
    paths: pl.DataFrame,
    *,
    n_x: int,
    n_groups: int,
    group_centers: list[float],
    dendrogram_ratio: float = 0.15,
) -> pl.DataFrame:
    """Map normalized dendrogram paths into plot coordinates (right side along y-axis)."""
    y_dendrogram_size = n_x * dendrogram_ratio
    leaf_xp = np.arange(n_groups, dtype=float)
    plot_y = np.interp(paths["x"].to_numpy(), leaf_xp, group_centers)
    return pl.DataFrame(
        {
            "x": (n_x - 0.5) + paths["y"].to_numpy() * y_dendrogram_size,
            "y": plot_y,
            "group": paths["group"],
        }
    )
