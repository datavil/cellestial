from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from anndata import AnnData


def _get_dendrogram(data: AnnData, group_by: str) -> tuple[list[str], pl.DataFrame]:
    """
    Get or compute the dendrogram for a group_by key and extract segments.

    Checks ``data.uns[f'dendrogram_{group_by}']``.
    For AnnData, if not present, runs ``scanpy.tl.dendrogram`` on  to compute it.
    Extracts the icoord/dcoord arrays into a Polars DataFrame of segments with normalized positions.
    """
    import numpy as np

    key = f"dendrogram_{group_by}"
    if key not in data.uns:
        import scanpy as sc

        sc.tl.dendrogram(data, groupby=group_by)

    dendro = data.uns[key]
    categories_ordered = list(dendro["categories_ordered"])
    dendro_info = dendro["dendrogram_info"]

    icoord = np.array(dendro_info["icoord"])
    dcoord = np.array(dendro_info["dcoord"])

    # scipy places leaves at 5, 15, 25, ... → normalize to 0, 1, 2, ...
    icoord = (icoord - 5) / 10

    max_height = dcoord.max()
    if max_height > 0:
        dcoord = dcoord / max_height

    x, xend, y, yend = [], [], [], []
    for i in range(len(icoord)):
        for j in range(3):
            x.append(float(icoord[i][j]))
            xend.append(float(icoord[i][j + 1]))
            y.append(float(dcoord[i][j]))
            yend.append(float(dcoord[i][j + 1]))

    segments = pl.DataFrame({"x": x, "xend": xend, "y": y, "yend": yend})
    return categories_ordered, segments


def _get_dendrogram_segment_frame(
    segments: pl.DataFrame,
    *,
    n_x: int,
    n_groups: int,
    group_centers: list[float],
    dendrogram_ratio: float = 0.15,
) -> pl.DataFrame:
    """Map normalized dendrogram segments into plot coordinates (right side along y-axis)."""
    import numpy as np

    y_dendrogram_size = n_x * dendrogram_ratio
    leaf_xp = np.arange(n_groups, dtype=float)
    seg_y = np.interp(segments["x"].to_numpy(), leaf_xp, group_centers)
    seg_yend = np.interp(segments["xend"].to_numpy(), leaf_xp, group_centers)
    return pl.DataFrame(
        {
            "x": (n_x - 0.5) + segments["y"].to_numpy() * y_dendrogram_size,
            "xend": (n_x - 0.5) + segments["yend"].to_numpy() * y_dendrogram_size,
            "y": seg_y,
            "yend": seg_yend,
        }
    )
