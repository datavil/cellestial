from __future__ import annotations

import polars as pl
from anndata import AnnData


# AI-GENERATED: Claude 4.6
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line, edge cases unverified
def _get_dendrogram(data: AnnData, group_by: str) -> tuple[list[str], pl.DataFrame]:
    """
    Get or compute the dendrogram for a group_by key and extract paths.

    Checks ``data.uns[f'dendrogram_{group_by}']``.
    For AnnData, if not present, runs ``scanpy.tl.dendrogram`` to compute it.
    Extracts the icoord/dcoord arrays into a Polars DataFrame of paths with normalized positions.
    """
    import numpy as np

    if isinstance(data, AnnData):
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


# AI-GENERATED: Claude 4.6
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line, edge cases unverified
def _get_dendrogram_path_frame(
    paths: pl.DataFrame,
    *,
    n_x: int,
    n_groups: int,
    group_centers: list[float],
    dendrogram_ratio: float = 0.15,
) -> pl.DataFrame:
    """Map normalized dendrogram paths into plot coordinates (right side along y-axis)."""
    import numpy as np

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
