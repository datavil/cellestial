from __future__ import annotations

import numpy as np
import polars as pl
from anndata import AnnData

from cellestial.util.errors import KeyNotFoundError


def _get_dendrogram(
    data: AnnData,
    group_by: str,
    *,
    use_key: str | None = None,
) -> tuple[list[str], pl.DataFrame]:
    """Get or compute the dendrogram for `group_by` and extract paths."""
    if isinstance(data, AnnData):
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
