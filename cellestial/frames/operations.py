from __future__ import annotations

import numpy as np
import polars as pl
from anndata import AnnData
from polars import DataFrame
from scipy.sparse import issparse

from cellestial.util.errors import UnsupportedDataTypeError


def _highest_expressed_genes_frame(
    data: AnnData,
    n: int = 10,
) -> DataFrame:
    """Get the top n highest expressed genes by mean percentage across all cells."""
    if isinstance(data, AnnData):
        # VALIDATE: inputs
        if n > data.n_vars:
            msg = f"Requested n={n} genes, but only {data.n_vars} genes available in data."
            raise ValueError(msg)
        X = data.X
        # matmul and fancy indexing below require an in-memory ndarray/sparse;
        # fail fast on backed/dask/zappy rather than crashing deep in the math.
        if not (isinstance(X, np.ndarray) or issparse(X)):
            msg = (
                f"Unsupported `X` type for this operation: {type(X).__name__}. "
                "Load AnnData in non-backed mode or materialize `X` manually."
            )
            raise UnsupportedDataTypeError(msg)

        # BUILD: per-cell normalizer so each cell's expression sums to 100
        if issparse(X):
            row_sums = np.asarray(X.sum(axis=1)).ravel()
        else:
            row_sums = X.sum(axis=1)
        row_sums[row_sums == 0] = 1  # guard cells with zero total to avoid div-by-zero
        inv_rows = 100.0 / row_sums

        # RANK: gene means via a single matvec instead of materializing the
        # full normalized matrix. Identity: mean(X / row_sums * 100, axis=0)
        # equals (inv_rows @ X) / n_cells, but uses O(n_vars) memory instead of O(n_obs * n_vars).
        mean_percent = np.asarray(inv_rows @ X).ravel() / X.shape[0]
        top_idx = np.argsort(mean_percent)[::-1][:n]
        genes = data.var_names[top_idx].tolist()

        # EXTRACT: normalize only the top n columns (small n_obs * n slice),
        # skipping the full n_obs * n_vars materialization the naive path would do.
        X_top = X[:, top_idx]
        if issparse(X_top):
            X_top = X_top.toarray()
        X_top_normalized = X_top * inv_rows[:, np.newaxis]

        # BUILD: polars frame keyed by gene name
        frame = pl.DataFrame({gene: X_top_normalized[:, i] for i, gene in enumerate(genes)})
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    return frame


def _pca_variance_frame(
    data: AnnData,
    n_pcs: int | None = None,
    component_column: str = "Principal Component",
    variance_column: str = "Variance Ratio",
    use_key: str = "pca",
) -> DataFrame:
    """Get PCA variance ratio per principal component as a polars DataFrame."""
    if isinstance(data, AnnData):
        pca_uns = data.uns.get(use_key)
        if pca_uns is None or "variance_ratio" not in pca_uns:
            msg = f"PCA variance ratio not found under `{use_key}`. Run PCA first."
            raise ValueError(msg)
        variance_ratio = np.asarray(pca_uns["variance_ratio"]).ravel()
        n_available = variance_ratio.size
        if n_pcs is not None:
            if n_pcs > n_available:
                msg = (
                    f"Requested n_pcs={n_pcs} components, "
                    f"but only {n_available} components available."
                )
                raise ValueError(msg)
            variance_ratio = variance_ratio[:n_pcs]
        frame = pl.DataFrame(
            {
                component_column: np.arange(1, variance_ratio.size + 1, dtype=np.int64),
                variance_column: variance_ratio,
            }
        )
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    return frame
