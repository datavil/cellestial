from __future__ import annotations

import numpy as np
import polars as pl
from anndata import AnnData
from polars import DataFrame
from scipy.sparse import issparse

from cellestial.frames.build import anndata_variable_columns
from cellestial.util.errors import UnsupportedDataTypeError


def _highest_expressed_genes_frame(
    data: AnnData,
    n: int = 10,
) -> DataFrame:
    """Get the top n highest expressed genes by mean percentage across all cells."""
    if isinstance(data, AnnData):
        if n > data.n_vars:
            msg = f"Requested n={n} genes, but only {data.n_vars} genes available in data."
            raise ValueError(msg)
        X = data.X
        # normalize each cell to sum to 100 (percentage)
        if issparse(X):
            row_sums = np.array(X.sum(axis=1)).ravel()
            row_sums[row_sums == 0] = 1
            X_normalized = X.multiply(100 / row_sums[:, np.newaxis])
            mean_percent = np.array(X_normalized.mean(axis=0)).ravel()
        else:
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_normalized = X / row_sums * 100
            mean_percent = X_normalized.mean(axis=0)

        top_idx = np.argsort(mean_percent)[::-1][:n]
        genes = data.var_names[top_idx].tolist()
        if issparse(X_normalized):
            X_normalized = X_normalized.tocsr()
        norm_data = AnnData(X=X_normalized, var=data.var, obs=data.obs)
        frame = pl.DataFrame(anndata_variable_columns(norm_data, keys=genes, column_names=[]))
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
