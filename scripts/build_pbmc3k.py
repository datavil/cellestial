#!/usr/bin/env python
"""
Build the bundled `pbmc3k` dataset: the 10x Genomics PBMC 3k, preprocessed.

The raw counts and the cell type labels come from scanpy's copies of the dataset
(`scanpy.datasets.pbmc3k` and `scanpy.datasets.pbmc3k_processed`). Processing
follows scanpy's "Preprocessing and clustering 3k PBMCs" tutorial, with one
difference: the tutorial keeps only the scaled highly variable genes in the main
matrix, while this file keeps every gene log-normalized there, so any gene can be
plotted. Regression and scaling run on a copy and only feed the PCA.

Output: `data/pbmc3k_10x_pped.h5ad` with
- `X`: log1p of counts normalized to 10,000 per cell, all genes that pass QC;
- `raw`: the integer counts of the same cells and genes;
- `obs["cell_type"]`: the tutorial's annotation, matched by barcode;
- `obs["leiden"]`, QC metrics, `obsm` with `X_pca`, `X_umap`, `X_tsne`.

Usage
-----
    poetry run python scripts/build_pbmc3k.py
"""

from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy import sparse

DATA_DIRECTORY = Path("data")
OUTPUT_PATH = DATA_DIRECTORY / "pbmc3k_10x_pped.h5ad"
LABEL_KEY = "cell_type"


def build() -> AnnData:
    """
    Preprocess the raw PBMC 3k counts and attach the tutorial's cell types.

    Returns
    -------
    AnnData
        The preprocessed dataset.

    Raises
    ------
    RuntimeError
        If the cells left after QC differ from the annotated cells of the tutorial.
    """
    sc.settings.datasetdir = DATA_DIRECTORY
    data = sc.datasets.pbmc3k()
    annotated = sc.datasets.pbmc3k_processed()
    data.var_names_make_unique()

    sc.pp.filter_cells(data, min_genes=200)
    sc.pp.filter_genes(data, min_cells=3)
    data.var["mt"] = data.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(data, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    data = data[(data.obs["n_genes_by_counts"] < 2500) & (data.obs["pct_counts_mt"] < 5)].copy()

    if set(data.obs_names) != set(annotated.obs_names):
        message = (
            f"QC kept {data.n_obs} cells but the tutorial annotates {annotated.n_obs}; "
            "the filters no longer match the tutorial."
        )
        raise RuntimeError(message)
    data.obs[LABEL_KEY] = annotated.obs.loc[data.obs_names, "louvain"].astype("category")

    data.raw = data.copy()
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, min_mean=0.0125, max_mean=3, min_disp=0.5)

    scaled = data[:, data.var["highly_variable"]].copy()
    sc.pp.regress_out(scaled, ["total_counts", "pct_counts_mt"])
    sc.pp.scale(scaled, max_value=10)
    sc.tl.pca(scaled, svd_solver="arpack")
    sc.pp.neighbors(scaled, n_neighbors=10, n_pcs=40)
    sc.tl.umap(scaled)
    sc.tl.tsne(scaled, n_pcs=40)
    sc.tl.leiden(scaled, flavor="igraph", n_iterations=2, directed=False)

    for key in ("X_pca", "X_umap", "X_tsne"):
        data.obsm[key] = scaled.obsm[key]
    for key in ("connectivities", "distances"):
        data.obsp[key] = scaled.obsp[key]
    for key in ("pca", "neighbors", "umap", "tsne", "leiden"):
        data.uns[key] = scaled.uns[key]
    data.obs["leiden"] = scaled.obs["leiden"]
    data.X = sparse.csr_matrix(data.X, dtype=np.float32)
    return data


def main() -> None:
    """Build the dataset, check it, and write it."""
    data = build()
    counts = data.raw.X
    assert np.allclose(counts.data, np.round(counts.data)), "raw must hold integer counts"
    assert data.obs[LABEL_KEY].notna().all(), "every cell needs a cell type"
    data.write_h5ad(OUTPUT_PATH, compression="gzip")
    print(f"wrote {OUTPUT_PATH}: {data.n_obs} cells x {data.n_vars} genes")
    print(data.obs[LABEL_KEY].value_counts().to_string())


if __name__ == "__main__":
    main()
