"""Dataset loaders for cellestial benchmarks.

Goals:
- Provide a few real, preprocessed scanpy datasets for realism.
- Provide a scalable synthetic dataset (`blobs`) with UMAP/PCA/cluster labels
  patched in, so we can sweep cell counts without paying for real dimred.

The patched UMAP/PCA coordinates are random — we are benchmarking the
*plotting pipeline*, not dimensionality reduction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scanpy as sc
from anndata import AnnData


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    adata: AnnData
    cluster_key: str
    gene_keys: list[str]

    @property
    def n_cells(self) -> int:
        return self.adata.n_obs

    @property
    def n_vars(self) -> int:
        return self.adata.n_vars


def _pick_genes(adata: AnnData, n: int = 10) -> list[str]:
    """Pick the top-n highest-mean genes as representative markers."""
    means = np.asarray(adata.X.mean(axis=0)).ravel()
    top = np.argsort(means)[::-1][:n]
    return [str(g) for g in adata.var_names[top]]


def load_pbmc68k_reduced() -> DatasetSpec:
    adata = sc.datasets.pbmc68k_reduced()
    cluster_key = "bulk_labels"
    return DatasetSpec(
        name="pbmc68k_reduced",
        adata=adata,
        cluster_key=cluster_key,
        gene_keys=_pick_genes(adata, n=10),
    )


def load_pbmc3k_processed() -> DatasetSpec:
    adata = sc.datasets.pbmc3k_processed()
    cluster_key = "louvain"
    return DatasetSpec(
        name="pbmc3k_processed",
        adata=adata,
        cluster_key=cluster_key,
        gene_keys=_pick_genes(adata, n=10),
    )


def load_pbmc3k() -> DatasetSpec:
    """Raw pbmc3k (~2.7k cells, no UMAP/clusters) — patch embeddings."""
    adata = sc.datasets.pbmc3k()
    _patch_synthetic_embeddings(adata, n_centers=8, seed=3)
    return DatasetSpec(
        name="pbmc3k",
        adata=adata,
        cluster_key="cluster",
        gene_keys=_pick_genes(adata, n=10),
    )


def load_paul15() -> DatasetSpec:
    """Paul et al. 2015 (~2.7k cells); has `paul15_clusters`, patch UMAP only."""
    adata = sc.datasets.paul15()
    if "X_umap" not in adata.obsm:
        rng = np.random.default_rng(15)
        labels = adata.obs["paul15_clusters"].astype("category").cat.codes.to_numpy()
        n_centers = int(labels.max()) + 1
        centers = rng.normal(0, 5, size=(n_centers, 2))
        adata.obsm["X_umap"] = (
            centers[labels] + rng.normal(0, 0.8, size=(adata.n_obs, 2))
        ).astype(np.float32)
    return DatasetSpec(
        name="paul15",
        adata=adata,
        cluster_key="paul15_clusters",
        gene_keys=_pick_genes(adata, n=10),
    )


def load_ebi_expression_atlas(
    accession: str = "E-MTAB-4888",
) -> DatasetSpec:
    """EBI Expression Atlas dataset (downloads on first run).

    No UMAP/clusters in the raw download — we patch synthetic embeddings so
    we're still benchmarking plotting, not preprocessing.
    """
    adata = sc.datasets.ebi_expression_atlas(accession)
    _patch_synthetic_embeddings(adata, n_centers=8, seed=42)
    return DatasetSpec(
        name=f"ebi_{accession}",
        adata=adata,
        cluster_key="cluster",
        gene_keys=_pick_genes(adata, n=10),
    )


def _patch_synthetic_embeddings(
    adata: AnnData,
    *,
    n_centers: int,
    seed: int,
) -> None:
    """Attach random UMAP/PCA coords and integer cluster labels in-place.

    Cells are grouped into `n_centers` clusters; UMAP/PCA coords are drawn
    around per-cluster centers so the plots don't degenerate to a single blob.
    """
    rng = np.random.default_rng(seed)
    n = adata.n_obs
    labels = rng.integers(0, n_centers, size=n)

    centers_2d = rng.normal(0, 5, size=(n_centers, 2))
    umap = centers_2d[labels] + rng.normal(0, 0.8, size=(n, 2))

    centers_50d = rng.normal(0, 5, size=(n_centers, 50))
    pca = centers_50d[labels] + rng.normal(0, 1.0, size=(n, 50))

    adata.obsm["X_umap"] = umap.astype(np.float32)
    adata.obsm["X_pca"] = pca.astype(np.float32)
    adata.obs["cluster"] = [f"c{label}" for label in labels]
    adata.obs["cluster"] = adata.obs["cluster"].astype("category")


def load_blobs(
    n_observations: int,
    *,
    n_variables: int = 200,
    n_centers: int = 8,
    seed: int = 0,
) -> DatasetSpec:
    """Synthetic scalable dataset with patched UMAP/PCA/cluster metadata."""
    adata = sc.datasets.blobs(
        n_variables=n_variables,
        n_centers=n_centers,
        n_observations=n_observations,
    )
    # scanpy's blobs stores X as a numpy array; ensure gene names exist.
    if adata.var_names[0].startswith("0"):
        adata.var_names = [f"gene_{i}" for i in range(adata.n_vars)]
    # blobs already has an integer 'blobs' obs; we overwrite with our own
    # cluster column for consistency with other datasets.
    _patch_synthetic_embeddings(adata, n_centers=n_centers, seed=seed)
    return DatasetSpec(
        name=f"blobs_{n_observations}",
        adata=adata,
        cluster_key="cluster",
        gene_keys=_pick_genes(adata, n=10),
    )


REAL_LOADERS = {
    "pbmc68k_reduced": load_pbmc68k_reduced,
    "pbmc3k_processed": load_pbmc3k_processed,
    "pbmc3k": load_pbmc3k,
    "paul15": load_paul15,
    "ebi_expression_atlas": load_ebi_expression_atlas,
}
