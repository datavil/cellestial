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
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from anndata import AnnData

LOCAL_DATA_DIR = Path("~/data").expanduser()


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
        gene_keys=_pick_genes(adata, n=40),
    )


def load_pbmc3k_processed() -> DatasetSpec:
    adata = sc.datasets.pbmc3k_processed()
    cluster_key = "louvain"
    return DatasetSpec(
        name="pbmc3k_processed",
        adata=adata,
        cluster_key=cluster_key,
        gene_keys=_pick_genes(adata, n=40),
    )


def load_pbmc3k() -> DatasetSpec:
    """Raw pbmc3k (~2.7k cells, no UMAP/clusters) — patch embeddings."""
    adata = sc.datasets.pbmc3k()
    _patch_synthetic_embeddings(adata, n_centers=8, seed=3)
    return DatasetSpec(
        name="pbmc3k",
        adata=adata,
        cluster_key="cluster",
        gene_keys=_pick_genes(adata, n=40),
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
        gene_keys=_pick_genes(adata, n=40),
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
        gene_keys=_pick_genes(adata, n=40),
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
        gene_keys=_pick_genes(adata, n=40),
    )


REAL_LOADERS = {
    "pbmc68k_reduced": load_pbmc68k_reduced,
    "pbmc3k_processed": load_pbmc3k_processed,
    "pbmc3k": load_pbmc3k,
    "paul15": load_paul15,
    "ebi_expression_atlas": load_ebi_expression_atlas,
}


def _autodetect_cluster_key(adata: AnnData) -> str:
    """Pick the categorical obs column with the most reasonable group count."""
    candidates: list[tuple[str, int]] = []
    for column_name in adata.obs.columns:
        series = adata.obs[column_name]
        if series.dtype.name == "category" or series.dtype == object:
            n_unique = series.nunique(dropna=True)
            if 2 <= n_unique <= 100:
                candidates.append((column_name, n_unique))
    if not candidates:
        raise ValueError("No suitable cluster column found in adata.obs")
    preferred = ("cell_type", "celltype", "cluster", "clusters", "leiden", "louvain")
    for name in preferred:
        for column_name, _ in candidates:
            if column_name.lower() == name:
                return column_name
    candidates.sort(key=lambda item: item[1])
    return candidates[0][0]


def load_local_h5ad(
    path: Path | str,
    *,
    name: str | None = None,
    cluster_key: str | None = None,
    n_genes: int = 40,
    subsample: int | None = None,
    seed: int = 0,
) -> DatasetSpec:
    """Load a local .h5ad file and wrap it as a DatasetSpec.

    Missing UMAP/PCA embeddings are patched with synthetic ones so we remain
    focused on benchmarking the plotting pipeline. ``subsample`` caps the
    number of cells (random without replacement) to keep iteration quick on
    very large atlases.
    """
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Local h5ad not found: {resolved}")
    adata = ad.read_h5ad(resolved)
    if subsample is not None and adata.n_obs > subsample:
        rng = np.random.default_rng(seed)
        indices = rng.choice(adata.n_obs, size=subsample, replace=False)
        indices.sort()
        adata = adata[indices].to_memory() if hasattr(adata, "to_memory") else adata[indices].copy()
    if cluster_key is None:
        cluster_key = _autodetect_cluster_key(adata)
    if adata.obs[cluster_key].dtype.name != "category":
        adata.obs[cluster_key] = adata.obs[cluster_key].astype("category")
    if "X_umap" not in adata.obsm or "X_pca" not in adata.obsm:
        n_centers = max(2, int(adata.obs[cluster_key].nunique()))
        _patch_synthetic_embeddings(adata, n_centers=min(n_centers, 20), seed=7)
    return DatasetSpec(
        name=name or resolved.stem,
        adata=adata,
        cluster_key=cluster_key,
        gene_keys=_pick_genes(adata, n=n_genes),
    )


def load_breast_cancer_atlas(*, subsample: int | None = None) -> DatasetSpec:
    return load_local_h5ad(
        LOCAL_DATA_DIR / "breast_cancer_atlas.h5ad",
        name="breast_cancer_atlas" if subsample is None else f"breast_cancer_atlas_{subsample}",
        subsample=subsample,
    )


LOCAL_LOADERS = {
    "breast_cancer_atlas": load_breast_cancer_atlas,
}
