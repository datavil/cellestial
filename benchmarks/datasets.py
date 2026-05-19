"""
Lazy dataset registry for the benchmarks suite.

Each entry exposes a `load()` callable that returns a ready-to-plot AnnData.
Loaders are cached in-process so we don't repay the read cost when many cases
share a dataset. Metadata describing the dataset (group_by column, a list of
benchmark gene keys, a continuous obs key, whether it carries spatial info)
is derived on first load and reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

from benchmarks import _console

if TYPE_CHECKING:
    from anndata import AnnData


DatasetKind = Literal["scrna", "spatial"]


@dataclass
class DatasetMetadata:
    """Information extracted from a loaded dataset, used to build cases."""

    name: str
    kind: DatasetKind
    n_cells: int
    n_vars: int
    group_by: str
    second_group_by: str | None
    genes: list[str]
    continuous_obs: str | None
    numeric_obs: list[str]
    has_rank_genes: bool


@dataclass
class DatasetEntry:
    """Registry entry: how to load a dataset and what kind it is."""

    name: str
    kind: DatasetKind
    loader: Callable[[], "AnnData"]
    _data: "AnnData | None" = field(default=None, repr=False)
    _metadata: DatasetMetadata | None = field(default=None, repr=False)

    def load(self) -> "AnnData":
        if self._data is None:
            _console.step(f"loading dataset '{self.name}' ({self.kind})")
            self._data = self.loader()
            _console.kv("n_cells", self._data.n_obs)
            _console.kv("n_vars", self._data.n_vars)
        return self._data

    def metadata(self) -> DatasetMetadata:
        if self._metadata is None:
            data = self.load()
            self._metadata = _describe(self.name, self.kind, data)
        return self._metadata


def _describe(name: str, kind: DatasetKind, data: "AnnData") -> DatasetMetadata:
    """Pick a default group_by, marker genes and continuous obs from `data`."""
    group_by = _pick_group_by(data)
    try:
        second_group_by = _pick_group_by(data, exclude={group_by} if group_by else set())
    except RuntimeError:
        second_group_by = None
    genes = _pick_genes(data, n=200)
    numeric_obs = _list_numeric_obs(data)
    continuous_obs = numeric_obs[0] if numeric_obs else None
    has_rank_genes = "rank_genes_groups" in data.uns
    metadata = DatasetMetadata(
        name=name,
        kind=kind,
        n_cells=int(data.n_obs),
        n_vars=int(data.n_vars),
        group_by=group_by,
        second_group_by=second_group_by,
        genes=genes,
        continuous_obs=continuous_obs,
        numeric_obs=numeric_obs,
        has_rank_genes=has_rank_genes,
    )
    _console.kv("group_by", group_by)
    _console.kv("continuous_obs", continuous_obs or "(none)")
    _console.kv("rank_genes_groups precomputed", has_rank_genes)
    _console.kv("genes available for keys", len(genes))
    return metadata


def _pick_group_by(data: "AnnData", *, exclude: set[str] | None = None) -> str:
    """Choose a sensible categorical column from obs (largest small cardinality)."""
    exclude = exclude or set()
    candidates: list[tuple[int, str]] = []
    preferred = [
        "leiden",
        "louvain",
        "clusters",
        "cluster",
        "cell_type_lvl1",
        "cell_type",
        "celltype",
        "leiden_res_0.50",
        "leiden_res_0.02",
    ]
    for name in preferred:
        if name in exclude:
            continue
        if name in data.obs.columns:
            return name
    for name in data.obs.columns:
        if name in exclude:
            continue
        column = data.obs[name]
        # categorical with a useful cardinality (2..40)
        if hasattr(column, "cat"):
            n_categories = len(column.cat.categories)
        elif column.dtype == object:
            n_categories = column.nunique()
        else:
            continue
        if 2 <= n_categories <= 40:
            candidates.append((n_categories, name))
    if not candidates:
        message = (
            f"No usable categorical obs column for dataset (have {list(data.obs.columns)})"
        )
        raise RuntimeError(message)
    candidates.sort()
    return candidates[0][1]


def _pick_genes(data: "AnnData", *, n: int) -> list[str]:
    """Return up to `n` benchmark genes: HVG-ranked if available, else top-mean."""
    var = data.var
    if "highly_variable_rank" in var.columns:
        ordered = var.sort_values("highly_variable_rank")
        ordered = ordered.dropna(subset=["highly_variable_rank"])
        names = list(ordered.index[:n])
        if names:
            return names
    if "highly_variable" in var.columns:
        mask = var["highly_variable"].astype(bool).to_numpy()
        names = list(var.index[mask][:n])
        if names:
            return names
    # fallback: top-N by mean expression
    matrix = data.X
    if hasattr(matrix, "toarray"):
        mean_expression = np.asarray(matrix.mean(axis=0)).ravel()
    else:
        mean_expression = np.asarray(matrix).mean(axis=0)
    order = np.argsort(-mean_expression)[:n]
    return [str(name) for name in data.var_names[order]]


def _list_numeric_obs(data: "AnnData") -> list[str]:
    """Return numeric obs column names, preferred ones first."""
    import pandas as pd

    preferred = ["total_counts", "n_genes_by_counts", "pct_counts_mt", "n_counts"]
    found: list[str] = []
    for name in preferred:
        if name in data.obs.columns and pd.api.types.is_numeric_dtype(data.obs[name]):
            found.append(name)
    for name in data.obs.columns:
        if name in found:
            continue
        if pd.api.types.is_numeric_dtype(data.obs[name]):
            found.append(name)
    return found


# --- Loaders -----------------------------------------------------------------


def _load_cl_pbmc3k() -> "AnnData":
    import cellestial as cl

    return cl.datasets.pbmc3k(cache_directory="data")


def _load_cl_pancreas() -> "AnnData":
    import cellestial as cl

    return cl.datasets.pancreas(cache_directory="data")


def _load_cl_lymph_node() -> "AnnData":
    import cellestial as cl

    return cl.datasets.human_lymph_node(cache_directory="data")


def _load_sc_pbmc68k() -> "AnnData":
    import scanpy as sc

    return sc.datasets.pbmc68k_reduced()


def _load_sc_paul15() -> "AnnData":
    import scanpy as sc

    data = sc.datasets.paul15()
    # paul15 ships raw counts; normalize so plots are meaningful and HVGs exist.
    sc.pp.normalize_total(data)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=2000)
    sc.tl.pca(data)
    sc.pp.neighbors(data)
    sc.tl.umap(data)
    sc.tl.tsne(data)
    sc.tl.rank_genes_groups(data, groupby="paul15_clusters", method="wilcoxon")
    return data


def _load_sq_visium_hne() -> "AnnData":
    import squidpy as sq

    return sq.datasets.visium_hne_adata()


_REGISTRY: dict[str, DatasetEntry] = {
    "pbmc3k": DatasetEntry("pbmc3k", "scrna", _load_cl_pbmc3k),
    "pancreas": DatasetEntry("pancreas", "scrna", _load_cl_pancreas),
    "pbmc68k_reduced": DatasetEntry("pbmc68k_reduced", "scrna", _load_sc_pbmc68k),
    "paul15": DatasetEntry("paul15", "scrna", _load_sc_paul15),
    "human_lymph_node": DatasetEntry("human_lymph_node", "spatial", _load_cl_lymph_node),
    "visium_hne": DatasetEntry("visium_hne", "spatial", _load_sq_visium_hne),
}


def available() -> list[str]:
    return list(_REGISTRY)


def get(name: str) -> DatasetEntry:
    if name not in _REGISTRY:
        message = f"unknown dataset {name!r}; available: {available()}"
        raise KeyError(message)
    return _REGISTRY[name]


def resolve(names: list[str] | None) -> list[DatasetEntry]:
    if not names:
        return list(_REGISTRY.values())
    return [get(name) for name in names]
