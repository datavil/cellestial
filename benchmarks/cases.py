"""
Case registry: pairs of (cellestial fn, scanpy fn) and their parameter grids.

Each `Case` describes how to call both libraries for one plot type, what kinds
of datasets it supports (scrna / spatial), and the list of parameter dicts it
should be evaluated across (data day-to-day vs high-coverage).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from anndata import AnnData

    from benchmarks.datasets import DatasetMetadata


KeyKind = Literal[
    "gene",
    "category",
    "obs_obs",
    "gene_gene",
    "spatial_gene",
    "n_top",
    "n_genes",
    "n_keys",
]


@dataclass
class Params:
    """Concrete parameters resolved against a dataset for one run."""

    n_keys: int
    key_kind: KeyKind
    slug: str
    extras: dict[str, object]


@dataclass
class Case:
    """One head-to-head benchmark case."""

    name: str
    cellestial_fn: Callable[[AnnData, Params, DatasetMetadata], object]
    scanpy_fn: Callable[[AnnData, Params, DatasetMetadata], object]
    expand: Callable[[DatasetMetadata], list[Params]]
    supports: tuple[Literal["scrna", "spatial"], ...] = ("scrna",)


# --- Helpers to resolve params from dataset metadata -------------------------


def _expand_dim_reduction(metadata: DatasetMetadata) -> list[Params]:
    """Two variants: gene (continuous) and category (discrete)."""
    results: list[Params] = []
    if metadata.genes:
        results.append(
            Params(
                n_keys=1,
                key_kind="gene",
                slug="gene",
                extras={"key": metadata.genes[0]},
            )
        )
    results.append(
        Params(
            n_keys=1,
            key_kind="category",
            slug="category",
            extras={"key": metadata.group_by},
        )
    )
    return results


def _expand_n_keys(
    metadata: DatasetMetadata,
    counts: tuple[int, ...],
    *,
    kind: KeyKind = "n_genes",
) -> list[Params]:
    """Vary number of genes selected from metadata.genes."""
    results: list[Params] = []
    for count in counts:
        if count > len(metadata.genes):
            continue
        keys = metadata.genes[:count]
        results.append(
            Params(
                n_keys=count,
                key_kind=kind,
                slug=f"n{count}",
                extras={"keys": keys, "group_by": metadata.group_by},
            )
        )
    return results


def _expand_violin_single(metadata: DatasetMetadata) -> list[Params]:
    """Single-key violin on a gene split by group_by."""
    if not metadata.genes:
        return []
    return [
        Params(
            n_keys=1,
            key_kind="gene",
            slug="gene",
            extras={"key": metadata.genes[0], "group_by": metadata.group_by},
        )
    ]


def _expand_highest(metadata: DatasetMetadata) -> list[Params]:
    results: list[Params] = []
    for top in (10, 30, 60, 100):
        if top > metadata.n_vars:
            continue
        results.append(
            Params(n_keys=top, key_kind="n_top", slug=f"n{top}", extras={"n": top})
        )
    return results


def _expand_markers(metadata: DatasetMetadata) -> list[Params]:
    if not metadata.has_rank_genes:
        return []
    results: list[Params] = []
    for n in (5, 20, 50):
        results.append(
            Params(n_keys=n, key_kind="n_genes", slug=f"n{n}", extras={"n_genes": n})
        )
    return results


def _expand_scatter_obs_obs(metadata: DatasetMetadata) -> list[Params]:
    if len(metadata.numeric_obs) < 2:
        return []
    first, second = metadata.numeric_obs[0], metadata.numeric_obs[1]
    return [
        Params(
            n_keys=1,
            key_kind="obs_obs",
            slug="obs_obs",
            extras={"x": first, "y": second},
        )
    ]


def _expand_scatter_gene_gene(metadata: DatasetMetadata) -> list[Params]:
    if len(metadata.genes) < 2:
        return []
    return [
        Params(
            n_keys=1,
            key_kind="gene_gene",
            slug="gene_gene",
            extras={"x": metadata.genes[0], "y": metadata.genes[1]},
        )
    ]


def _expand_spatial(metadata: DatasetMetadata) -> list[Params]:
    if not metadata.genes:
        return []
    results: list[Params] = []
    for count in (1, 5, 20):
        if count > len(metadata.genes):
            continue
        results.append(
            Params(
                n_keys=count,
                key_kind="spatial_gene",
                slug=f"n{count}",
                extras={"keys": metadata.genes[:count]},
            )
        )
    return results


# --- Cellestial callers ------------------------------------------------------


def _cl_umap(data, params, metadata):
    import cellestial as cl

    return cl.umap(data, key=params.extras["key"])


def _cl_tsne(data, params, metadata):
    import cellestial as cl

    return cl.tsne(data, key=params.extras["key"])


def _cl_pca(data, params, metadata):
    import cellestial as cl

    return cl.pca(data, key=params.extras["key"])


def _cl_expression(data, params, metadata):
    import cellestial as cl

    return cl.expression(data, key=params.extras["key"])


def _cl_dotplot(data, params, metadata):
    import cellestial as cl

    return cl.dotplot(data, keys=params.extras["keys"], group_by=params.extras["group_by"])


def _cl_heatmap(data, params, metadata):
    import cellestial as cl

    return cl.heatmap(data, keys=params.extras["keys"], group_by=params.extras["group_by"])


def _cl_matrixplot(data, params, metadata):
    import cellestial as cl

    return cl.matrixplot(data, keys=params.extras["keys"], group_by=params.extras["group_by"])


def _cl_stacked_violin(data, params, metadata):
    import cellestial as cl

    return cl.stacked_violin(
        data, keys=params.extras["keys"], group_by=params.extras["group_by"]
    )


def _cl_violin(data, params, metadata):
    import cellestial as cl

    return cl.violin(
        data, key=params.extras["key"], group_by=params.extras["group_by"]
    )


def _cl_highest(data, params, metadata):
    import cellestial as cl

    return cl.highest_expressed_genes(data, n=params.extras["n"])


def _cl_markers(data, params, metadata):
    import cellestial as cl

    return cl.markers(data, n_genes=params.extras["n_genes"])


def _cl_scatter_obs_obs(data, params, metadata):
    from lets_plot import aes

    import cellestial as cl

    return cl.scatter(
        data, mapping=aes(x=params.extras["x"], y=params.extras["y"]), axis=0
    )


def _cl_xyplot_gene_gene(data, params, metadata):
    import cellestial as cl

    return cl.xyplot(data, x=params.extras["x"], y=params.extras["y"])


def _cl_spatial(data, params, metadata):
    import cellestial as cl

    keys = params.extras["keys"]
    if params.n_keys == 1:
        return cl.spatial(data, key=keys[0])
    return cl.spatials(data, keys=keys)


# --- Scanpy callers ----------------------------------------------------------
#
# Best-effort phase split: call returns either a Figure or an Axes. We treat
# the call as the "construct" phase, and a subsequent fig.savefig() as the
# "render" phase. For functions whose return is an Axes (or dict of Axes),
# the runner pulls the underlying figure via `figure_of`.


def _sc_embedding(basis: str, data, params, metadata):
    import scanpy as sc

    return sc.pl.embedding(data, basis=basis, color=params.extras["key"], show=False, return_fig=True)


def _sc_umap(data, params, metadata):
    return _sc_embedding("umap", data, params, metadata)


def _sc_tsne(data, params, metadata):
    return _sc_embedding("tsne", data, params, metadata)


def _sc_pca(data, params, metadata):
    return _sc_embedding("pca", data, params, metadata)


def _sc_dotplot(data, params, metadata):
    import scanpy as sc

    return sc.pl.dotplot(
        data,
        var_names=params.extras["keys"],
        groupby=params.extras["group_by"],
        show=False,
        return_fig=True,
    )


def _sc_heatmap(data, params, metadata):
    import scanpy as sc

    return sc.pl.heatmap(
        data,
        var_names=params.extras["keys"],
        groupby=params.extras["group_by"],
        show=False,
    )


def _sc_matrixplot(data, params, metadata):
    import scanpy as sc

    return sc.pl.matrixplot(
        data,
        var_names=params.extras["keys"],
        groupby=params.extras["group_by"],
        show=False,
        return_fig=True,
    )


def _sc_stacked_violin(data, params, metadata):
    import scanpy as sc

    return sc.pl.stacked_violin(
        data,
        var_names=params.extras["keys"],
        groupby=params.extras["group_by"],
        show=False,
        return_fig=True,
    )


def _sc_violin(data, params, metadata):
    import scanpy as sc

    return sc.pl.violin(
        data,
        keys=params.extras["key"],
        groupby=params.extras["group_by"],
        show=False,
    )


def _sc_highest(data, params, metadata):
    import scanpy as sc

    return sc.pl.highest_expr_genes(data, n_top=params.extras["n"], show=False)


def _sc_markers(data, params, metadata):
    import scanpy as sc

    return sc.pl.rank_genes_groups(data, n_genes=params.extras["n_genes"], show=False)


def _sc_scatter_obs_obs(data, params, metadata):
    import scanpy as sc

    return sc.pl.scatter(data, x=params.extras["x"], y=params.extras["y"], show=False)


def _sc_scatter_gene_gene(data, params, metadata):
    import scanpy as sc

    return sc.pl.scatter(data, x=params.extras["x"], y=params.extras["y"], show=False)


def _sc_spatial(data, params, metadata):
    import scanpy as sc

    keys = params.extras["keys"]
    color = keys[0] if params.n_keys == 1 else keys
    return sc.pl.spatial(data, color=color, show=False, return_fig=True)


# --- Registry ----------------------------------------------------------------


CASES: list[Case] = [
    Case("umap", _cl_umap, _sc_umap, _expand_dim_reduction),
    Case("tsne", _cl_tsne, _sc_tsne, _expand_dim_reduction),
    Case("pca", _cl_pca, _sc_pca, _expand_dim_reduction),
    Case(
        "expression",
        _cl_expression,
        _sc_umap,
        lambda metadata: [
            Params(n_keys=1, key_kind="gene", slug="gene", extras={"key": metadata.genes[0]})
        ]
        if metadata.genes
        else [],
    ),
    Case(
        "dotplot",
        _cl_dotplot,
        _sc_dotplot,
        lambda metadata: _expand_n_keys(metadata, (5, 20, 50, 100, 200)),
    ),
    Case(
        "heatmap",
        _cl_heatmap,
        _sc_heatmap,
        lambda metadata: _expand_n_keys(metadata, (5, 20, 50, 100, 200)),
    ),
    Case(
        "matrixplot",
        _cl_matrixplot,
        _sc_matrixplot,
        lambda metadata: _expand_n_keys(metadata, (5, 20, 50, 100, 200)),
    ),
    Case(
        "stacked_violin",
        _cl_stacked_violin,
        _sc_stacked_violin,
        lambda metadata: _expand_n_keys(metadata, (5, 20, 50, 100)),
    ),
    Case("violin", _cl_violin, _sc_violin, _expand_violin_single),
    Case("highest_expressed", _cl_highest, _sc_highest, _expand_highest),
    Case("markers", _cl_markers, _sc_markers, _expand_markers),
    Case("scatter_obs_obs", _cl_scatter_obs_obs, _sc_scatter_obs_obs, _expand_scatter_obs_obs),
    Case("xyplot_gene_gene", _cl_xyplot_gene_gene, _sc_scatter_gene_gene, _expand_scatter_gene_gene),
    Case("spatial", _cl_spatial, _sc_spatial, _expand_spatial, supports=("spatial",)),
]


def available() -> list[str]:
    return [case.name for case in CASES]


def resolve(names: list[str] | None) -> list[Case]:
    if not names:
        return list(CASES)
    requested = set(names)
    unknown = requested - set(available())
    if unknown:
        message = f"unknown cases {sorted(unknown)}; available: {available()}"
        raise KeyError(message)
    return [case for case in CASES if case.name in requested]
