"""Benchmark cases: paired cellestial vs scanpy plot operations.

Each case exposes two phases:
- ``construct``: build the plot object (cellestial PlotSpec / matplotlib Figure).
- ``render``: serialize to SVG (to_svg() for lets-plot, fig.savefig for matplotlib).

The split is intentional: cellestial's pre-render path (Polars frame build +
ggplot construction) is what we claim is fast; SVG rendering is a separate
cost that goes through a different engine per framework.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

import cellestial as cl

matplotlib.use("Agg")


@dataclass
class Case:
    name: str
    construct_cellestial: Callable[[AnnData, dict], Any]
    construct_scanpy: Callable[[AnnData, dict], Any]


def _render_lets_plot(plot) -> None:
    plot.to_svg()


def _render_lets_plot_html(plot) -> None:
    plot.to_html()


def _render_matplotlib(fig) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)


# ---------- cellestial constructors ----------
# sampling='none' disables lets-plot's default point sampling (~100k cap) so
# large datasets are fully rendered, matching scanpy's no-sampling behavior.

def _cl_umap_gene(adata: AnnData, ctx: dict):
    return cl.expression(adata, key=ctx["gene"], sampling="none")


def _cl_umap_cluster(adata: AnnData, ctx: dict):
    return cl.umap(adata, key=ctx["cluster_key"], sampling="none")


def _cl_dotplot(adata: AnnData, ctx: dict):
    return cl.dotplot(adata, keys=ctx["genes"], group_by=ctx["cluster_key"])


def _cl_heatmap(adata: AnnData, ctx: dict):
    return cl.heatmap(adata, keys=ctx["genes"], group_by=ctx["cluster_key"])


def _cl_violin(adata: AnnData, ctx: dict):
    return cl.violins(adata, keys=ctx["genes"][:4])



# ---------- scanpy constructors ----------

def _sc_umap_gene(adata: AnnData, ctx: dict):
    return sc.pl.umap(adata, color=ctx["gene"], show=False, return_fig=True)


def _sc_umap_cluster(adata: AnnData, ctx: dict):
    return sc.pl.umap(adata, color=ctx["cluster_key"], show=False, return_fig=True)


def _sc_dotplot(adata: AnnData, ctx: dict):
    dp = sc.pl.dotplot(
        adata,
        var_names=ctx["genes"],
        groupby=ctx["cluster_key"],
        show=False,
        return_fig=True,
    )
    # DotPlot has .fig attribute after make_figure; force it:
    dp.make_figure()
    return dp.fig


def _sc_heatmap(adata: AnnData, ctx: dict):
    sc.pl.heatmap(
        adata,
        var_names=ctx["genes"],
        groupby=ctx["cluster_key"],
        show=False,
    )
    return plt.gcf()


def _sc_violin(adata: AnnData, ctx: dict):
    sc.pl.violin(adata, keys=ctx["genes"][:4], show=False)
    return plt.gcf()



CASES: dict[str, Case] = {
    "umap_gene": Case("umap_gene", _cl_umap_gene, _sc_umap_gene),
    "umap_cluster": Case("umap_cluster", _cl_umap_cluster, _sc_umap_cluster),
    "dotplot": Case("dotplot", _cl_dotplot, _sc_dotplot),
    "heatmap": Case("heatmap", _cl_heatmap, _sc_heatmap),
    "violin": Case("violin", _cl_violin, _sc_violin),
}


def render_for(framework: str, *, fmt: str = "svg") -> Callable[[Any], None]:
    if framework == "cellestial":
        if fmt == "svg":
            return _render_lets_plot
        if fmt == "html":
            return _render_lets_plot_html
        raise ValueError(fmt)
    if framework == "scanpy":
        return _render_matplotlib
    raise ValueError(framework)
