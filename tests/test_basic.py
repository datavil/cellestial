import pytest
from lets_plot import aes
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl

# ---- scatter ----


def test_scatter_requires_mapping(adata):
    plot = cl.scatter(
        adata,
        mapping=aes(x="n_genes_by_counts", y="pct_counts_in_top_50_genes"),
        color="grey",
        alpha=0.6,
    )
    assert isinstance(plot, PlotSpec)


def test_scatter_color_via_mapping(adata, group_key):
    plot = cl.scatter(
        adata,
        mapping=aes(
            x="n_genes_by_counts",
            y="pct_counts_in_top_50_genes",
            color=group_key,
        ),
        alpha=0.6,
    )
    assert isinstance(plot, PlotSpec)


def test_scatter_without_mapping_raises(adata):
    with pytest.raises(Exception):
        cl.scatter(adata)


# ---- bar ----


def test_bar_basic(adata, cluster_key):
    plot = cl.bar(adata, mapping=aes(cluster_key, fill="predicted_doublet"))
    assert isinstance(plot, PlotSpec)


def test_bar_categorical_fill(adata, group_key, cluster_key):
    plot = cl.bar(adata, mapping=aes(group_key, fill=cluster_key))
    assert isinstance(plot, PlotSpec)


# ---- xyplot / xyplots ----


def test_xyplot_basic(adata):
    plot = cl.xyplot(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_in_top_50_genes",
        color="grey",
    )
    assert isinstance(plot, PlotSpec)


def test_xyplot_mapping_overrides(adata, group_key):
    plot = cl.xyplot(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_in_top_50_genes",
        mapping=aes(color=group_key),
    )
    assert isinstance(plot, PlotSpec)


def test_xyplots_sequence_xy(adata):
    plot = cl.xyplots(
        adata,
        x=["n_genes_by_counts", "total_counts"],
        y=["pct_counts_mt", "pct_counts_ribo"],
    )
    assert isinstance(plot, SupPlotsSpec)


def test_xyplots_scalar_x(adata):
    plot = cl.xyplots(
        adata,
        x="n_genes_by_counts",
        y=["pct_counts_mt", "pct_counts_ribo"],
    )
    assert isinstance(plot, SupPlotsSpec)


# ---- plot ----


def test_plot_smoke(adata):
    # `plot` requires axis to build the frame
    plot = cl.plot(adata, axis=0, include_dimensions=2)
    assert isinstance(plot, PlotSpec)
