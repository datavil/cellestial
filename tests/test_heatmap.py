import pytest
from lets_plot.plot.core import PlotSpec

import cellestial as cl


def test_heatmap_default_aggregate(adata, markers, group_key):
    plot = cl.heatmap(adata, group_by=group_key, keys=markers)
    assert isinstance(plot, PlotSpec)


def test_heatmap_tile_with_tooltips(adata, markers, group_key):
    plot = cl.heatmap(
        adata, group_by=group_key, keys=markers, geom="tile", tooltips=["value"]
    )
    assert isinstance(plot, PlotSpec)


def test_heatmap_dendrogram(adata, markers, group_key):
    plot = cl.heatmap(adata, group_by=group_key, keys=markers, dendrogram=True)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("scale_axis", [0, 1])
def test_heatmap_scale_axis(adata, markers, group_key, scale_axis):
    plot = cl.heatmap(
        adata, group_by=group_key, keys=markers, scale_axis=scale_axis
    )
    assert isinstance(plot, PlotSpec)


def test_heatmap_aggregate_false(adata, markers, group_key):
    # Non-aggregated: one row per cell. Keep small: few cells x 5 genes.
    sub = adata[:300].copy()
    plot = cl.heatmap(
        sub, group_by=group_key, keys=markers[:5], aggregate=False, geom="raster"
    )
    assert isinstance(plot, PlotSpec)


def test_heatmap_aggregate_false_with_group_bars(adata, markers, group_key):
    sub = adata[:300].copy()
    plot = cl.heatmap(
        sub,
        group_by=group_key,
        keys=markers[:5],
        aggregate=False,
        geom="raster",
        group_bars=True,
    )
    assert isinstance(plot, PlotSpec)



def test_heatmap_invalid_group_by(adata, markers):
    with pytest.raises(Exception):
        cl.heatmap(adata, group_by="NOT_AN_OBS_COL_xyz", keys=markers)


def test_heatmap_invalid_gene(adata, group_key):
    with pytest.raises(Exception):
        cl.heatmap(adata, group_by=group_key, keys=["NOT_A_GENE_xyz"])


# ---- dotplot ----


def test_dotplot_basic(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, PlotSpec)


def test_dotplot_sort_by_avg_exp(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, sort_by="avg_exp")
    assert isinstance(plot, PlotSpec)


def test_dotplot_fill_true(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, fill=True)
    assert isinstance(plot, PlotSpec)


def test_dotplot_custom_colors(adata, markers, group_key):
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        color_low="lightgrey",
        color_high="red",
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_threshold(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, threshold=0.2)
    assert isinstance(plot, PlotSpec)


def test_dotplot_invalid_gene(adata, group_key):
    with pytest.raises(Exception):
        cl.dotplot(adata, keys=["NOT_A_GENE_xyz"], group_by=group_key)


# ---- highest_expressed_genes ----


def test_highest_expressed_genes_default(adata):
    plot = cl.highest_expressed_genes(adata, n=10)
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_threshold(adata):
    plot = cl.highest_expressed_genes(adata, n=10, threshold=0.1)
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_custom_outliers(adata):
    plot = cl.highest_expressed_genes(
        adata, n=5, outlier_size=0.1, outlier_alpha=0.2, outlier_shape=5
    )
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_n_larger_than_var(adata):
    # n larger than var count should clip gracefully, not crash.
    with pytest.raises(ValueError):
        cl.highest_expressed_genes(adata, n=adata.n_vars + 100)
