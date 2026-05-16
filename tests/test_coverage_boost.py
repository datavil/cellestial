"""
Additional tests to boost total coverage above 85%.

Covers uncovered paths in: dimensional, distribution, subdimensionals,
arrow, legends, operations, xyplot, xyplots, ridge, highest, heatmap.
"""

import pytest
from lets_plot import aes, layer_tooltips
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.layers import DeferredLayer

# ===========================================================================
# dimensional.py, uncovered: TypeError, use_key, legend_ondata, color_mid,
#                  interactive, FeatureSpec tooltips
# ===========================================================================


def test_dimensional_invalid_data():
    with pytest.raises(TypeError):
        cl.dimensional("not anndata", "leiden")


def test_dimensional_use_key(adata):
    plot = cl.umap(adata, "leiden", use_key="X_umap")
    assert isinstance(plot, PlotSpec)


def test_dimensional_legend_ondata(adata, group_key):
    plot = cl.umap(adata, group_key, legend_ondata=True)
    assert isinstance(plot, PlotSpec)


def test_dimensional_color_mid(adata):
    plot = cl.umap(adata, "CD14", color_mid="yellow", mid_point="mean")
    assert isinstance(plot, PlotSpec)


def test_dimensional_color_mid_midpoint(adata):
    plot = cl.umap(adata, "CD14", color_mid="yellow", mid_point="mid")
    assert isinstance(plot, PlotSpec)


def test_dimensional_interactive(adata):
    plot = cl.umap(adata, "leiden", interactive=True)
    assert isinstance(plot, PlotSpec)


def test_dimensional_tooltips_featurespec(adata, group_key):
    plot = cl.umap(adata, group_key, tooltips=layer_tooltips([group_key]))
    assert isinstance(plot, PlotSpec)


def test_dimensional_xy_invalid_length(adata):
    with pytest.raises(KeyError):
        cl.umap(adata, "leiden", xy=(1, 2, 3))


# ===========================================================================
# distribution.py, uncovered: TypeError, FeatureSpec tooltips, interactive,
#                   invalid point_geom
# ===========================================================================


def test_violin_invalid_data():
    with pytest.raises(TypeError):
        cl.violin("not anndata", "CD14")


def test_violin_tooltips_featurespec(adata, group_key):
    plot = cl.violin(adata, "CD14", fill=group_key, tooltips=layer_tooltips(["variable"]))
    assert isinstance(plot, PlotSpec)


def test_violin_interactive(adata, group_key):
    plot = cl.violin(adata, "CD14", fill=group_key, interactive=True)
    assert isinstance(plot, PlotSpec)


def test_boxplot_interactive(adata, group_key):
    plot = cl.boxplot(adata, "CD14", fill=group_key, interactive=True)
    assert isinstance(plot, PlotSpec)


def test_violin_tooltips_none(adata, group_key):
    plot = cl.violin(adata, "CD14", fill=group_key, tooltips="none")
    assert isinstance(plot, PlotSpec)


def test_violin_multi_key_with_group(adata, group_key):
    plot = cl.violin(adata, ["CD14", "MS4A1"], group_by=group_key)
    assert isinstance(plot, PlotSpec)


# ===========================================================================
# distributions.py, uncovered: share_ticks, share_axis, layers
# ===========================================================================


def test_violins_share_ticks(adata, group_key):
    plot = cl.violins(adata, ["CD14", "MS4A1"], fill=group_key, ncol=1, share_ticks=True)
    assert isinstance(plot, SupPlotsSpec)


def test_violins_share_axis(adata, group_key):
    plot = cl.violins(adata, ["CD14", "MS4A1"], fill=group_key, ncol=1, share_axis=True)
    assert isinstance(plot, SupPlotsSpec)


def test_boxplots_share_ticks(adata, group_key):
    plot = cl.boxplots(adata, ["CD14", "MS4A1"], fill=group_key, ncol=1, share_ticks=True)
    assert isinstance(plot, SupPlotsSpec)


def test_boxplots_share_axis(adata, group_key):
    plot = cl.boxplots(adata, ["CD14", "MS4A1"], fill=group_key, ncol=1, share_axis=True)
    assert isinstance(plot, SupPlotsSpec)


def test_violins_interactive_with_layers(adata, group_key):
    from lets_plot import geom_point

    plot = cl.violins(
        adata,
        ["CD14", "MS4A1"],
        fill=group_key,
        layers=geom_point(size=0.1),
        interactive=True,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_boxplots_interactive_with_layers(adata, group_key):
    from lets_plot import geom_point

    plot = cl.boxplots(
        adata,
        ["CD14", "MS4A1"],
        fill=group_key,
        layers=geom_point(size=0.1),
        interactive=True,
    )
    assert isinstance(plot, SupPlotsSpec)


# ===========================================================================
# subdimensionals.py, uncovered: share_labels, share_axis, layers, TypeError
# ===========================================================================


def test_umaps_share_labels(adata, group_key):
    plot = cl.umaps(adata, [group_key, "CD14"], ncol=1, share_labels=True)
    assert isinstance(plot, SupPlotsSpec)


def test_umaps_share_axis(adata, group_key):
    plot = cl.umaps(adata, [group_key, "CD14"], ncol=1, share_axis=True, axis_type="axis")
    assert isinstance(plot, SupPlotsSpec)


def test_dimensionals_share_labels(adata, group_key):
    plot = cl.dimensionals(adata, [group_key, "CD14"], ncol=1, share_labels=True)
    assert isinstance(plot, SupPlotsSpec)


def test_dimensionals_share_axis(adata, group_key):
    plot = cl.dimensionals(adata, [group_key, "CD14"], ncol=1, share_axis=True, axis_type="axis")
    assert isinstance(plot, SupPlotsSpec)


def test_umaps_invalid_keys_type(adata):
    with pytest.raises(TypeError):
        cl.umaps(adata, "single_string")


def test_dimensionals_invalid_keys_type(adata):
    with pytest.raises(TypeError):
        cl.dimensionals(adata, "single_string")


def test_tsnes_basic(adata, group_key):
    plot = cl.tsnes(adata, [group_key, "CD14"])
    assert isinstance(plot, SupPlotsSpec)


def test_tsnes_invalid_keys_type(adata):
    with pytest.raises(TypeError):
        cl.tsnes(adata, "single_string")


def test_pcas_basic(adata, group_key):
    plot = cl.pcas(adata, [group_key, "CD14"])
    assert isinstance(plot, SupPlotsSpec)


def test_pcas_invalid_keys_type(adata):
    with pytest.raises(TypeError):
        cl.pcas(adata, "single_string")


def test_expressions_invalid_keys_type(adata):
    with pytest.raises(TypeError):
        cl.expressions(adata, "single_string")


def test_tsnes_interactive_share_axis_with_layers(adata, group_key):
    from lets_plot import geom_point

    plot = cl.tsnes(
        adata,
        [group_key, "CD14"],
        axis_type="axis",
        share_labels=True,
        share_axis=True,
        layers=geom_point(size=0.1),
        interactive=True,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_pcas_interactive_share_axis_with_layers(adata, group_key):
    from lets_plot import geom_point

    plot = cl.pcas(
        adata,
        [group_key, "CD14"],
        axis_type="axis",
        share_labels=True,
        share_axis=True,
        layers=geom_point(size=0.1),
        interactive=True,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_expressions_interactive_share_axis_with_layers(adata, group_key):
    from lets_plot import geom_point

    plot = cl.expressions(
        adata,
        ["CD14", "MS4A1"],
        axis_type="axis",
        share_labels=True,
        share_axis=True,
        layers=geom_point(size=0.1),
        interactive=True,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_umaps_with_layers(adata, group_key):
    from lets_plot import geom_point

    layer = geom_point(size=0.1, alpha=0.1)
    plot = cl.umaps(adata, [group_key, "CD14"], layers=layer)
    assert isinstance(plot, SupPlotsSpec)


def test_dimensionals_with_layers(adata, group_key):
    from lets_plot import geom_point

    layer = geom_point(size=0.1, alpha=0.1)
    plot = cl.dimensionals(adata, [group_key, "CD14"], layers=layer)
    assert isinstance(plot, SupPlotsSpec)


# ===========================================================================
# arrow.py, uncovered: _modify_axis paths, arrow_axis
# ===========================================================================


def test_dimensional_axis_type_arrow(adata):
    plot = cl.umap(adata, "CD14", axis_type="arrow")
    assert isinstance(plot, PlotSpec)


def test_dimensional_axis_type_none(adata):
    plot = cl.umap(adata, "CD14", axis_type=None)
    assert isinstance(plot, PlotSpec)


def test_arrow_axis_default(adata):
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis()
    assert isinstance(arrow, DeferredLayer)
    combined = umap + arrow
    assert isinstance(combined, PlotSpec)


# ===========================================================================
# legends.py, _legend_ondata (covered via legend_ondata tests above)
# ===========================================================================


# ===========================================================================
# operations.py, uncovered: error paths in get_slice, retrieve
# ===========================================================================


def test_slice_invalid_grid_type():
    with pytest.raises(TypeError):
        cl.slice("not a grid", index=0)


def test_retrieve_invalid_type():
    with pytest.raises(TypeError):
        cl.retrieve("not a plot")


# ===========================================================================
# xyplot.py, uncovered: TypeError, tooltips variants, interactive
# ===========================================================================


def test_xyplot_invalid_data():
    with pytest.raises(TypeError):
        cl.xyplot("not anndata", x="a", y="b")


def test_xyplot_tooltips_sequence(adata):
    plot = cl.xyplot(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_in_top_50_genes",
        tooltips=["n_genes_by_counts"],
    )
    assert isinstance(plot, PlotSpec)


def test_xyplot_tooltips_featurespec(adata):
    plot = cl.xyplot(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_in_top_50_genes",
        tooltips=layer_tooltips(["n_genes_by_counts"]),
    )
    assert isinstance(plot, PlotSpec)


def test_xyplot_tooltips_none(adata):
    plot = cl.xyplot(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_in_top_50_genes",
        tooltips="none",
    )
    assert isinstance(plot, PlotSpec)


def test_xyplot_interactive(adata):
    plot = cl.xyplot(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_in_top_50_genes",
        interactive=True,
    )
    assert isinstance(plot, PlotSpec)


# ===========================================================================
# xyplots.py, uncovered: layers, broadcasting y
# ===========================================================================


def test_xyplots_scalar_y(adata):
    plot = cl.xyplots(
        adata,
        x=["n_genes_by_counts", "total_counts"],
        y="pct_counts_mt",
    )
    assert isinstance(plot, SupPlotsSpec)


def test_xyplots_with_layers(adata):
    from lets_plot import geom_point

    layer = geom_point(size=0.1, alpha=0.1)
    plot = cl.xyplots(
        adata,
        x="n_genes_by_counts",
        y=["pct_counts_mt", "pct_counts_ribo"],
        layers=layer,
    )
    assert isinstance(plot, SupPlotsSpec)


# ===========================================================================
# ridge.py, uncovered: TypeError, tooltips, interactive
# ===========================================================================


def test_ridge_invalid_data(group_key):
    with pytest.raises(TypeError):
        cl.ridge("not anndata", key="CD14", group_by=group_key)


def test_ridge_tooltips_none(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key, tooltips="none")
    assert isinstance(plot, PlotSpec)


def test_ridge_tooltips_sequence(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key, tooltips=[group_key, "CD14"])
    assert isinstance(plot, PlotSpec)


def test_ridge_tooltips_featurespec(adata, group_key):
    plot = cl.ridge(
        adata,
        key="CD14",
        group_by=group_key,
        tooltips=layer_tooltips([group_key]),
    )
    assert isinstance(plot, PlotSpec)


def test_ridge_interactive(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key, interactive=True)
    assert isinstance(plot, PlotSpec)


def test_ridges_with_layers(adata, group_key):
    from lets_plot import geom_point

    layer = geom_point(size=0.1, alpha=0.1)
    plot = cl.ridges(adata, keys=["CD14", "MS4A1"], group_by=group_key, layers=layer)
    assert isinstance(plot, SupPlotsSpec)


# ===========================================================================
# highest.py, uncovered: TypeError, mapping
# ===========================================================================


def test_highest_expressed_genes_invalid_data():
    with pytest.raises(TypeError):
        cl.highest_expressed_genes("not anndata", n=10)


def test_highest_expressed_genes_with_mapping(adata):
    plot = cl.highest_expressed_genes(adata, n=5, mapping=aes(color="variable"))
    assert isinstance(plot, PlotSpec)


# ===========================================================================
# heatmap.py, uncovered: TypeError, raster tooltips warning
# ===========================================================================


def test_heatmap_invalid_data(markers, group_key):
    with pytest.raises(TypeError):
        cl.heatmap("not anndata", group_by=group_key, keys=markers)


def test_heatmap_raster_tooltips_warning(adata, markers, group_key):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot = cl.heatmap(
            adata,
            group_by=group_key,
            keys=markers[:3],
            geom="raster",
            tooltips=["value"],
        )
    assert isinstance(plot, PlotSpec)


def test_heatmap_scale_axis_1(adata, markers, group_key):
    plot = cl.heatmap(adata, group_by=group_key, keys=markers, scale_axis=1, aggregate=False)
    assert isinstance(plot, PlotSpec)
