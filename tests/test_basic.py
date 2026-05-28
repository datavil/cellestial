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


# ---- frame narrowing: custom tooltips on shared frames and the variable axis ----


def test_xyplots_custom_metadata_tooltip(adata):
    # Regression: the shared xyplots frame must contain a custom tooltip column
    # that is not one of the x/y/mapping fields.
    obs_column = "log1p_total_counts"
    plot = cl.xyplots(
        adata,
        x=["n_genes_by_counts", "total_counts"],
        y=["pct_counts_mt", "pct_counts_ribo"],
        tooltips=[obs_column],
    )
    assert isinstance(plot, SupPlotsSpec)
    for panel in plot.as_dict()["figures"]:
        assert obs_column in panel["data"]


def test_xyplot_variable_axis_custom_tooltip(adata):
    # Regression: on the variable axis, a var-metadata tooltip field outside
    # x/y must still be materialised into the frame.
    plot = cl.xyplot(
        adata,
        x="mean_counts",
        y="total_counts",
        axis=1,
        tooltips=["n_cells_by_counts"],
    )
    assert isinstance(plot, PlotSpec)
    assert "n_cells_by_counts" in plot.as_dict()["data"]


def test_xyplot_add_columns_materializes_obs_metadata_and_gene(adata):
    plot = cl.xyplot(
        adata,
        x="n_genes_by_counts",
        y="pct_counts_mt",
        add_columns=["log1p_total_counts", "MS4A1"],
        tooltips="none",
    )
    assert isinstance(plot, PlotSpec)
    data_columns = plot.as_dict()["data"].columns
    assert "log1p_total_counts" in data_columns
    assert "MS4A1" in data_columns


def test_xyplot_variable_axis_add_columns_materializes_var_metadata(adata):
    plot = cl.xyplot(
        adata,
        x="mean_counts",
        y="total_counts",
        axis=1,
        add_columns="n_cells_by_counts",
        tooltips="none",
    )
    assert isinstance(plot, PlotSpec)
    assert "n_cells_by_counts" in plot.as_dict()["data"].columns


def test_xyplots_add_columns_materializes_shared_frame_columns(adata):
    plot = cl.xyplots(
        adata,
        x="n_genes_by_counts",
        y=["pct_counts_mt", "pct_counts_ribo"],
        add_columns=["log1p_total_counts", "MS4A1"],
        tooltips="none",
    )
    assert isinstance(plot, SupPlotsSpec)
    for panel in plot.as_dict()["figures"]:
        panel_columns = panel["data"].columns
        assert "log1p_total_counts" in panel_columns
        assert "MS4A1" in panel_columns
