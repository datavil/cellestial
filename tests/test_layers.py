import pytest
from lets_plot import ggplot
from lets_plot.plot.core import PlotSpec

import cellestial as cl
from cellestial.layers import DeferredLayer


def test_arrow_axis_returns_deferred(adata):
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis()
    assert isinstance(arrow, DeferredLayer)
    # Addition should succeed
    combined = umap + arrow
    assert isinstance(combined, PlotSpec)


def test_arrow_axis_customization(adata):
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis(length=0.20, color="dark_violet")
    assert isinstance(arrow, DeferredLayer)
    combined = umap + arrow
    assert isinstance(combined, PlotSpec)


def test_arrow_axis_explicit_plot_kwarg(adata):
    # Explicit `plot=` pins the data source regardless of the receiving plot.
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis(plot=umap, length=0.20)
    assert isinstance(arrow, DeferredLayer)
    # Even when added to an empty ggplot, the data source stays `umap`.
    combined = ggplot() + arrow
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_single_group(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(groups="B Cells")
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_multiple_groups(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(groups=["B Cells", "Erythroid"])
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_nested_groups(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(groups=[["Lymphocytes", "Monocytes"], "B Cells"])
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_invalid_group_raises(adata, group_key):
    # DeferredLayer defers construction to `+` time, so the error fires there.
    umap = cl.umap(adata, key=group_key)
    with pytest.raises(Exception):
        _ = umap + cl.cluster_outlines(groups="NOT_A_GROUP_xyz")


def test_stream_requires_velocity(adata):
    # pbmc3k fixture has no velocity columns; stream should raise at `+` time.
    umap = cl.umap(adata)
    with pytest.raises(Exception):
        _ = umap + cl.stream()
