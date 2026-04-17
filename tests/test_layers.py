import pytest
from lets_plot.plot.core import FeatureSpec, FeatureSpecArray, PlotSpec

import cellestial as cl


def test_arrow_axis_returns_feature(adata):
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis(umap)
    assert isinstance(arrow, (FeatureSpec, FeatureSpecArray))
    # Addition should succeed
    combined = umap + arrow
    assert isinstance(combined, PlotSpec)


def test_arrow_axis_customization(adata):
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis(umap, length=0.20, color="dark_violet")
    assert isinstance(arrow, (FeatureSpec, FeatureSpecArray))


def test_cluster_outlines_single_group(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(umap, groups="B Cells")
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_multiple_groups(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(umap, groups=["B Cells", "Erythroid"])
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_nested_groups(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(umap, groups=[["Lymphocytes", "Monocytes"], "B Cells"])
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_invalid_group_raises(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    with pytest.raises(Exception):
        cl.cluster_outlines(umap, groups="NOT_A_GROUP_xyz")


def test_stream_requires_velocity(adata):
    # pbmc3k fixture has no velocity layers; stream should raise.
    umap = cl.umap(adata)
    with pytest.raises(Exception):
        cl.stream(umap)
