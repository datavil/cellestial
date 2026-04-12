import pytest
from lets_plot import aes
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl


# ---- singular: dimensional / umap / pca / tsne ----


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_singular_default_no_key(adata, fn):
    plot = fn(adata)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_singular_with_obs_key(adata, fn, group_key):
    plot = fn(adata, group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_singular_with_gene_key(adata, fn):
    plot = fn(adata, "CD14")
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize(
    "dims",
    ["umap", "pca", "tsne"],
)
def test_dimensional_dimensions_param(adata, dims):
    plot = cl.dimensional(adata, "leiden", dimensions=dims)
    assert isinstance(plot, PlotSpec)


def test_dimensional_xy_swap(adata):
    # Non-default embedding components
    plot = cl.pca(adata, "CD14", xy=(2, 3))
    assert isinstance(plot, PlotSpec)


def test_dimensional_mapping_extra_aesthetic(adata, group_key):
    # Providing a key + an extra aesthetic via mapping (shape)
    plot = cl.umap(adata, group_key, mapping=aes(shape=group_key))
    assert isinstance(plot, PlotSpec)


def test_dimensional_with_variable_keys(adata):
    plot = cl.umap(adata, "CD14", variable_keys=["MS4A1", "NKG7"])
    assert isinstance(plot, PlotSpec)


def test_dimensional_tooltips_none(adata, group_key):
    plot = cl.umap(adata, group_key, tooltips="none")
    assert isinstance(plot, PlotSpec)


def test_dimensional_tooltips_sequence(adata, group_key):
    plot = cl.umap(adata, group_key, tooltips=[group_key, "n_genes_by_counts"])
    assert isinstance(plot, PlotSpec)


def test_dimensional_invalid_key_raises(adata):
    with pytest.raises((KeyError, ValueError, Exception)):
        cl.umap(adata, "NOT_A_REAL_KEY_xyz")


# ---- plural: dimensionals / umaps / pcas / tsnes ----


@pytest.mark.parametrize("fn", [cl.umaps, cl.pcas, cl.tsnes])
def test_plural_returns_subplots(adata, fn, group_key):
    plot = fn(adata, [group_key, "leiden"])
    assert isinstance(plot, SupPlotsSpec)


def test_plural_mixed_gene_and_obs(adata, group_key):
    plot = cl.umaps(adata, ["CD14", group_key, "MS4A1"])
    assert isinstance(plot, SupPlotsSpec)


def test_plural_single_entry_list(adata):
    plot = cl.umaps(adata, ["CD14"])
    assert isinstance(plot, SupPlotsSpec)


def test_dimensionals_dimensions(adata, group_key):
    plot = cl.dimensionals(adata, [group_key, "CD14"], dimensions="pca")
    assert isinstance(plot, SupPlotsSpec)


def test_dimensionals_ncol(adata, group_key):
    plot = cl.umaps(adata, [group_key, "CD14", "MS4A1"], ncol=2)
    assert isinstance(plot, SupPlotsSpec)


def test_plural_invalid_key_raises(adata):
    with pytest.raises(Exception):
        cl.umaps(adata, ["NOT_A_KEY_xyz"])
