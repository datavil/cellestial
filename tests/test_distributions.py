import pytest
from lets_plot import aes
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl

# ---- violin / boxplot (singular) ----


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_single_key_gene(adata, fn, group_key):
    plot = fn(adata, "CD14", fill=group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_single_key_obs(adata, fn, group_key):
    plot = fn(adata, "n_genes_by_counts", fill=group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_sequence_of_keys_returns_single_plot(adata, fn, group_key):
    # Singular violin/boxplot accepts a list of keys and still returns a single PlotSpec
    # (keys become facets / panels within one plot).
    plot = fn(
        adata,
        ["pct_counts_in_top_200_genes", "n_genes_by_counts"],
        fill=group_key,
    )
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_threshold(adata, fn, group_key):
    plot = fn(adata, "CD14", fill=group_key, threshold=0.1)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_show_points_false(adata, fn, group_key):
    plot = fn(adata, "CD14", fill=group_key, show_points=False)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_group_by(adata, fn, group_key):
    plot = fn(adata, "CD14", group_by=group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_add_keys(adata, fn, group_key):
    plot = fn(adata, "CD14", group_by=group_key, add_keys=["MS4A1", "NKG7"])
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_invalid_key(adata, fn):
    with pytest.raises(Exception):
        fn(adata, "NOT_A_REAL_KEY_xyz")


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_invalid_point_geom_raises_value_error(adata, fn, group_key):
    with pytest.raises(ValueError, match="point_geom must be one of"):
        fn(adata, "CD14", fill=group_key, point_geom="bad")


# ---- violins / boxplots (plural) ----


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_grid(adata, fn, group_key):
    plot = fn(
        adata,
        ["n_genes_by_counts", "HLA-DRA", "pct_counts_hb"],
        fill=group_key,
        ncol=2,
    )
    assert isinstance(plot, SupPlotsSpec)


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_single_key_list(adata, fn, group_key):
    plot = fn(adata, ["CD14"], fill=group_key)
    assert isinstance(plot, SupPlotsSpec)


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_group_by(adata, fn, group_key):
    plot = fn(adata, ["CD14", "MS4A1"], group_by=group_key)
    assert isinstance(plot, SupPlotsSpec)


# ---- ridge / ridges ----


def test_ridge_basic(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key)
    assert isinstance(plot, PlotSpec)


def test_ridge_threshold_and_scale(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key, threshold=0.1, scale=1.5)
    assert isinstance(plot, PlotSpec)


def test_ridge_add_keys(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key, add_keys=["MS4A1"])
    assert isinstance(plot, PlotSpec)


def test_ridges_multiple(adata, group_key):
    plot = cl.ridges(adata, keys=["CD14", "MS4A1", "NKG7"], group_by=group_key)
    assert isinstance(plot, SupPlotsSpec)


def test_ridges_ncol(adata, group_key):
    plot = cl.ridges(adata, keys=["CD14", "MS4A1", "NKG7", "CST3"], group_by=group_key, ncol=2)
    assert isinstance(plot, SupPlotsSpec)


# ---- mapping override ----


def test_violin_mapping_extra_aes(adata, group_key):
    plot = cl.violin(adata, "CD14", fill=group_key, mapping=aes(color=group_key))
    assert isinstance(plot, PlotSpec)
