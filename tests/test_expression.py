import pytest
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl


def test_expression_basic(adata):
    plot = cl.expression(adata, key="CD14")
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("dims", ["umap", "pca", "tsne"])
def test_expression_dimensions(adata, dims):
    plot = cl.expression(adata, key="CD14", dimensions=dims)
    assert isinstance(plot, PlotSpec)


def test_expression_color_high(adata):
    plot = cl.expression(adata, key="CD14", color_high="red")
    assert isinstance(plot, PlotSpec)


def test_expression_invalid_gene(adata):
    with pytest.raises(Exception):
        cl.expression(adata, key="NOT_A_REAL_GENE_xyz")


def test_expression_obs_key_raises_or_returns(adata, group_key):
    # expression is intended for genes; passing an obs column should either
    # raise or handle gracefully, either is acceptable as long as it's deterministic.
    try:
        result = cl.expression(adata, key=group_key)
    except Exception:
        return
    assert isinstance(result, PlotSpec)


def test_expressions_grid(adata):
    plot = cl.expressions(adata, keys=["CD14", "MS4A1", "NKG7"])
    assert isinstance(plot, SupPlotsSpec)


def test_expressions_ncol(adata):
    plot = cl.expressions(adata, keys=["CD14", "MS4A1", "NKG7", "CST3"], ncol=2)
    assert isinstance(plot, SupPlotsSpec)


def test_expressions_single_key_list(adata):
    plot = cl.expressions(adata, keys=["CD14"])
    assert isinstance(plot, SupPlotsSpec)


def test_expressions_invalid_key_raises(adata):
    with pytest.raises(Exception):
        cl.expressions(adata, keys=["NOT_A_GENE_xyz"])
