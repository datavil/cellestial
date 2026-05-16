import polars as pl
import pytest
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl


def test_get_mapping_returns_dict(adata):
    umap = cl.umap(adata, key="CD14")
    mapping = cl.get_mapping(umap)
    assert isinstance(mapping, dict)


def test_retrieve_returns_dataframe(adata):
    umap = cl.umap(adata, key="CD14")
    frame = cl.retrieve(umap)
    assert isinstance(frame, pl.DataFrame)
    assert frame.height > 0


def test_retrieve_from_grid(adata):
    grid = cl.expressions(adata, keys=["CD14", "MS4A1"])
    frame = cl.retrieve(grid, index=0)
    assert isinstance(frame, pl.DataFrame)


def test_get_figure_single_index(adata):
    grid = cl.expressions(adata, keys=["CD14", "MS4A1", "NKG7", "CST3"], ncol=2)
    single = cl.get_figure(grid, index=0)
    assert isinstance(single, PlotSpec)


def test_get_figures_sequence_index(adata):
    grid = cl.expressions(adata, keys=["CD14", "MS4A1", "NKG7", "CST3"], ncol=2)
    subset = cl.get_figures(grid, indices=[0, 2])
    assert isinstance(subset, SupPlotsSpec)


def test_get_figure_out_of_range(adata):
    grid = cl.expressions(adata, keys=["CD14", "MS4A1"])
    with pytest.raises(IndexError):
        cl.get_figure(grid, index=99)


def test_get_figures_out_of_range(adata):
    grid = cl.expressions(adata, keys=["CD14", "MS4A1"])
    with pytest.raises(IndexError):
        cl.get_figures(grid, indices=[0, 99])


# ---- colors ----


def test_show_colors_returns_subplots():
    from cellestial.util.colors import show_colors

    plot = show_colors()
    assert isinstance(plot, SupPlotsSpec)


def test_color_constants_are_hex():
    from cellestial.util.colors import (
        BLUE,
        CHERRY,
        LIGHT_GRAY,
        ORANGE,
        PINK,
        PURPLE,
        RED,
        SNOW,
        TEAL,
    )

    for c in [BLUE, CHERRY, LIGHT_GRAY, ORANGE, PINK, PURPLE, RED, SNOW, TEAL]:
        assert isinstance(c, str)
        assert c.startswith("#")
        assert len(c) == 7
