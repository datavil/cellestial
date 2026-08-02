import polars as pl
import pytest
from lets_plot import ggplot
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.util.operations import _normalize_widths


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


def test_normalize_widths_accepts_nested_sequences():
    """Nested widths should accept any non-string sequence."""
    plot = ggplot()

    assert _normalize_widths([[plot, plot]], [range(2)]) == [[0, 1]]


def test_layout_preserves_ragged_rows_and_sizing():
    plot = ggplot()

    grid = cl.layout(
        [[plot, plot], plot],
        widths=[[1, 2], [1]],
        heights=[2, 1],
        hspace=3,
        vspace=4,
    )
    spec = grid.as_dict()

    assert spec["layout"] == {
        "ncol": 1,
        "nrow": 2,
        "heights": [2, 1],
        "vspace": 4,
        "name": "grid",
    }
    assert [row["layout"]["widths"] for row in spec["figures"]] == [[1, 2], [1]]
    assert [len(row["figures"]) for row in spec["figures"]] == [2, 1]


def test_layout_broadcasts_flat_widths_and_accepts_nested_grids():
    plot = ggplot()
    nested = cl.layout([[plot]])

    spec = cl.layout([[plot, nested], [plot, plot]], widths=[1, 2]).as_dict()

    assert [row["layout"]["widths"] for row in spec["figures"]] == [[1, 2], [1, 2]]
    assert spec["figures"][0]["figures"][1]["kind"] == "subplots"


@pytest.mark.parametrize(
    ("plots", "kwargs", "error"),
    [
        ("not rows", {}, TypeError),
        ([], {}, ValueError),
        ([[]], {}, ValueError),
        ([123], {}, TypeError),
        ([[ggplot()]], {"widths": 1}, TypeError),
        ([[ggplot()], [ggplot()]], {"widths": [[1]]}, ValueError),
        ([[ggplot(), ggplot()]], {"widths": [[1]]}, ValueError),
        ([[ggplot()], [ggplot(), ggplot()]], {"widths": [1]}, ValueError),
        ([[ggplot()]], {"widths": [1, [2]]}, ValueError),
        ([[ggplot()], [ggplot()]], {"heights": [1]}, ValueError),
    ],
)
def test_layout_rejects_invalid_shapes(plots, kwargs, error):
    with pytest.raises(error):
        cl.layout(plots, **kwargs)


def test_layout_rejects_non_plot_members_when_serialized():
    with pytest.raises(AttributeError):
        cl.layout([[ggplot(), 123]]).as_dict()


def test_get_figure_and_get_figures_validate_index_types(adata):
    grid = cl.expressions(adata, keys=["CD14", "MS4A1"])

    with pytest.raises(TypeError, match="int"):
        cl.get_figure(grid, index=True)
    with pytest.raises(TypeError):
        cl.get_figures(grid, indices=[0.5])
    with pytest.raises(TypeError, match="Sequence"):
        cl.get_figures(grid, indices=0)
    assert isinstance(cl.get_figure(grid, index=-1), PlotSpec)


def test_retrieve_raises_when_plot_has_no_data():
    with pytest.raises(ValueError, match="Could not retrieve"):
        cl.retrieve(ggplot())


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
