"""`outline=True` on dimensional plots draws a scanpy-style outline around the point cloud."""

import pytest
from ggrust import aes
from ggrust.plot.core import PlotSpec
from ggrust.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.util.errors import InvalidArgumentError


def _layers(plot: PlotSpec) -> list[dict]:
    return plot.as_dict()["layers"]


def test_outline_off_by_default(adata, group_key):
    layers = _layers(cl.umap(adata, group_key))
    assert len(layers) == 1
    assert "alpha" not in layers[0]


def test_outline_adds_outline_and_gap_layers_below_points(adata, group_key):
    outline_layer, gap_layer, points = _layers(cl.umap(adata, group_key, outline=True, size=1.0))
    # Sizes follow scanpy: gap = size * (1 + 2 * 0.05), outline = gap + 2 * size * 0.3.
    assert gap_layer["size"] == pytest.approx(1.1)
    assert outline_layer["size"] == pytest.approx(1.7)
    assert (outline_layer["color"], outline_layer["fill"]) == ("black", "black")
    assert (gap_layer["color"], gap_layer["fill"]) == ("white", "white")
    assert points["size"] == 1.0
    assert points["alpha"] == 0.7
    for layer in (outline_layer, gap_layer):
        assert layer["show_legend"] is False
        assert layer["tooltips"] == "none"
        # Same mapping as the points, so `+` layers reading layer 0 still work.
        assert layer["mapping"] == points["mapping"]


def test_outline_width_color_and_alpha_overrides(adata, group_key):
    outline_layer, gap_layer, points = _layers(
        cl.umap(
            adata,
            group_key,
            outline=True,
            size=2.0,
            outline_width=(0.5, 0.0),
            outline_color=("red", "blue"),
            alpha=1,
        )
    )
    assert gap_layer["size"] == pytest.approx(2.0)
    assert outline_layer["size"] == pytest.approx(4.0)
    assert outline_layer["color"] == "red"
    assert gap_layer["color"] == "blue"
    assert points["alpha"] == 1


def test_outline_layers_follow_shape_and_stroke(adata, group_key):
    outline_layer, gap_layer, _ = _layers(
        cl.umap(adata, group_key, outline=True, shape=15, stroke=0)
    )
    for layer in (outline_layer, gap_layer):
        assert layer["shape"] == 15
        assert layer["stroke"] == 0


def test_outline_on_continuous_key(adata):
    plot = cl.expression(adata, "CD14", outline=True)
    assert len(_layers(plot)) == 3


def test_outline_keeps_ondata_legend_working(adata, group_key):
    plot = cl.umap(adata, group_key, outline=True, legend_ondata=True)
    assert [layer["geom"] for layer in _layers(plot)] == ["point", "point", "point", "text"]


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne, cl.dimensional])
def test_singular_variants_outline(adata, group_key, fn):
    assert len(_layers(fn(adata, group_key, outline=True))) == 3


@pytest.mark.parametrize("fn", [cl.umaps, cl.pcas, cl.tsnes, cl.dimensionals, cl.expressions])
def test_plural_variants_outline(adata, group_key, fn):
    keys = ["CD14", "LYZ"] if fn is cl.expressions else [group_key, "CD14"]
    plot = fn(adata, keys, outline=True, outline_color=("navy", "white"))
    assert isinstance(plot, SupPlotsSpec)
    for figure in plot.as_dict()["figures"]:
        layers = figure["layers"]
        assert len(layers) == 3
        assert layers[0]["color"] == "navy"


def test_outline_rejects_mapped_size(adata, group_key):
    with pytest.raises(InvalidArgumentError, match="constant `size`"):
        cl.umap(adata, group_key, outline=True, mapping=aes(size="n_genes"))


def test_outline_rejects_size_none(adata, group_key):
    with pytest.raises(InvalidArgumentError, match="constant `size`"):
        cl.umap(adata, group_key, outline=True, size=None)


@pytest.mark.parametrize("width", [(0.3,), (0.3, 0.05, 0.1), (-0.1, 0.05), ("a", 0.05), (True, 0)])
def test_outline_rejects_bad_width(adata, group_key, width):
    with pytest.raises(InvalidArgumentError, match="`outline_width`"):
        cl.umap(adata, group_key, outline=True, outline_width=width)


@pytest.mark.parametrize("color", ["black", ("black",), ("black", 1)])
def test_outline_rejects_bad_color(adata, group_key, color):
    with pytest.raises(InvalidArgumentError, match="`outline_color`"):
        cl.umap(adata, group_key, outline=True, outline_color=color)


def test_outline_keeps_mapped_alpha(adata, group_key):
    outline_layer, gap_layer, points = _layers(
        cl.umap(adata, group_key, outline=True, mapping=aes(alpha="n_genes"))
    )
    assert "alpha" not in points
    assert points["mapping"]["alpha"] == "n_genes"
    assert outline_layer["alpha"] == 1
    assert gap_layer["alpha"] == 1


@pytest.mark.parametrize("size", [0, -1])
def test_outline_rejects_non_positive_size(adata, group_key, size):
    with pytest.raises(InvalidArgumentError, match="positive `size`"):
        cl.umap(adata, group_key, outline=True, size=size)


def test_outline_rejects_scalar_width_and_color(adata, group_key):
    with pytest.raises(InvalidArgumentError, match="`outline_width`"):
        cl.umap(adata, group_key, outline=True, outline_width=0.3)
    with pytest.raises(InvalidArgumentError, match="`outline_color`"):
        cl.umap(adata, group_key, outline=True, outline_color=None)
