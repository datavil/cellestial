import polars as pl
import pytest
from ggrust import aes, geom_point, geom_text, ggplot
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.layers import DeferredLayer


@pytest.fixture
def plot():
    frame = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 1.0, 2.0], "group": ["a", "b", "a"]})
    return ggplot(frame, aes("x", "y")) + geom_point()


def test_deferred_layer_builds_from_the_receiving_plot(plot):
    received = []

    def build(receiving_plot):
        received.append(receiving_plot)
        return geom_text(aes(label="group"))

    combined = plot + DeferredLayer(build)
    assert isinstance(combined, PlotSpec)
    assert received == [plot]
    assert [layer["geom"] for layer in combined.as_dict()["layers"]] == ["point", "text"]


def test_deferred_layer_chains_with_plain_layers(plot):
    combined = plot + DeferredLayer(lambda _: geom_text(aes(label="group"))) + geom_point()
    assert [layer["geom"] for layer in combined.as_dict()["layers"]] == ["point", "text", "point"]


def test_deferred_layer_adds_a_feature_array(plot):
    combined = plot + DeferredLayer(lambda _: geom_text(aes(label="group")) + geom_point())
    assert len(combined.as_dict()["layers"]) == 3


def test_deferred_layer_leaves_the_plot_unchanged(plot):
    _ = plot + DeferredLayer(lambda _: geom_point())
    assert len(plot.as_dict()["layers"]) == 1


def test_cellestial_deferred_layers_chain_on_a_umap(adata, group_key):
    umap = cl.umap(adata, group_key)
    combined = umap + cl.cluster_outlines(groups=["B Cells"]) + cl.ondata_legend()
    assert isinstance(combined, PlotSpec)
    assert len(combined.as_dict()["layers"]) > len(umap.as_dict()["layers"]) + 1
