import polars as pl
import pytest
from ggrust import geom_blank
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.complex.annotated_heatmap import _bin_observations
from cellestial.util.errors import UnsupportedDataTypeError


def test_annotated_heatmap_default(adata, markers, group_key):
    plot = cl.annotated_heatmap(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_both_axes(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key, "pct_counts_mt"],
        scale_axis=0,
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_numeric_column_annotation(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, column_annotations=["means"]
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_string_annotations(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations="highly_variable",
        row_annotations=cluster_key,
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_no_legend(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, row_annotations=[cluster_key], legend=False
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_no_group_by(adata, markers):
    plot = cl.annotated_heatmap(adata, keys=markers, row_annotations="pct_counts_mt")
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_max_rows(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, row_annotations=[cluster_key], max_rows=200
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_annotation_colors(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations="highly_variable",
        row_annotations=["pct_counts_mt"],
        annotation_colors={
            "highly_variable": {"true": "#000000", "false": "#dddddd"},
            "pct_counts_mt": ["#ffffcc", "#800026"],
        },
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_dendrogram(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        row_annotations=[group_key],
        dendrogram=True,
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_dendrogram_requires_group_by(adata, markers):
    with pytest.raises(ValueError, match="group_by"):
        cl.annotated_heatmap(adata, keys=markers, dendrogram=True)


def test_annotated_heatmap_passthrough_params(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        group_lines_kwargs={"linetype": "dashed"},
        interactive=True,
        alpha=0.9,
    )
    assert isinstance(plot, PlotSpec)


def test_annotated_heatmap_unsupported_data_type(markers, group_key):
    with pytest.raises(UnsupportedDataTypeError):
        cl.annotated_heatmap("not anndata", keys=markers, group_by=group_key)


def test_bin_observations_caps_rows():
    frame = pl.DataFrame(
        {
            "group": ["a"] * 50 + ["b"] * 50,
            "value": list(range(100)),
            "label": ["x"] * 50 + ["y"] * 50,
        }
    )
    binned = _bin_observations(frame, group_by="group", max_rows=10)
    # proportional allocation: 5 bins per equal-sized group, total == max_rows
    assert binned.height == 10
    # group column and a representative categorical label are preserved
    assert binned["group"].to_list() == ["a"] * 5 + ["b"] * 5
    assert set(binned["label"].to_list()) <= {"x", "y"}


def test_bin_observations_noop_when_small():
    frame = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
    assert _bin_observations(frame, group_by=None, max_rows=1000) is frame


def _side_layers(spec: dict, band: str) -> list[dict]:
    return [layer for layer in spec["layers"] if layer.get("ggside") == band]


def test_annotated_heatmap_is_one_plot_with_side_tracks(adata, markers, group_key, cluster_key):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key, "pct_counts_mt"],
        max_rows=20,
    ).as_dict()

    assert spec["mapping"] == {"x": "position_x", "y": "position_y"}
    assert spec["layers"][0]["mapping"] == {"fill": "value"}
    assert spec["ggside"] == {"x_pos": "top", "y_pos": "left"}
    assert spec["theme"]["ggside_panel_scale_x"] == pytest.approx(0.025)
    assert spec["theme"]["ggside_panel_scale_y"] == pytest.approx(0.05)
    assert [layer["mapping"]["fill"] for layer in _side_layers(spec, "x")] == ["highly_variable"]
    assert [layer["mapping"]["fill"] for layer in _side_layers(spec, "y")] == [
        cluster_key,
        "pct_counts_mt",
    ]
    # every track has its own fill scale group, so its own legend
    groups = [layer["scale_group"]["fill"] for layer in spec["layers"] if "ggside" in layer]
    assert len(set(groups)) == 3
    side_scales = {scale["ggside"]: scale for scale in spec["scales"] if "ggside" in scale}
    # first listed track outermost: leftmost in the left band
    assert side_scales["y"]["limits"] == [cluster_key, "pct_counts_mt"]
    assert side_scales["x"]["position"] == "right"


def test_annotated_heatmap_transpose_moves_tracks_and_group_lines(
    adata, markers, group_key, cluster_key
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key],
        transpose=True,
        max_rows=20,
    ).as_dict()

    assert spec["mapping"] == {"x": "position_y", "y": "position_x"}
    assert spec["ggside"] == {"x_pos": "top", "y_pos": "left"}
    assert [layer["mapping"]["fill"] for layer in _side_layers(spec, "x")] == [cluster_key]
    assert [layer["mapping"]["fill"] for layer in _side_layers(spec, "y")] == ["highly_variable"]
    assert spec["layers"][1]["mapping"] == {"x": "x", "xend": "x"}


def test_annotated_heatmap_right_and_bottom_track_positions(
    adata, markers, group_key, cluster_key
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key, "pct_counts_mt"],
        row_annotation_position="right",
        column_annotation_position="bottom",
        max_rows=20,
    ).as_dict()

    assert spec["ggside"] == {"x_pos": "bottom", "y_pos": "right"}
    side_scales = {scale["ggside"]: scale for scale in spec["scales"] if "ggside" in scale}
    # first listed track outermost: rightmost in the right band
    assert side_scales["y"]["limits"] == ["pct_counts_mt", cluster_key]


def test_annotated_heatmap_transpose_moves_dendrogram_above_heatmap(adata, markers, group_key):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        dendrogram=True,
        transpose=True,
        max_rows=20,
    ).as_dict()

    assert spec["layers"][0]["geom"] == "raster"
    (tree,) = _side_layers(spec, "x")
    assert tree["geom"] == "line"
    assert spec["ggside"]["x_pos"] == "top"
    # root away from the heatmap: heights grow upward from the band's inner edge
    assert min(tree["data"]["height"]) == pytest.approx(-0.5)
    assert max(tree["data"]["height"]) > 0


def test_annotated_heatmap_layers_and_layers_all_are_added(adata, markers, group_key, cluster_key):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        row_annotations=[cluster_key],
        layers=geom_blank(),
        layers_all=geom_blank(),
        max_rows=20,
    ).as_dict()
    geoms = [layer["geom"] for layer in spec["layers"]]

    # both come right after the heatmap, before the tracks' new fill scales
    assert geoms[:3] == ["raster", "blank", "blank"]


def test_annotated_heatmap_raster_warns_and_removes_tooltips(adata, markers, group_key):
    with pytest.warns(cl.util.errors.CellestialWarning, match="tooltips are not supported"):
        plot = cl.annotated_heatmap(
            adata,
            keys=markers[:2],
            group_by=group_key,
            geom="raster",
            tooltips=["value"],
            max_rows=20,
        )

    heatmap_layer = next(layer for layer in plot.as_dict()["layers"] if layer["geom"] == "raster")
    assert "tooltips" not in heatmap_layer
