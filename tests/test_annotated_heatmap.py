import polars as pl
import pytest
from ggrust import geom_blank
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.complex.annotated_heatmap import _bin_observations, _spread_labels
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
    # track titles hidden by default: the legends name the tracks, and the band
    # axes stay on the main axes' sides (no stray line on the far side)
    assert side_scales["x"]["breaks"] == []
    assert side_scales["y"]["breaks"] == []
    assert "position" not in side_scales["x"]
    assert "position" not in side_scales["y"]


def test_annotated_heatmap_track_titles_shown(adata, markers, group_key, cluster_key):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key],
        row_track_titles=True,
        column_track_titles=True,
        max_rows=20,
    ).as_dict()
    side_scales = {scale["ggside"]: scale for scale in spec["scales"] if "ggside" in scale}

    assert "breaks" not in side_scales["x"]
    assert "breaks" not in side_scales["y"]
    assert side_scales["x"]["position"] == "right"
    # both shown: the y band's names move below, out of the x band's corner
    assert side_scales["y"]["position"] == "bottom"


@pytest.mark.parametrize(
    ("row_track_titles", "column_track_titles"), [(True, False), (False, True)]
)
def test_annotated_heatmap_track_titles_per_band(
    adata, markers, group_key, cluster_key, row_track_titles, column_track_titles
):
    # transposed: row tracks in the top (x) band, column tracks in the left (y) band
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key],
        row_track_titles=row_track_titles,
        column_track_titles=column_track_titles,
        transpose=True,
        max_rows=20,
    ).as_dict()
    side_scales = {scale["ggside"]: scale for scale in spec["scales"] if "ggside" in scale}

    assert ("breaks" not in side_scales["x"]) == row_track_titles
    assert ("breaks" not in side_scales["y"]) == column_track_titles
    if column_track_titles:
        # turned names on top, so they don't inflate the bottom axis area
        assert side_scales["y"]["position"] == "top"


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


def test_annotated_heatmap_transpose_puts_first_group_left_and_first_key_top(
    adata, markers, group_key
):
    # Regression: transposed, the first group landed on the right and the first
    # key at the bottom, reversing the diagonal.
    keys = markers[:3]
    spec = cl.annotated_heatmap(
        adata,
        keys=keys,
        group_by=group_key,
        row_annotations=[group_key],
        transpose=True,
        max_rows=40,
    ).as_dict()

    (track,) = _side_layers(spec, "x")
    track_frame = pl.DataFrame(track["data"]).sort("position_y")
    assert track_frame[group_key][0] == adata.obs[group_key].iloc[0]
    key_scale = next(scale for scale in spec["scales"] if scale.get("labels") == keys)
    assert key_scale["aesthetic"] == "y"
    assert keys[key_scale["breaks"].index(max(key_scale["breaks"]))] == keys[0]


def test_annotated_heatmap_upright_keeps_first_group_top_and_first_key_left(
    adata, markers, group_key
):
    keys = markers[:3]
    spec = cl.annotated_heatmap(
        adata, keys=keys, group_by=group_key, row_annotations=[group_key], max_rows=40
    ).as_dict()

    (track,) = _side_layers(spec, "y")
    track_frame = pl.DataFrame(track["data"]).sort("position_y", descending=True)
    assert track_frame[group_key][0] == adata.obs[group_key].iloc[0]
    key_scale = next(scale for scale in spec["scales"] if scale.get("labels") == keys)
    assert key_scale["aesthetic"] == "x"
    assert key_scale["breaks"] == [0, 1, 2]


@pytest.mark.parametrize(
    ("transpose", "position", "expected"),
    [(False, "x", [1.5, 2.5]), (True, "y", [1.5, 0.5])],
)
def test_annotated_heatmap_variable_group_by_separates_key_blocks(
    adata, markers, transpose, position, expected
):
    keys = markers[:4]
    data = adata[:, keys].copy()
    data.var["module"] = ["a", "a", "b", "c"]
    spec = cl.annotated_heatmap(
        data, keys=keys, variable_group_by="module", transpose=transpose, max_rows=20
    ).as_dict()

    (lines,) = [layer for layer in spec["layers"] if layer["geom"] == "segment"]
    assert pl.DataFrame(lines["data"])[position].to_list() == expected
    assert lines["color"] == "white"


def test_annotated_heatmap_variable_group_by_needs_group_lines(adata, markers):
    keys = markers[:4]
    data = adata[:, keys].copy()
    data.var["module"] = ["a", "a", "b", "c"]
    spec = cl.annotated_heatmap(
        data, keys=keys, variable_group_by="module", group_lines=False, max_rows=20
    ).as_dict()

    assert not [layer for layer in spec["layers"] if layer["geom"] == "segment"]


def test_spread_labels_keeps_order_in_even_slots():
    # positions 7, 1, 4 on a 10-key axis: slots of 10 / 3 starting at -0.5
    spread = _spread_labels([7, 1, 4], n_keys=10)
    slot = 10 / 3
    assert spread == pytest.approx([-0.5 + 2.5 * slot, -0.5 + 0.5 * slot, -0.5 + 1.5 * slot])


def test_annotated_heatmap_key_labels_label_chosen_keys_with_leader_lines(
    adata, markers, group_key
):
    keys = markers[:4]
    spec = cl.annotated_heatmap(
        adata,
        keys=keys,
        group_by=group_key,
        column_annotations=["highly_variable"],
        column_annotation_position="bottom",
        transpose=True,
        key_labels=[keys[3], keys[0]],
        max_rows=20,
    ).as_dict()

    key_scale = next(scale for scale in spec["scales"] if scale.get("aesthetic") == "y")
    # transposed: first key at the top, so keys[0] takes the upper slot
    assert key_scale["labels"] == [keys[3], keys[0]]
    assert key_scale["breaks"] == pytest.approx([0.5, 2.5])
    # opposite the right-hand column tracks
    assert key_scale["position"] == "left"
    (leaders,) = [
        layer
        for layer in spec["layers"]
        if layer["geom"] == "segment" and "xend" in layer["mapping"] and "yend" in layer["mapping"]
    ]
    leader_frame = pl.DataFrame(leaders["data"])
    assert leader_frame.height == 6
    # lines leave the heatmap's left edge into a margin of the widened x axis
    assert leader_frame["x"].max() == pytest.approx(-0.5)
    observation_scale = next(
        scale
        for scale in spec["scales"]
        if scale.get("aesthetic") == "x" and "ggside" not in scale
    )
    lower, upper = observation_scale["limits"]
    margin = 0.06 * (upper + 0.5) / 0.94
    assert lower == pytest.approx(-0.5 - margin)
    assert leader_frame["xend"].min() == pytest.approx(-0.5 - margin)


def test_annotated_heatmap_key_labels_unknown_key_raises(adata, markers, group_key):
    with pytest.raises(ValueError, match="key_labels"):
        cl.annotated_heatmap(adata, keys=markers[:3], group_by=group_key, key_labels=["nope"])


def test_annotated_heatmap_without_key_labels_labels_every_key(adata, markers, group_key):
    keys = markers[:3]
    spec = cl.annotated_heatmap(adata, keys=keys, group_by=group_key, max_rows=20).as_dict()

    key_scale = next(scale for scale in spec["scales"] if scale.get("aesthetic") == "x")
    assert key_scale["labels"] == keys
    assert "position" not in key_scale
    assert not [layer for layer in spec["layers"] if layer["geom"] == "line"]


def _module_data(adata, keys):
    data = adata[:, keys].copy()
    data.var["module"] = ["a", "a", "b", "c"]
    return data


def test_annotated_heatmap_group_labels_bracket_each_labelled_block(adata, markers):
    keys = markers[:4]
    spec = cl.annotated_heatmap(
        _module_data(adata, keys),
        keys=keys,
        variable_group_by="module",
        group_labels={"a": "Cell adhesion\nLipid metabolism", "c": "Cilium"},
        group_labels_kwargs={"size": 4},
        column_annotation_position="bottom",
        transpose=True,
        max_rows=20,
    ).as_dict()

    lines, texts, reach = (layer for layer in _side_layers(spec, "y"))
    assert (lines["geom"], texts["geom"], reach["geom"]) == ("line", "text", "point")
    text_frame = pl.DataFrame(texts["data"])
    assert text_frame["text"].to_list() == ["Cell adhesion\nLipid metabolism", "Cilium"]
    # transposed, block "a" (keys 0-1) sits at the top, "c" (key 3) at the bottom
    assert text_frame["along"].to_list() == pytest.approx([2.5, 0.0])
    assert texts["hjust"] == 0
    assert texts["size"] == 4
    # four segments (a pointed bracket) per labelled block, all outward of the band edge
    line_frame = pl.DataFrame(lines["data"])
    assert line_frame["segment"].n_unique() == 8
    assert line_frame["cross"].min() == pytest.approx(-0.5)
    assert spec["theme"]["ggside_panel_scale_y"] == pytest.approx(0.03 + 0.25)


def test_annotated_heatmap_group_labels_requires_variable_group_by(adata, markers):
    with pytest.raises(ValueError, match="variable_group_by"):
        cl.annotated_heatmap(adata, keys=markers[:4], group_labels={"a": "text"})


def test_annotated_heatmap_group_labels_unknown_block_raises(adata, markers):
    keys = markers[:4]
    with pytest.raises(ValueError, match="group_labels"):
        cl.annotated_heatmap(
            _module_data(adata, keys),
            keys=keys,
            variable_group_by="module",
            group_labels={"nope": "text"},
        )


@pytest.mark.parametrize("annotation_legends", [True, False])
def test_annotated_heatmap_annotation_legends_toggle_track_legends(
    adata, markers, group_key, cluster_key, annotation_legends
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable", "means"],
        row_annotations=[cluster_key, "pct_counts_mt"],
        annotation_legends=annotation_legends,
        max_rows=20,
    ).as_dict()

    fill_scales = [
        scale
        for scale in spec["scales"]
        if scale.get("aesthetic") == "fill" and "ggside" not in scale
    ]
    tracks = {"highly_variable", "means", cluster_key, "pct_counts_mt"}
    track_scales = [scale for scale in fill_scales if scale.get("name") in tracks]
    assert len(track_scales) == 4
    hidden = [scale.get("guide") == "none" for scale in track_scales]
    assert hidden == [not annotation_legends] * 4
    # the heatmap's color bar keeps its legend either way
    assert all(scale.get("guide") != "none" for scale in fill_scales if scale not in track_scales)


def test_annotated_heatmap_empty_key_labels_label_no_key(adata, markers, group_key):
    spec = cl.annotated_heatmap(
        adata, keys=markers[:3], group_by=group_key, key_labels=[], max_rows=20
    ).as_dict()

    key_scale = next(
        scale
        for scale in spec["scales"]
        if scale.get("aesthetic") == "x" and "ggside" not in scale
    )
    assert key_scale["breaks"] == []
    assert not [
        layer
        for layer in spec["layers"]
        if layer["geom"] == "segment" and "xend" in layer["mapping"] and "yend" in layer["mapping"]
    ]
