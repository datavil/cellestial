import polars as pl
import pytest
from ggrust import element_text, geom_blank, theme
from ggrust.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.complex.annotated_heatmap import (
    _bin_observations,
    _bin_variables,
    _run_outlines,
)
from cellestial.util.errors import UnsupportedDataTypeError


def test_annotated_heatmap_default(adata, markers, group_key):
    plot = cl.annotated_heatmap(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_both_axes(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key, "pct_counts_mt"],
        scale_axis=0,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_numeric_column_annotation(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, column_annotations=["means"]
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_string_annotations(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations="highly_variable",
        row_annotations=cluster_key,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_no_legend(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, row_annotations=[cluster_key], legend=False
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_no_group_by(adata, markers):
    plot = cl.annotated_heatmap(adata, keys=markers, row_annotations="pct_counts_mt")
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_max_rows(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, row_annotations=[cluster_key], max_rows=200
    )
    assert isinstance(plot, SupPlotsSpec)


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
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_dendrogram(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        row_annotations=[group_key],
        dendrogram=True,
    )
    assert isinstance(plot, SupPlotsSpec)


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
    assert isinstance(plot, SupPlotsSpec)


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

    assert spec["layout"]["widths"] == [0.025, 1.0]
    assert spec["layout"]["heights"] == [0.025, 1.0]
    assert spec["figures"][0] is None
    heatmap = spec["figures"][3]
    assert heatmap["layers"][0]["mapping"] == {
        "x": "position_y",
        "y": "position_x",
        "fill": "value",
    }
    assert heatmap["layers"][1]["mapping"] == {"x": "x", "xend": "x"}


def test_annotated_heatmap_right_and_bottom_track_positions(
    adata, markers, group_key, cluster_key
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key],
        row_annotation_position="right",
        column_annotation_position="bottom",
        max_rows=20,
    ).as_dict()

    assert spec["layout"]["widths"] == [1.0, 0.025]
    assert spec["layout"]["heights"] == [1.0, 0.025]
    assert spec["figures"][3] is None
    assert spec["figures"][0]["layers"][0]["mapping"] == {
        "x": "position_x",
        "y": "position_y",
        "fill": "value",
    }


def test_annotated_heatmap_transpose_moves_dendrogram_above_heatmap(adata, markers, group_key):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        dendrogram=True,
        transpose=True,
        max_rows=20,
    ).as_dict()

    assert spec["layout"]["ncol"] == 1
    assert spec["layout"]["nrow"] == 2
    assert spec["figures"][0]["layers"][0]["geom"] == "path"
    assert spec["figures"][1]["layers"][0]["geom"] == "raster"


def test_annotated_heatmap_layers_and_layers_all_have_distinct_scope(
    adata, markers, group_key, cluster_key
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key],
        layers=geom_blank(),
        layers_all=geom_blank(),
        max_rows=20,
    ).as_dict()
    panels = [panel for panel in spec["figures"] if panel is not None]
    blank_counts = [sum(layer["geom"] == "blank" for layer in panel["layers"]) for panel in panels]

    assert sorted(blank_counts) == [1, 1, 2]


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

    heatmap_layer = next(
        layer
        for panel in plot.as_dict()["figures"]
        if panel is not None
        for layer in panel["layers"]
        if layer["geom"] == "raster"
    )
    assert "tooltips" not in heatmap_layer


def _layers_of(panel: dict, geom: str) -> list[dict]:
    return [layer for layer in panel["layers"] if layer["geom"] == geom]


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
    strip, heatmap = spec["figures"][0], spec["figures"][1]

    first_group = adata.obs[group_key].iloc[0]
    leftmost = strip["data"].sort("position_y")[group_key][0]
    assert leftmost == first_group

    key_scale = next(scale for scale in heatmap["scales"] if scale.get("labels") == keys)
    assert key_scale["aesthetic"] == "y"
    top_key = keys[key_scale["breaks"].index(max(key_scale["breaks"]))]
    assert top_key == keys[0]


def test_annotated_heatmap_upright_keeps_first_group_top_and_first_key_left(
    adata, markers, group_key
):
    keys = markers[:3]
    spec = cl.annotated_heatmap(
        adata, keys=keys, group_by=group_key, row_annotations=[group_key], max_rows=40
    ).as_dict()
    strip, heatmap = spec["figures"][0], spec["figures"][1]

    topmost = strip["data"].sort("position_y", descending=True)[group_key][0]
    assert topmost == adata.obs[group_key].iloc[0]
    key_scale = next(scale for scale in heatmap["scales"] if scale.get("labels") == keys)
    assert key_scale["aesthetic"] == "x"
    assert key_scale["breaks"] == [0, 1, 2]


def test_run_outlines_spans_each_block_of_equal_values():
    frame = pl.DataFrame({"position": [0, 1, 2, 3, 4], "value": ["a", "a", "b", None, "a"]})
    runs = _run_outlines(frame, position="position", value="value")
    assert runs.rows() == [(-0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5)]


def test_annotated_heatmap_annotation_outline_draws_block_borders(
    adata, markers, group_key, cluster_key
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=cluster_key,
        column_annotations="highly_variable",
        row_annotations=[group_key, cluster_key],
        aggregate="mean",
        annotation_outline="black",
        annotation_outline_size=0.8,
    ).as_dict()
    strips = [panel for panel in spec["figures"] if panel is not None][:-1]
    row_strips = [panel for panel in strips if "position_y" in panel["data"].columns]
    for panel in strips:
        (outline,) = _layers_of(panel, "rect")
        assert outline["color"] == "black"
        assert outline["size"] == 0.8

    # One border per cluster (each a single pseudobulk cell), fewer per cell type.
    group_strip, cluster_strip = row_strips
    n_clusters = adata.obs[cluster_key].nunique()
    assert _layers_of(cluster_strip, "rect")[0]["data"].height == n_clusters
    group_runs = _layers_of(group_strip, "rect")[0]["data"].height
    assert group_runs < n_clusters


def test_annotated_heatmap_no_outline_by_default(adata, markers, group_key):
    spec = cl.annotated_heatmap(
        adata, keys=markers[:2], group_by=group_key, row_annotations=group_key, max_rows=20
    ).as_dict()
    assert all(not _layers_of(panel, "rect") for panel in spec["figures"] if panel is not None)


def test_annotated_heatmap_annotation_spacing_sets_gaps_and_drops_insets(
    adata, markers, group_key
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        column_annotations="highly_variable",
        row_annotations=group_key,
        dendrogram=True,
        annotation_spacing=0,
        max_rows=20,
    ).as_dict()
    assert spec["layout"]["hspace"] == 0
    assert spec["layout"]["vspace"] == 0
    panels = [panel for panel in spec["figures"] if panel is not None]
    assert len(panels) == 4
    assert all(panel["theme"]["plot_inset"] == 0 for panel in panels)


def test_annotated_heatmap_default_spacing_keeps_insets(adata, markers, group_key):
    spec = cl.annotated_heatmap(
        adata, keys=markers[:2], group_by=group_key, row_annotations=group_key, max_rows=20
    ).as_dict()
    assert spec["layout"].get("vspace") is None
    assert all("plot_inset" not in panel["theme"] for panel in spec["figures"] if panel)


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_annotated_heatmap_aggregate_plots_one_row_per_group(adata, markers, group_key, statistic):
    keys = markers[:2]
    spec = cl.annotated_heatmap(
        adata,
        keys=keys,
        group_by=group_key,
        row_annotations=[group_key, "pct_counts_mt"],
        aggregate=statistic,
        max_rows=2,  # ignored when aggregating
    ).as_dict()
    group_strip, numeric_strip, heatmap = spec["figures"][:3]
    n_groups = adata.obs[group_key].nunique(dropna=False)
    assert group_strip["data"].height == n_groups
    assert heatmap["data"]["position_y"].n_unique() == n_groups

    # The top row is the first group, summarized with `statistic`.
    first_group = adata.obs[group_key].iloc[0]
    in_group = (adata.obs[group_key] == first_group).to_numpy()
    expected = getattr(adata[in_group, keys[0]].to_df()[keys[0]], statistic)()
    top = heatmap["data"].filter(
        (pl.col("position_y") == n_groups - 1) & (pl.col("variable") == keys[0])
    )
    assert top["value"][0] == pytest.approx(expected)
    expected_mt = getattr(adata.obs.loc[in_group, "pct_counts_mt"], statistic)()
    top_mt = numeric_strip["data"].sort("position_y")["pct_counts_mt"][-1]
    assert top_mt == pytest.approx(expected_mt)


def test_annotated_heatmap_aggregate_requires_group_by(adata, markers):
    with pytest.raises(ValueError, match="`group_by` is required when `aggregate` is set"):
        cl.annotated_heatmap(adata, keys=markers, aggregate="mean")


def test_annotated_heatmap_aggregate_rejects_unknown_statistic(adata, markers, group_key):
    with pytest.raises(ValueError, match="`aggregate` must be 'mean', 'median', or None"):
        cl.annotated_heatmap(adata, keys=markers, group_by=group_key, aggregate="sum")


def test_bin_variables_keeps_blocks_intact():
    keys = [f"g{index}" for index in range(10)]
    blocks = [("a",)] * 6 + [("b",)] * 4
    bins = _bin_variables(keys, blocks=blocks, max_columns=5)
    assert bins == [["g0", "g1"], ["g2", "g3"], ["g4", "g5"], ["g6", "g7"], ["g8", "g9"]]
    assert all(
        len({blocks[keys.index(key)] for key in variable_bin}) == 1 for variable_bin in bins
    )


def test_bin_variables_gives_every_block_a_bin():
    keys = ["g0", "g1", "g2", "g3", "g4"]
    blocks = [("a",)] * 4 + [("b",)]
    bins = _bin_variables(keys, blocks=blocks, max_columns=2)
    assert bins == [["g0", "g1"], ["g2", "g3"], ["g4"]]


def test_bin_variables_noop_when_small():
    assert _bin_variables(["g0", "g1"], blocks=None, max_columns=5) is None
    assert _bin_variables(["g0", "g1"], blocks=None, max_columns=None) is None


def test_annotated_heatmap_max_columns_averages_contiguous_keys(adata, group_key):
    # highly_variable is False for CD3D and True for the rest: two blocks.
    keys = ["CD3D", "CD8A", "MS4A1", "NKG7"]
    spec = cl.annotated_heatmap(
        adata,
        keys=keys,
        group_by=group_key,
        column_annotations="highly_variable",
        max_columns=2,
        max_rows=None,
    ).as_dict()
    column_strip, heatmap = spec["figures"]
    labels = next(scale for scale in heatmap["scales"] if scale.get("labels"))["labels"]
    assert labels == ["CD3D", "CD8A..MS4A1", "NKG7"]
    assert column_strip["data"]["highly_variable"].to_list() == [False, True, True]

    # A bin value is the mean of its keys for that observation.
    expected = adata[:, ["CD8A", "MS4A1"]].to_df().mean(axis=1).iloc[0]
    first_row = heatmap["data"]["position_y"].max()
    value = heatmap["data"].filter(
        (pl.col("position_y") == first_row) & (pl.col("variable") == "CD8A..MS4A1")
    )["value"][0]
    assert value == pytest.approx(expected)


def test_annotated_heatmap_layers_legend_theme_reaches_collected_legends(
    adata, markers, group_key, cluster_key
):
    spec = cl.annotated_heatmap(
        adata,
        keys=markers[:2],
        group_by=group_key,
        row_annotations=[cluster_key],
        layers=theme(
            legend_text=element_text(size=20),
            legend_title=element_text(size=24),
            legend_position="none",
        ),
        max_rows=20,
    ).as_dict()
    strip, heatmap = spec["figures"][0], spec["figures"][1]
    for panel in (strip, heatmap):
        assert panel["theme"]["legend_text"]["size"] == 20
        assert panel["theme"]["legend_title"]["size"] == 24
    # Visibility stays per panel: only the heatmap's legend is hidden.
    assert heatmap["theme"]["legend_position"] == "none"
    assert strip["theme"]["legend_position"] == "bottom"
