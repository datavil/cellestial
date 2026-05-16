import polars as pl
import pytest
from lets_plot import aes
from lets_plot.plot.core import PlotSpec

import cellestial as cl
from cellestial.single.heatmap.heatmap import _scale_values
from cellestial.single.heatmap.stacked_violin import _compute_violin_polygons
from cellestial.util.errors import UnsupportedDataTypeError


def test_heatmap_default_aggregate(adata, markers, group_key):
    plot = cl.heatmap(adata, group_by=group_key, keys=markers)
    assert isinstance(plot, PlotSpec)


def test_heatmap_tile_with_tooltips(adata, markers, group_key):
    plot = cl.heatmap(adata, group_by=group_key, keys=markers, geom="tile", tooltips=["value"])
    assert isinstance(plot, PlotSpec)


def test_heatmap_dendrogram(adata, markers, group_key):
    plot = cl.heatmap(adata, group_by=group_key, keys=markers, dendrogram=True)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("scale_axis", [0, 1])
def test_heatmap_scale_axis(adata, markers, group_key, scale_axis):
    plot = cl.heatmap(adata, group_by=group_key, keys=markers, scale_axis=scale_axis)
    assert isinstance(plot, PlotSpec)


def test_heatmap_scale_values_constant_partition_returns_zero():
    frame = pl.DataFrame({"group": ["a", "a", "b"], "value": [2.0, 2.0, 5.0]})
    scaled = _scale_values(frame, value_column="value", partition_key="group")
    assert scaled["value"].to_list() == [0.0, 0.0, 0.0]


def test_heatmap_aggregate_false(adata, markers, group_key):
    # Non-aggregated: one row per cell. Keep small: few cells x 5 genes.
    sub = adata[:300].copy()
    plot = cl.heatmap(sub, group_by=group_key, keys=markers[:5], aggregate=False, geom="raster")
    assert isinstance(plot, PlotSpec)


def test_heatmap_aggregate_false_with_group_bars(adata, markers, group_key):
    sub = adata[:300].copy()
    plot = cl.heatmap(
        sub,
        group_by=group_key,
        keys=markers[:5],
        aggregate=False,
        geom="raster",
        group_bars=True,
    )
    assert isinstance(plot, PlotSpec)


def test_heatmap_invalid_group_by(adata, markers):
    with pytest.raises(Exception):
        cl.heatmap(adata, group_by="NOT_AN_OBS_COL_xyz", keys=markers)


def test_heatmap_invalid_gene(adata, group_key):
    with pytest.raises(Exception):
        cl.heatmap(adata, group_by=group_key, keys=["NOT_A_GENE_xyz"])


# ---- key groups (dict keys) ----


def _marker_groups():
    return {
        "B-cell": ["CD79A", "MS4A1"],
        "NK": ["NKG7"],
        "T-cell": ["CD3D"],
    }


def test_heatmap_dict_keys_aggregate(adata, group_key):
    plot = cl.heatmap(adata, group_by=group_key, keys=_marker_groups(), aggregate=True)
    assert isinstance(plot, PlotSpec)


def test_heatmap_dict_keys_non_aggregate(adata, group_key):
    sub = adata[:300].copy()
    plot = cl.heatmap(sub, group_by=group_key, keys=_marker_groups(), aggregate=False)
    assert isinstance(plot, PlotSpec)


def test_matrixplot_dict_keys(adata, group_key):
    plot = cl.matrixplot(adata, group_by=group_key, keys=_marker_groups())
    assert isinstance(plot, PlotSpec)


def test_dotplot_dict_keys(adata, group_key):
    plot = cl.dotplot(adata, group_by=group_key, keys=_marker_groups())
    assert isinstance(plot, PlotSpec)


def test_stacked_violin_dict_keys(adata, group_key):
    plot = cl.stacked_violin(adata, group_by=group_key, keys=_marker_groups())
    assert isinstance(plot, PlotSpec)


def test_stacked_violin_options(adata, group_key):
    sub = adata[:300].copy()
    plot = cl.stacked_violin(
        sub,
        group_by=group_key,
        keys=_marker_groups(),
        mapping=aes(alpha="expression"),
        threshold=0.0,
        scale="count",
        color_by="group",
        geom_fill=None,
        rectangle=False,
        key_labels=False,
        interactive=True,
        n_points=16,
    )
    assert isinstance(plot, PlotSpec)


def test_stacked_violin_variable_fill_without_rectangle(adata, group_key):
    sub = adata[:300].copy()
    plot = cl.stacked_violin(
        sub,
        group_by=group_key,
        keys=(markers := ["CD79A", "MS4A1"]),
        color_by="variable",
        rectangle_kwargs={"alpha": 0.5},
        n_points=16,
    )
    assert isinstance(plot, PlotSpec)
    assert markers == ["CD79A", "MS4A1"]


def test_stacked_violin_required_inputs_and_data_type(adata):
    with pytest.raises(ValueError, match="keys"):
        cl.stacked_violin(adata)
    with pytest.raises(UnsupportedDataTypeError):
        cl.stacked_violin("not adata", keys=["A"], group_by="group")


def test_compute_violin_polygons_edge_paths():
    frame = pl.DataFrame(
        {
            "variable": ["a", "a", "a", "b"],
            "group": ["g1", "g1", "g1", "g1"],
            "value": [1.0, 2.0, 3.0, 5.0],
        }
    )

    polygons = _compute_violin_polygons(
        frame,
        variable_column="variable",
        value_column="value",
        group_by="group",
        x_keys=["a", "missing", "b"],
        y_order_groups=["g1"],
        n_points=8,
        scale="area",
        width_scale=0.8,
        height_scale=0.8,
        aggregate="mean",
        aggregate_key="expression",
    )
    empty = _compute_violin_polygons(
        frame,
        variable_column="variable",
        value_column="value",
        group_by="group",
        x_keys=["b"],
        y_order_groups=["g1"],
        n_points=8,
        scale="width",
        width_scale=0.8,
        height_scale=0.8,
        aggregate="median",
        aggregate_key="expression",
    )

    assert polygons.height > 0
    assert empty.is_empty()
    with pytest.raises(ValueError, match="scale"):
        _compute_violin_polygons(
            frame,
            variable_column="variable",
            value_column="value",
            group_by="group",
            x_keys=["a"],
            y_order_groups=["g1"],
            n_points=8,
            scale="bad",
            width_scale=0.8,
            height_scale=0.8,
            aggregate="median",
            aggregate_key="expression",
        )


def test_heatmap_dict_keys_duplicate_raises(adata, group_key):
    from cellestial.util.errors import DuplicateKeysError

    with pytest.raises(DuplicateKeysError, match="multiple groups"):
        cl.heatmap(
            adata,
            group_by=group_key,
            keys={"A": ["CD3D"], "B": ["CD3D"]},
        )


# ---- key groups internals ----


def test_resolve_key_groups_flat_passthrough():
    from cellestial.single.heatmap._key_groups import _resolve_key_groups

    flat, groups = _resolve_key_groups(["A", "B", "C"])
    assert flat == ["A", "B", "C"]
    assert groups is None


def test_resolve_key_groups_dict_flatten_preserves_order():
    from cellestial.single.heatmap._key_groups import _resolve_key_groups

    flat, groups = _resolve_key_groups({"G1": ["A", "B"], "G2": ["C"]})
    assert flat == ["A", "B", "C"]
    assert groups == {"G1": ["A", "B"], "G2": ["C"]}


def test_resolve_key_groups_dict_rejects_single_string_value():
    from cellestial.single.heatmap._key_groups import _resolve_key_groups

    with pytest.raises(TypeError, match="not a single string"):
        _resolve_key_groups({"G1": "A"})


def test_key_groups_bar_y_above_top_edge():
    from cellestial.single.heatmap._key_groups import _key_groups_bar_y

    top_edge = 7.5
    bar_y = _key_groups_bar_y(top_edge)
    assert bar_y > top_edge, "bracket bar must sit above the plot top edge"
    assert bar_y - top_edge < 1.0, "bracket bar should be close to the top edge"


def test_resolve_padding_scales_with_label_length():
    from cellestial.single.heatmap._key_groups import _resolve_padding

    short = _resolve_padding({"A": ["x"], "B": ["y"]}, padding=None)
    long = _resolve_padding({"Lymphocytes": ["x"], "Monocytes": ["y"]}, padding=None)
    assert short < long, "longer labels need more padding"
    assert short >= 1.5, "padding has a sensible floor"


def test_resolve_padding_explicit_overrides_auto():
    from cellestial.single.heatmap._key_groups import _resolve_padding

    explicit = _resolve_padding({"Lymphocytes": ["x"]}, padding=0.5)
    assert explicit == 0.5


def test_key_groups_x_unit_label_size_ignores_y_span():
    from cellestial.single.heatmap._key_groups import _key_groups_layers

    groups = {"Group_1": ["A", "B"], "Group_2": ["C", "D"]}
    small_y_text = _key_groups_layers(
        groups,
        y=3.5,
        total_span=4,
        label_size_span=4,
        size_unit="x",
    )[-1].as_dict()
    large_y_text = _key_groups_layers(
        groups,
        y=15.5,
        total_span=16,
        label_size_span=4,
        size_unit="x",
    )[-1].as_dict()
    assert small_y_text["size_unit"] == "x"
    assert large_y_text["size_unit"] == "x"
    assert small_y_text["size"] == large_y_text["size"]


def test_dotplot_dict_keys_extend_y_limit_for_brackets(adata, group_key):
    groups = _marker_groups()
    plot = cl.dotplot(adata, group_by=group_key, keys=groups)
    n_y = adata.obs[group_key].nunique()
    spec = plot.as_dict()
    scales = [
        s
        for s in spec.get("scales", [])
        if s.get("aesthetic") == "y" and "limits" in s
    ]
    assert scales, "dotplot should keep explicit y limits"
    # Brackets are drawn above the data area, so y limit must extend past it.
    assert scales[-1]["limits"][1] > n_y - 0.5


def test_dotplot_dict_keys_without_key_labels_keeps_tight_y_limit(adata, group_key):
    groups = _marker_groups()
    plot = cl.dotplot(adata, group_by=group_key, keys=groups, key_labels=False)
    n_y = adata.obs[group_key].nunique()
    spec = plot.as_dict()
    scales = [
        s
        for s in spec.get("scales", [])
        if s.get("aesthetic") == "y" and "limits" in s
    ]
    assert scales, "dotplot should keep explicit y limits"
    assert scales[-1]["limits"][1] == n_y - 0.5


def test_dotplot_key_labels_use_x_unit_text_size(adata, group_key):
    plot = cl.dotplot(adata, group_by=group_key, keys=_marker_groups())
    text_layers = [
        layer for layer in plot.as_dict()["layers"] if layer.get("geom") == "text"
    ]
    assert text_layers[-1]["size_unit"] == "x"


def test_stacked_violin_key_labels_use_absolute_text_size(adata, group_key):
    plot = cl.stacked_violin(adata, group_by=group_key, keys=_marker_groups())
    text_layers = [
        layer for layer in plot.as_dict()["layers"] if layer.get("geom") == "text"
    ]
    assert "size_unit" not in text_layers[-1]
    assert text_layers[-1]["size"] >= 3.75


# ---- dotplot ----


def test_dotplot_basic(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, PlotSpec)


def test_dotplot_sort_by_avg_exp(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, sort_by="avg_exp")
    assert isinstance(plot, PlotSpec)


def test_dotplot_fill_mapping(adata, markers, group_key):
    from lets_plot import aes

    plot = cl.dotplot(adata, keys=markers, group_by=group_key, mapping=aes(fill="avg_exp"))
    assert isinstance(plot, PlotSpec)


def test_dotplot_custom_colors(adata, markers, group_key):
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        color_low="lightgrey",
        color_high="red",
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_threshold(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, threshold=0.2)
    assert isinstance(plot, PlotSpec)


def test_dotplot_invalid_gene(adata, group_key):
    with pytest.raises(Exception):
        cl.dotplot(adata, keys=["NOT_A_GENE_xyz"], group_by=group_key)


def test_dotplot_dendrogram(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, dendrogram=True)
    assert isinstance(plot, PlotSpec)
    # Render to SVG so lets-plot validates the computed x/y limits against all layers.
    assert plot.to_svg() is not None


def test_dotplot_dendrogram_without_rectangle(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, dendrogram=True, rectangle=False)
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


def test_dotplot_single_key(adata, group_key):
    # n_x == 1: x limits collapse to [-0.5, 0.5]; dendrogram extends rightward.
    plot = cl.dotplot(adata, keys=["CD3D"], group_by=group_key, dendrogram=True)
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


def test_dotplot_dendrogram_custom_style(adata, markers, group_key):
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        dendrogram=True,
        dendrogram_color="blue",
        dendrogram_size=1.2,
    )
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


def test_dotplot_no_dendrogram_no_rectangle(adata, markers, group_key):
    # Fallback path: x_max_limit should default to n_x - 0.5.
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, dendrogram=False, rectangle=False)
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


def test_dotplot_invalid_data_type(markers, group_key):
    with pytest.raises(UnsupportedDataTypeError, match="Unsupported data type"):
        cl.dotplot({"not": "anndata"}, keys=markers, group_by=group_key)


def test_dotplot_tooltips_sequence(adata, markers, group_key):
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        tooltips=[group_key, "variable"],
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_tooltips_missing_column(adata, markers, group_key):
    with pytest.raises(ValueError, match="not in data"):
        cl.dotplot(
            adata,
            keys=markers,
            group_by=group_key,
            tooltips=["nonexistent_column"],
        )


def test_dotplot_tooltips_feature_spec(adata, markers, group_key):
    from lets_plot import layer_tooltips

    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        tooltips=layer_tooltips([group_key, "variable"]),
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_tooltips_none_string(adata, markers, group_key):
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        tooltips="none",
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_interactive(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, interactive=True)
    assert isinstance(plot, PlotSpec)


# ---- highest_expressed_genes ----


def test_highest_expressed_genes_default(adata):
    plot = cl.highest_expressed_genes(adata, n=10)
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_threshold(adata):
    plot = cl.highest_expressed_genes(adata, n=10, threshold=0.1)
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_custom_outliers(adata):
    plot = cl.highest_expressed_genes(
        adata, n=5, outlier_size=0.1, outlier_alpha=0.2, outlier_shape=5
    )
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_n_larger_than_var(adata):
    # n larger than var count should clip gracefully, not crash.
    with pytest.raises(ValueError):
        cl.highest_expressed_genes(adata, n=adata.n_vars + 100)
