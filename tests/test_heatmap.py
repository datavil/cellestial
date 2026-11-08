import polars as pl
import pytest
from ggrust import aes
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.single.heatmap.heatmap import _scale_values
from cellestial.single.heatmap.utilities import _compute_violin_polygons
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
    assert plot.to_svg() is not None


def _stage_custom_dendrogram(adata, group_key, custom_key="dendrogram_custom_v2"):
    """Move the canonical dendrogram aside so only a custom-keyed one is present."""
    import scanpy as sc

    local = adata.copy()
    sc.tl.dendrogram(local, groupby=group_key, key_added=custom_key)
    local.uns.pop(f"dendrogram_{group_key}", None)
    return local, custom_key


def _dendrogram_leaf_order(plot):
    """Leaf labels of the plot's dendrogram scale, in the tree's order."""
    (tree,) = (scale["dendrogram"] for scale in plot.as_dict()["scales"] if "dendrogram" in scale)
    return [tree["labels"][leaf] for leaf in tree["order"]]


@pytest.mark.parametrize("function", ["dotplot", "stacked_violin", "matrixplot"])
def test_dendrogram_leaf_order_matches_scanpy(adata, markers, group_key, function):
    import scanpy as sc

    local = adata.copy()
    sc.tl.dendrogram(local, groupby=group_key)
    plot = getattr(cl, function)(local, keys=markers, group_by=group_key, dendrogram=True)
    expected = list(local.uns[f"dendrogram_{group_key}"]["categories_ordered"])
    assert _dendrogram_leaf_order(plot) == expected


def test_heatmap_dendrogram_key_uses_custom_key(adata, markers, group_key):
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.heatmap(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    # canonical key should NOT have been (re)written by a silent recompute
    assert f"dendrogram_{group_key}" not in local.uns


def test_matrixplot_dendrogram_key_uses_custom_key(adata, markers, group_key):
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.matrixplot(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    assert f"dendrogram_{group_key}" not in local.uns


def test_dotplot_dendrogram_key_uses_custom_key(adata, markers, group_key):
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.dotplot(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    assert f"dendrogram_{group_key}" not in local.uns


def test_stacked_violin_dendrogram_key_uses_custom_key(adata, markers, group_key):
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.stacked_violin(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    assert f"dendrogram_{group_key}" not in local.uns


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
        fill_statistic="group",
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
        fill_statistic="variable",
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
        aggregate_column="expression",
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
        aggregate_column="expression",
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
            aggregate_column="expression",
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
    from cellestial.single.heatmap.utilities import _resolve_key_groups

    flat, groups = _resolve_key_groups(["A", "B", "C"])
    assert flat == ["A", "B", "C"]
    assert groups is None


def test_resolve_key_groups_dict_flatten_preserves_order():
    from cellestial.single.heatmap.utilities import _resolve_key_groups

    flat, groups = _resolve_key_groups({"G1": ["A", "B"], "G2": ["C"]})
    assert flat == ["A", "B", "C"]
    assert groups == {"G1": ["A", "B"], "G2": ["C"]}


def test_resolve_key_groups_dict_rejects_single_string_value():
    from cellestial.single.heatmap.utilities import _resolve_key_groups

    with pytest.raises(TypeError, match="not a single string"):
        _resolve_key_groups({"G1": "A"})


def test_dotplot_dict_keys_keep_tight_y_limit_with_brackets(adata, group_key):
    groups = _marker_groups()
    plot = cl.dotplot(adata, group_by=group_key, keys=groups)
    n_y = adata.obs[group_key].nunique()
    spec = plot.as_dict()
    scales = [s for s in spec.get("scales", []) if s.get("aesthetic") == "y" and "limits" in s]
    assert scales, "dotplot should keep explicit y limits"
    # Brackets are an axis guide outside the panel, so the y limit stays tight.
    assert scales[-1]["limits"][1] == n_y - 0.5


def test_dotplot_dict_keys_without_key_labels_keeps_tight_y_limit(adata, group_key):
    groups = _marker_groups()
    plot = cl.dotplot(adata, group_by=group_key, keys=groups, key_labels=False)
    n_y = adata.obs[group_key].nunique()
    spec = plot.as_dict()
    scales = [s for s in spec.get("scales", []) if s.get("aesthetic") == "y" and "limits" in s]
    assert scales, "dotplot should keep explicit y limits"
    assert scales[-1]["limits"][1] == n_y - 0.5


# ---- dotplot ----


def test_dotplot_basic(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, PlotSpec)


def test_dotplot_sort_by_avg_exp(adata, markers, group_key):
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, sort_by="avg_exp")
    assert isinstance(plot, PlotSpec)


def test_dotplot_fill_mapping(adata, markers, group_key):
    from ggrust import aes

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
    with pytest.raises(UnsupportedDataTypeError, match="Expected"):
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
    from ggrust import layer_tooltips

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


def test_heatmap_variable_axis_aggregate(adata):
    # Regression: on the variable axis the `keys` are variable metadata columns
    # (not genes from X), so they must be materialised via metadata_columns or
    # the unpivot raises ColumnNotFoundError.
    plot = cl.heatmap(
        adata,
        keys=["mean_counts", "log1p_mean_counts"],
        group_by="mt",
        axis=1,
        aggregate=True,
    )
    assert isinstance(plot, PlotSpec)


def test_heatmap_variable_axis_nonaggregate(adata):
    """Use variable identifiers for nonaggregated variable-axis rows."""
    local = adata[:, :100].copy()
    plot = cl.heatmap(
        local,
        keys=["mean_counts", "log1p_mean_counts"],
        group_by="mt",
        axis=1,
        aggregate=False,
        variable_id_column="Feature",
        max_rows=None,
    )

    assert isinstance(plot, PlotSpec)
    assert plot.as_dict()["data"]["Feature"].n_unique() == local.n_vars
    y_scale = next(scale for scale in plot.as_dict()["scales"] if scale.get("aesthetic") == "y")
    half_step = 1 / (local.n_vars - 1) / 2
    assert y_scale["limits"] == pytest.approx([-half_step, 1 + half_step])


def test_heatmap_variable_axis_single_key_preserves_row_span(adata):
    """Keep usable row spacing and aligned separators with one heatmap column."""
    local = adata[:, :100].copy()
    local.var["test_group"] = ["a"] * 50 + ["b"] * 50
    plot = cl.heatmap(
        local,
        keys=["mean_counts"],
        group_by="test_group",
        axis=1,
        aggregate=False,
        variable_id_column="Feature",
        max_rows=None,
    )

    spec = plot.as_dict()
    y_scale = next(scale for scale in spec["scales"] if scale.get("aesthetic") == "y")
    half_step = 1 / (local.n_vars - 1) / 2
    assert y_scale["limits"] == pytest.approx([-half_step, 1 + half_step])

    group_lines = [
        layer["data"]
        for layer in spec["layers"]
        if layer.get("geom") == "segment"
        and hasattr(layer.get("data"), "columns")
        and "xend" in layer["data"].columns
    ]
    assert group_lines[-1]["y"].to_list() == pytest.approx([0.5])


@pytest.mark.parametrize(
    ("scale_axis", "max_rows"),
    [
        (1, None),
        (None, 10),
    ],
)
def test_heatmap_variable_axis_nonaggregate_row_operations(adata, scale_axis, max_rows):
    """Keep the variable identifier through row-wise transformations."""
    local = adata[:, :100].copy()
    plot = cl.heatmap(
        local,
        keys=["mean_counts", "log1p_mean_counts"],
        group_by="mt",
        axis=1,
        aggregate=False,
        variable_id_column="Feature",
        scale_axis=scale_axis,
        max_rows=max_rows,
    )

    assert isinstance(plot, PlotSpec)
    assert "Feature" in plot.as_dict()["data"].columns
