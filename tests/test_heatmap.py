import re

import polars as pl
import pytest
from ggrust import aes, ggsize
from ggrust.plot.core import PlotSpec
from polars.testing import assert_frame_equal

import cellestial as cl
from cellestial.single.heatmap import utilities as heatmap_utilities
from cellestial.single.heatmap.dotplot import _DOT_SIZE_MAX, _POINT_PIXELS_PER_SIZE
from cellestial.single.heatmap.heatmap import _X_EXPAND, _scale_values
from cellestial.single.heatmap.utilities import (
    _CELL_SIZE,
    _PANEL_CLIP_PATTERN,
    _PANEL_MIN_SIZE,
    _compute_violin_polygons,
    _panel_size,
)
from cellestial.util.errors import (
    KeyNotFoundError,
    UnsupportedDataTypeError,
    VariableNotFoundError,
)


def test_heatmap_default_aggregate(adata, markers, group_key):
    """`heatmap` renders with its defaults (aggregated per group)."""
    plot = cl.heatmap(adata, group_by=group_key, keys=markers)
    assert isinstance(plot, PlotSpec)


def test_heatmap_tile_with_tooltips(adata, markers, group_key):
    """`heatmap` with `geom='tile'` accepts tooltips."""
    plot = cl.heatmap(adata, group_by=group_key, keys=markers, geom="tile", tooltips=["value"])
    assert isinstance(plot, PlotSpec)


def test_heatmap_dendrogram(adata, markers, group_key):
    """`heatmap` renders with a dendrogram."""
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


def _y_labels(plot):
    """Labels of the plot's y scale."""
    return [scale["labels"] for scale in plot.as_dict()["scales"] if scale.get("aesthetic") == "y"]


@pytest.mark.parametrize("function", ["heatmap", "matrixplot", "dotplot", "stacked_violin"])
def test_group_order_is_deterministic(adata, markers, group_key, function):
    """Group order is the same on every call."""
    keyword_arguments = {"aggregate": True} if function == "heatmap" else {}
    orders = [
        _y_labels(
            getattr(cl, function)(adata, keys=markers, group_by=group_key, **keyword_arguments)
        )
        for _ in range(10)
    ]
    assert orders[0]
    assert all(order == orders[0] for order in orders)
    # Groups follow their first appearance in the data, as in the per-cell view.
    first_appearance = adata.obs[group_key].dropna().astype(str).unique().tolist()
    assert orders[0][0] == first_appearance


@pytest.mark.parametrize("function", ["dotplot", "stacked_violin", "matrixplot"])
def test_dendrogram_leaf_order_matches_scanpy(adata, markers, group_key, function):
    """Groups follow the dendrogram leaf order scanpy computes."""
    import scanpy as sc

    local = adata.copy()
    sc.tl.dendrogram(local, groupby=group_key)
    plot = getattr(cl, function)(local, keys=markers, group_by=group_key, dendrogram=True)
    expected = list(local.uns[f"dendrogram_{group_key}"]["categories_ordered"])
    assert _dendrogram_leaf_order(plot) == expected


def test_heatmap_dendrogram_key_uses_custom_key(adata, markers, group_key):
    """`heatmap` reads a dendrogram stored under a custom key."""
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.heatmap(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    # canonical key should NOT have been (re)written by a silent recompute
    assert f"dendrogram_{group_key}" not in local.uns


def test_matrixplot_dendrogram_key_uses_custom_key(adata, markers, group_key):
    """`matrixplot` reads a dendrogram stored under a custom key."""
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.matrixplot(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    assert f"dendrogram_{group_key}" not in local.uns


def test_dotplot_dendrogram_key_uses_custom_key(adata, markers, group_key):
    """`dotplot` reads a dendrogram stored under a custom key."""
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.dotplot(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    assert f"dendrogram_{group_key}" not in local.uns


def test_stacked_violin_dendrogram_key_uses_custom_key(adata, markers, group_key):
    """`stacked_violin` reads a dendrogram stored under a custom key."""
    local, custom = _stage_custom_dendrogram(adata, group_key)
    plot = cl.stacked_violin(
        local, group_by=group_key, keys=markers, dendrogram=True, dendrogram_key=custom
    )
    assert isinstance(plot, PlotSpec)
    assert f"dendrogram_{group_key}" not in local.uns


@pytest.mark.parametrize("scale_axis", [0, 1])
def test_heatmap_scale_axis(adata, markers, group_key, scale_axis):
    """`heatmap` renders with each `scale_axis` value."""
    plot = cl.heatmap(adata, group_by=group_key, keys=markers, scale_axis=scale_axis)
    assert isinstance(plot, PlotSpec)


def test_heatmap_scale_values_constant_partition_returns_zero():
    """Min-max scaling a constant partition gives 0, not NaN."""
    frame = pl.DataFrame({"group": ["a", "a", "b"], "value": [2.0, 2.0, 5.0]})
    scaled = _scale_values(frame, value_column="value", partition_key="group")
    assert scaled["value"].to_list() == [0.0, 0.0, 0.0]


def test_heatmap_aggregate_false(adata, markers, group_key):
    """`heatmap(aggregate=False)` draws one row per cell."""
    # Non-aggregated: one row per cell. Keep small: few cells x 5 genes.
    sub = adata[:300].copy()
    plot = cl.heatmap(sub, group_by=group_key, keys=markers[:5], aggregate=False, geom="raster")
    assert isinstance(plot, PlotSpec)


def test_heatmap_aggregate_false_with_group_bars(adata, markers, group_key):
    """The per-cell `heatmap` renders with group bars."""
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
    """An unknown `group_by` column raises `KeyError`."""
    with pytest.raises(KeyError):
        cl.heatmap(adata, group_by="NOT_AN_OBS_COL_xyz", keys=markers)


def test_heatmap_invalid_gene(adata, group_key):
    """An unknown key raises `VariableNotFoundError`."""
    with pytest.raises(VariableNotFoundError):
        cl.heatmap(adata, group_by=group_key, keys=["NOT_A_GENE_xyz"])


# ---- key groups (dict keys) ----


def _marker_groups():
    return {
        "B-cell": ["CD79A", "MS4A1"],
        "NK": ["NKG7"],
        "T-cell": ["CD3D"],
    }


def test_heatmap_dict_keys_aggregate(adata, group_key):
    """Mapping `keys` render in the aggregated `heatmap`."""
    plot = cl.heatmap(adata, group_by=group_key, keys=_marker_groups(), aggregate=True)
    assert isinstance(plot, PlotSpec)


def test_heatmap_dict_keys_non_aggregate(adata, group_key):
    """Mapping `keys` render in the per-cell `heatmap`."""
    sub = adata[:300].copy()
    plot = cl.heatmap(sub, group_by=group_key, keys=_marker_groups(), aggregate=False)
    assert isinstance(plot, PlotSpec)


def test_matrixplot_dict_keys(adata, group_key):
    """Mapping `keys` render in `matrixplot`."""
    plot = cl.matrixplot(adata, group_by=group_key, keys=_marker_groups())
    assert isinstance(plot, PlotSpec)


def test_dotplot_dict_keys(adata, group_key):
    """Mapping `keys` render in `dotplot`."""
    plot = cl.dotplot(adata, group_by=group_key, keys=_marker_groups())
    assert isinstance(plot, PlotSpec)


def test_stacked_violin_dict_keys(adata, group_key):
    """Mapping `keys` render in `stacked_violin`."""
    plot = cl.stacked_violin(adata, group_by=group_key, keys=_marker_groups())
    assert isinstance(plot, PlotSpec)


def test_stacked_violin_options(adata, group_key):
    """`stacked_violin` accepts its styling options."""
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
    """`fill_statistic='variable'` renders without the outer rectangle."""
    sub = adata[:300].copy()
    plot = cl.stacked_violin(
        sub,
        group_by=group_key,
        keys=(markers := ["CD79A", "MS4A1"]),
        fill_statistic="variable",
        rectangle_kwargs={"linetype": "dashed"},
        n_points=16,
    )
    assert isinstance(plot, PlotSpec)
    assert markers == ["CD79A", "MS4A1"]


def test_stacked_violin_required_inputs_and_data_type(adata):
    """`stacked_violin` rejects missing inputs and unsupported data types."""
    with pytest.raises(ValueError, match="keys"):
        cl.stacked_violin(adata)
    with pytest.raises(UnsupportedDataTypeError):
        cl.stacked_violin("not adata", keys=["A"], group_by="group")


def test_compute_violin_polygons_edge_paths():
    """The violin polygon builder handles its edge cases."""
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
    single = _compute_violin_polygons(
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
    # a single observation has no density: a flat two-point line, not nothing
    assert single.height == 2
    assert single["y"].n_unique() == 1
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
    """A key in two mapping groups raises."""
    from cellestial.util.errors import DuplicateKeysError

    with pytest.raises(DuplicateKeysError, match="multiple groups"):
        cl.heatmap(
            adata,
            group_by=group_key,
            keys={"A": ["CD3D"], "B": ["CD3D"]},
        )


# ---- key groups internals ----


def test_resolve_key_groups_flat_passthrough():
    """Flat keys pass through without key groups."""
    from cellestial.single.heatmap.utilities import _resolve_key_groups

    flat, groups = _resolve_key_groups(["A", "B", "C"])
    assert flat == ["A", "B", "C"]
    assert groups is None


def test_resolve_key_groups_dict_flatten_preserves_order():
    """Mapping keys flatten in mapping order."""
    from cellestial.single.heatmap.utilities import _resolve_key_groups

    flat, groups = _resolve_key_groups({"G1": ["A", "B"], "G2": ["C"]})
    assert flat == ["A", "B", "C"]
    assert groups == {"G1": ["A", "B"], "G2": ["C"]}


def test_resolve_key_groups_dict_rejects_single_string_value():
    """A mapping value that is a bare string raises."""
    from cellestial.single.heatmap.utilities import _resolve_key_groups

    with pytest.raises(TypeError, match="got the string"):
        _resolve_key_groups({"G1": "A"})


def test_dotplot_dict_keys_keep_tight_y_limit_with_brackets(adata, group_key):
    """Mapping keys keep the `dotplot` y limit tight (brackets are an axis guide)."""
    groups = _marker_groups()
    plot = cl.dotplot(adata, group_by=group_key, keys=groups)
    n_y = adata.obs[group_key].nunique()
    spec = plot.as_dict()
    scales = [s for s in spec.get("scales", []) if s.get("aesthetic") == "y" and "limits" in s]
    assert scales, "dotplot should keep explicit y limits"
    # Brackets are an axis guide outside the panel, so the y limit stays tight.
    assert scales[-1]["limits"][1] == n_y - 0.5


def test_dotplot_dict_keys_without_key_labels_keeps_tight_y_limit(adata, group_key):
    """Without key labels the `dotplot` y limit stays tight."""
    groups = _marker_groups()
    plot = cl.dotplot(adata, group_by=group_key, keys=groups, key_labels=False)
    n_y = adata.obs[group_key].nunique()
    spec = plot.as_dict()
    scales = [s for s in spec.get("scales", []) if s.get("aesthetic") == "y" and "limits" in s]
    assert scales, "dotplot should keep explicit y limits"
    assert scales[-1]["limits"][1] == n_y - 0.5


# ---- dotplot ----


def test_dotplot_basic(adata, markers, group_key):
    """`dotplot` renders with its defaults."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, PlotSpec)


def test_dotplot_sort_by_mean_expression(adata, markers, group_key):
    """`dotplot` renders when sorting by average expression."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, sort_by="mean_expression")
    assert isinstance(plot, PlotSpec)


def test_dotplot_fill_mapping(adata, markers, group_key):
    """`dotplot` accepts a fill mapping."""
    from ggrust import aes

    plot = cl.dotplot(adata, keys=markers, group_by=group_key, mapping=aes(fill="mean_expression"))
    assert isinstance(plot, PlotSpec)


def test_dotplot_custom_colors(adata, markers, group_key):
    """`dotplot` accepts custom gradient colors."""
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        color_low="lightgrey",
        color_high="red",
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_threshold(adata, markers, group_key):
    """`dotplot` accepts an expression threshold."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, expression_threshold=0.2)
    assert isinstance(plot, PlotSpec)


def test_dotplot_invalid_gene(adata, group_key):
    """An unknown key raises `VariableNotFoundError`."""
    with pytest.raises(VariableNotFoundError):
        cl.dotplot(adata, keys=["NOT_A_GENE_xyz"], group_by=group_key)


def test_dotplot_dendrogram(adata, markers, group_key):
    """`dotplot` renders with a dendrogram."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, dendrogram=True)
    assert isinstance(plot, PlotSpec)
    # Render to SVG so lets-plot validates the computed x/y limits against all layers.
    assert plot.to_svg() is not None


def test_dotplot_dendrogram_without_rectangle(adata, markers, group_key):
    """`dotplot` renders a dendrogram without the outer rectangle."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, dendrogram=True, rectangle=False)
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


def test_dotplot_single_key(adata, group_key):
    """`dotplot` renders a single key."""
    # n_x == 1: x limits collapse to [-0.5, 0.5]; dendrogram extends rightward.
    plot = cl.dotplot(adata, keys=["CD3D"], group_by=group_key, dendrogram=True)
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


def test_dotplot_dendrogram_custom_style(adata, markers, group_key):
    """`dotplot` accepts dendrogram styling options."""
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
    """`dotplot` renders with neither dendrogram nor rectangle."""
    # Fallback path: x_max_limit should default to n_x - 0.5.
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, dendrogram=False, rectangle=False)
    assert isinstance(plot, PlotSpec)
    assert plot.to_svg() is not None


def test_dotplot_invalid_data_type(markers, group_key):
    """`dotplot` rejects an unsupported data type."""
    with pytest.raises(UnsupportedDataTypeError, match="Expected"):
        cl.dotplot({"not": "anndata"}, keys=markers, group_by=group_key)


def test_dotplot_tooltips_sequence(adata, markers, group_key):
    """`dotplot` accepts tooltips as a sequence of columns."""
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        tooltips=[group_key, "variable"],
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_tooltips_missing_column(adata, markers, group_key):
    """Tooltip columns absent from the frame raise."""
    with pytest.raises(KeyNotFoundError, match="not in data"):
        cl.dotplot(
            adata,
            keys=markers,
            group_by=group_key,
            tooltips=["nonexistent_column"],
        )


def test_dotplot_tooltips_feature_spec(adata, markers, group_key):
    """`dotplot` accepts tooltips built with `layer_tooltips`."""
    from ggrust import layer_tooltips

    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        tooltips=layer_tooltips([group_key, "variable"]),
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_tooltips_none_string(adata, markers, group_key):
    """`dotplot` accepts `tooltips='none'`."""
    plot = cl.dotplot(
        adata,
        keys=markers,
        group_by=group_key,
        tooltips="none",
    )
    assert isinstance(plot, PlotSpec)


def test_dotplot_interactive(adata, markers, group_key):
    """`dotplot(interactive=True)` adds the toolbar."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, interactive=True)
    assert isinstance(plot, PlotSpec)


# ---- highest_expressed_genes ----


def test_highest_expressed_genes_default(adata):
    """`highest_expressed_genes` renders with its defaults."""
    plot = cl.highest_expressed_genes(adata, n=10)
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_threshold(adata):
    """`highest_expressed_genes` accepts a threshold."""
    plot = cl.highest_expressed_genes(adata, n=10, threshold=0.1)
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_custom_outliers(adata):
    """`highest_expressed_genes` accepts custom outlier styling."""
    plot = cl.highest_expressed_genes(
        adata, n=5, outlier_size=0.1, outlier_alpha=0.2, outlier_shape=5
    )
    assert isinstance(plot, PlotSpec)


def test_highest_expressed_genes_n_larger_than_var(adata):
    """An `n` above the variable count is capped."""
    # n larger than var count should clip gracefully, not crash.
    with pytest.raises(ValueError):
        cl.highest_expressed_genes(adata, n=adata.n_vars + 100)


def test_heatmap_variable_axis_aggregate(adata):
    """`heatmap` aggregates along the variable axis."""
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
    assert y_scale["limits"] == pytest.approx([-0.5, local.n_vars - 0.5])


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
    assert y_scale["limits"] == pytest.approx([-0.5, local.n_vars - 0.5])

    group_lines = [
        layer["data"]
        for layer in spec["layers"]
        if layer.get("geom") == "segment"
        and hasattr(layer.get("data"), "columns")
        and "xend" in layer["data"].columns
    ]
    assert group_lines[-1]["y"].to_list() == pytest.approx([49.5])


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


# ---------------------------------------------------------------------------
# Fixed cells: every cell is `_CELL_SIZE` pixels and dots fit inside it
# ---------------------------------------------------------------------------

_CIRCLE_PATTERN = re.compile(r'<circle cx="([\d.]+)" cy="[\d.]+" r="([\d.]+)"')


def _size_scale_spec(plot):
    return next(scale for scale in plot.as_dict()["scales"] if scale.get("aesthetic") == "size")


@pytest.mark.parametrize("size_scale", [0.5, 1.0, 1.2])
def test_dotplot_size_range_fits_one_cell(adata, markers, group_key, size_scale):
    """Dot sizes are cell fractions; the legend draws the same sizes in pixels."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, size_scale=size_scale)
    point_layer = next(layer for layer in plot.as_dict()["layers"] if layer["geom"] == "point")
    assert point_layer["size_unit"] == "x"
    scale = _size_scale_spec(plot)
    assert scale["scale_mapper_kind"] == "size_area", "dot area is proportional to the share"
    size_max = scale["max_size"]
    assert size_max == pytest.approx(_DOT_SIZE_MAX * size_scale)
    assert size_max < 1, "a 100% dot must be narrower than one cell"
    assert scale["limits"] == [0, 100]
    assert not any(
        other.get("trans")
        for other in plot.as_dict()["scales"]
        if other.get("aesthetic") == "size"
    ), "a transform on top of the area scale would distort dot sizes"
    legend_sizes = scale["guide"]["override_aes"]["size"]
    assert len(legend_sizes) == len(scale["breaks"])
    # The 100% key is as wide in pixels as a 100% dot in a default cell.
    assert legend_sizes[-1] * _POINT_PIXELS_PER_SIZE == pytest.approx(size_max * _CELL_SIZE)
    # The 25% key is half as wide: diameter grows with the square root of the share.
    assert legend_sizes[0] * _POINT_PIXELS_PER_SIZE == pytest.approx(size_max * 0.5 * _CELL_SIZE)


def test_dotplot_zero_share_draws_no_dot():
    """Like scanpy, a group where no cell expresses the key gets an empty cell."""
    import re

    import anndata
    import numpy as np
    import pandas as pd

    X = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [3.0, 1.0]], dtype=np.float32)
    data = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "a", "b", "b"])}, index=[f"c{i}" for i in range(4)]
        ),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    svg = cl.dotplot(data, keys=["g0", "g1"], group_by="group").to_svg()
    radii = [float(r) for r in re.findall(r'<circle[^>]* r="([\d.]+)"', svg)]
    # Shares: g0 = 0% (a), 100% (b); g1 = 100% (a), 50% (b); plus 4 legend keys.
    assert len(radii) == 3 + 4, "the 0% cell must not draw a dot"
    data_radii = sorted(radii[:3])
    assert data_radii[0] / data_radii[-1] == pytest.approx(0.5**0.5, rel=0.01)


@pytest.mark.parametrize("canvas", [None, (640, 400), (400, 900), (300, 300)])
def test_dotplot_dots_stay_inside_cells(adata, markers, cluster_key, canvas):
    """Shrinking the canvas shrinks the dots with the cells (no overlap)."""
    plot = cl.dotplot(adata, keys=markers, group_by=cluster_key)
    if canvas is not None:
        plot += ggsize(*canvas)
    svg = plot.to_svg()
    panel = _PANEL_CLIP_PATTERN.search(svg)
    assert panel is not None
    panel_left = float(re.search(r'<clipPath[^>]*><rect x="([-\d.]+)"', svg).group(1))
    panel_width = float(panel.group(1))
    cell = panel_width / len(markers)
    diameters = [
        2 * float(radius)
        for center, radius in _CIRCLE_PATTERN.findall(svg)
        if panel_left <= float(center) <= panel_left + panel_width
    ]
    assert diameters
    assert max(diameters) <= cell


@pytest.mark.parametrize("function", ["dotplot", "stacked_violin", "matrixplot"])
@pytest.mark.parametrize("dendrogram", [False, True])
@pytest.mark.parametrize("group", ["group_key", "cluster_key"])
def test_fixed_cells_are_cell_size_wide(adata, markers, request, function, dendrogram, group):
    """Few groups, many groups and a dendrogram all keep `_CELL_SIZE` cells."""
    group_by = request.getfixturevalue(group)
    plot = getattr(cl, function)(adata, keys=markers, group_by=group_by, dendrogram=dendrogram)
    panel = _panel_size(plot)
    assert panel is not None
    x_span = len(markers) * (1 + 2 * _X_EXPAND if function == "matrixplot" else 1)
    assert panel[0] / x_span == pytest.approx(_CELL_SIZE, abs=0.1)
    y_span = adata.obs[group_by].nunique() * (1.5 if function == "stacked_violin" else 1)
    assert panel[1] / y_span == pytest.approx(_CELL_SIZE, abs=0.1)


def test_fixed_cells_single_key_keeps_legend_room(adata, group_key):
    """One key gets at least the backend's minimum panel instead of clipping the legend."""
    plot = cl.dotplot(adata, keys=["LYZ"], group_by=group_key)
    panel = _panel_size(plot)
    assert panel is not None
    assert panel[0] >= _PANEL_MIN_SIZE - 0.1
    # The size legend keeps all its keys instead of a "+K more" entry.
    assert "more" not in plot.to_svg()
    # A wider canvas widens the panel: nothing was squeezed at the chosen size.
    width = plot.as_dict()["ggsize"]["width"]
    wider = _panel_size(plot + ggsize(width + 10, plot.as_dict()["ggsize"]["height"]))
    assert wider is not None
    assert wider[0] == pytest.approx(panel[0] + 10, abs=0.1)


def test_fixed_cell_layout_falls_back_to_estimates(adata, markers, group_key, monkeypatch):
    """Without a measurable render the canvas comes from the label estimates."""
    monkeypatch.setattr(heatmap_utilities, "_panel_size", lambda plot: None)
    plot = cl.dotplot(adata, keys=markers, group_by=group_key)
    size = plot.as_dict()["ggsize"]
    assert size["width"] > len(markers) * _CELL_SIZE
    assert size["height"] >= 400


def test_dotplot_constant_size_stays_in_pixels(adata, markers, group_key):
    """A constant `size` is a point size, not a fraction of a cell."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, size=5)
    point_layer = next(layer for layer in plot.as_dict()["layers"] if layer["geom"] == "point")
    assert "size_unit" not in point_layer
    radii = {float(radius) for _, radius in _CIRCLE_PATTERN.findall(plot.to_svg())}
    assert max(radii) == pytest.approx(5 * _POINT_PIXELS_PER_SIZE / 2, abs=0.1)


def test_dotplot_other_size_mapping_stays_in_pixels(adata, markers, group_key):
    """Sizes mapped to another column keep pixel sizes and their plain legend."""
    plot = cl.dotplot(adata, keys=markers, group_by=group_key, mapping=aes(size="mean_expression"))
    point_layer = next(layer for layer in plot.as_dict()["layers"] if layer["geom"] == "point")
    assert "size_unit" not in point_layer
    scale = _size_scale_spec(plot)
    assert scale["range"][1] == pytest.approx(_DOT_SIZE_MAX * _CELL_SIZE / _POINT_PIXELS_PER_SIZE)
    assert "guide" not in scale


# ---------------------------------------------------------------------------
# _bin_within_groups: integer-keyed rewrite must match the string-keyed original
# ---------------------------------------------------------------------------


def _reference_bin_within_groups(
    frame: pl.DataFrame,
    *,
    observation_id_column: str,
    group_by: str,
    keys: list[str],
    y_order_groups: list[str],
    max_rows: int,
) -> pl.DataFrame:
    """
    String-keyed implementation of `_bin_within_groups` up to 0.62.1, kept as an oracle.

    Notes
    -----
    `frame` is wide, one row per observation and one column per key.
    Cells within a group are sorted by observation id and chunked into contiguous
    bins; each bin's expression value is the per-variable mean of its members.
    Group boundaries are preserved, so group bars/lines remain meaningful.
    Returns `frame` unchanged when the unique observation count fits in `max_rows`.
    Bins are matched on the observation id and the group (0.62.1 matched on the id
    alone, mixing groups that share an id).
    """
    group_key = "_group_key"
    frame_with_key = frame.with_columns(pl.col(group_by).cast(pl.String).alias(group_key))
    obs_per_group = (
        frame_with_key.select(observation_id_column, group_key)
        .unique()
        .group_by(group_key, maintain_order=False)
        .agg(pl.len().alias("_n_cells"))
    )
    total = int(obs_per_group["_n_cells"].sum())
    if total <= max_rows:
        return frame

    obs_per_group = obs_per_group.with_columns(
        pl.max_horizontal(
            pl.lit(1, dtype=pl.Int64),
            (pl.col("_n_cells") * max_rows / total).round().cast(pl.Int64),
        ).alias("_bin_count")
    )

    group_index_frame = pl.DataFrame(
        {
            group_key: y_order_groups,
            "_group_index": list(range(len(y_order_groups))),
        }
    )
    offsets_frame = (
        obs_per_group.join(group_index_frame, on=group_key)
        .sort("_group_index")
        .with_columns((pl.col("_bin_count").cum_sum() - pl.col("_bin_count")).alias("_bin_offset"))
        .select(group_key, "_bin_count", "_n_cells", "_bin_offset", "_group_index")
    )

    obs_frame = (
        frame_with_key.select(observation_id_column, group_key)
        .unique(maintain_order=True)
        .join(offsets_frame, on=group_key)
        .sort(["_group_index", observation_id_column])
        .with_columns(pl.int_range(pl.len()).over(group_key).alias("_rank_in_group"))
        .with_columns(
            (
                (pl.col("_rank_in_group") * pl.col("_bin_count") // pl.col("_n_cells"))
                + pl.col("_bin_offset")
            )
            .cast(pl.String)
            # bin names are zero-padded to the widest bin index (2026-11-18 follow-up S1)
            .str.zfill(len(str(int(offsets_frame["_bin_count"].sum()) - 1)))
            .alias("_binned_obs")
        )
        .select(observation_id_column, group_key, "_binned_obs")
    )

    # One observation id can appear in several groups, so match on the group too.
    return (
        frame_with_key.join(obs_frame, on=[observation_id_column, group_key])
        .drop(observation_id_column, group_key)
        .rename({"_binned_obs": observation_id_column})
        .group_by(observation_id_column, group_by, maintain_order=False)
        .agg(pl.col(keys).mean())
    )


def _assert_binning_matches_reference(frame: pl.DataFrame, **kwargs: object) -> None:
    expected = _reference_bin_within_groups(frame, **kwargs)
    result = heatmap_utilities._bin_within_groups(frame, **kwargs)
    sort_columns = [kwargs["observation_id_column"], kwargs["group_by"]]
    assert_frame_equal(
        result.sort(sort_columns, nulls_last=True), expected.sort(sort_columns, nulls_last=True)
    )


@pytest.mark.parametrize("max_rows", [1, 7, 100, 999])
def test_bin_within_groups_matches_reference_on_pbmc3k(adata, markers, cluster_key, max_rows):
    """Binning matches the reference implementation on pbmc3k."""
    frame = pl.DataFrame(
        {
            "observation_id": adata.obs_names.to_list(),
            cluster_key: adata.obs[cluster_key].to_numpy(),
        }
    ).with_columns(pl.col(cluster_key).cast(pl.Categorical))
    frame = frame.hstack(pl.from_pandas(adata[:, markers].to_df().reset_index(drop=True)))
    groups = sorted(adata.obs[cluster_key].unique().tolist(), reverse=True)
    _assert_binning_matches_reference(
        frame,
        observation_id_column="observation_id",
        group_by=cluster_key,
        keys=markers,
        y_order_groups=groups,
        max_rows=max_rows,
    )


def _small_binning_frame(groups: list[object], dtype: pl.DataType) -> pl.DataFrame:
    n = len(groups)
    return pl.DataFrame(
        {
            "observation_id": [f"cell_{(index * 7) % n:03d}" for index in range(n)],
            "group": pl.Series(groups, dtype=dtype),
            "gene_a": [float(index) for index in range(n)],
            "gene_b": [None if index % 5 == 0 else float(index * index) for index in range(n)],
        }
    )


@pytest.mark.parametrize(
    ("groups", "dtype", "y_order_groups"),
    [
        (["b", "a", "c"] * 20, pl.String, ["c", "a", "b"]),
        (["b", "a", "c"] * 20, pl.Categorical, ["a", "b", "c"]),
        ([3, 1, 2, 10] * 15, pl.Int64, ["10", "2", "1", "3"]),
        # A group absent from the order, plus null groups.
        (["b", "a", None, "absent"] * 15, pl.String, ["a", "b"]),
        (["b", "a", None, "absent"] * 15, pl.Categorical, ["a", None, "b"]),
    ],
)
@pytest.mark.parametrize("max_rows", [3, 10, 25])
def test_bin_within_groups_matches_reference(groups, dtype, y_order_groups, max_rows):
    """Binning matches the reference implementation across group dtypes."""
    _assert_binning_matches_reference(
        _small_binning_frame(groups, dtype),
        observation_id_column="observation_id",
        group_by="group",
        keys=["gene_a", "gene_b"],
        y_order_groups=y_order_groups,
        max_rows=max_rows,
    )


def test_bin_within_groups_matches_reference_with_null_and_duplicate_ids():
    """Binning matches the reference with null groups and repeated ids."""
    frame = _small_binning_frame(["a", "b"] * 20, pl.String)
    observation_ids = frame["observation_id"].to_list()
    observation_ids[3] = None
    observation_ids[8] = None
    observation_ids[5] = observation_ids[4]  # same id in two groups
    observation_ids[6] = observation_ids[4]  # exact duplicate (same group)
    frame = frame.with_columns(pl.Series("observation_id", observation_ids))
    _assert_binning_matches_reference(
        frame,
        observation_id_column="observation_id",
        group_by="group",
        keys=["gene_a", "gene_b"],
        y_order_groups=["b", "a"],
        max_rows=9,
    )


def test_bin_within_groups_returns_frame_when_it_fits():
    """A frame that fits in `max_rows` is returned unchanged."""
    frame = _small_binning_frame(["a", "b"] * 10, pl.String)
    result = heatmap_utilities._bin_within_groups(
        frame,
        observation_id_column="observation_id",
        group_by="group",
        keys=["gene_a", "gene_b"],
        y_order_groups=["a", "b"],
        max_rows=20,
    )
    assert result is frame
