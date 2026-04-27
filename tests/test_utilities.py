"""Tests for cellestial.util.utilities, internal helpers."""

import polars as pl
import pytest
from lets_plot.plot.core import FeatureSpec

from cellestial.util.utilities import (
    _are_observation_features,
    _are_observations,
    _are_variable_features,
    _are_variables,
    _build_tooltips,
    _color_gradient,
    _determine_axis,
    _fill_gradient,
    _is_observation_feature,
    _is_observation_key,
    _is_variable_feature,
    _is_variable_key,
    _range_inclusive,
    _select_variable_keys,
    _share_axis,
    _share_labels,
    _share_ticks,
    _wrap_legend,
    retrieve,
)

# ---- _build_tooltips ----


def test_build_tooltips_basic():
    result = _build_tooltips(
        tooltips=["col_a", "col_b"],
        cluster_name="Cluster",
        key=None,
    )
    assert isinstance(result, FeatureSpec)


def test_build_tooltips_none_string():
    result = _build_tooltips(tooltips="none", cluster_name="Cluster")
    assert result == "none"


def test_build_tooltips_clustering_mode():
    result = _build_tooltips(
        tooltips=["leiden", "n_genes"],
        cluster_name="Cluster",
        key="leiden",
        clustering=True,
    )
    assert isinstance(result, FeatureSpec)


def test_build_tooltips_with_title():
    result = _build_tooltips(
        tooltips=["col_a"],
        cluster_name="Cluster",
        title="My Title",
    )
    assert isinstance(result, FeatureSpec)


# ---- _range_inclusive ----


def test_range_inclusive_normal():
    result = _range_inclusive(0, 10, 5)
    assert len(result) >= 2
    assert result[0] == 0


def test_range_inclusive_small_range():
    result = _range_inclusive(0.1, 0.3, 3)
    assert len(result) >= 2


def test_range_inclusive_zero_range():
    result = _range_inclusive(5.0, 5.0, 3)
    assert result == [5.0]


# ---- _color_gradient ----


def test_color_gradient_no_mid():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _color_gradient(series, color_low="white", color_high="red")
    assert isinstance(result, FeatureSpec)


def test_color_gradient_with_mid_mean():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _color_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point="mean"
    )
    assert isinstance(result, FeatureSpec)


def test_color_gradient_with_mid_median():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _color_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point="median"
    )
    assert isinstance(result, FeatureSpec)


def test_color_gradient_with_mid_point_value():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _color_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point="mid"
    )
    assert isinstance(result, FeatureSpec)


def test_color_gradient_with_mid_float():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _color_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point=1.5
    )
    assert isinstance(result, FeatureSpec)


# ---- _fill_gradient ----


def test_fill_gradient_no_mid():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _fill_gradient(series, color_low="white", color_high="red")
    assert isinstance(result, FeatureSpec)


def test_fill_gradient_with_mid_mean():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _fill_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point="mean"
    )
    assert isinstance(result, FeatureSpec)


def test_fill_gradient_with_mid_median():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _fill_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point="median"
    )
    assert isinstance(result, FeatureSpec)


def test_fill_gradient_with_mid_mid():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _fill_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point="mid"
    )
    assert isinstance(result, FeatureSpec)


def test_fill_gradient_with_mid_float():
    series = pl.Series("x", [1.0, 2.0, 3.0])
    result = _fill_gradient(
        series, color_low="white", color_mid="yellow", color_high="red", mid_point=2.0
    )
    assert isinstance(result, FeatureSpec)


# ---- _wrap_legend ----


def test_wrap_legend_small():
    frame = pl.DataFrame({"color_col": ["a", "b", "c"]})
    result = _wrap_legend(frame, fill=None, color="color_col")
    assert isinstance(result, FeatureSpec)


def test_wrap_legend_large_fill():
    frame = pl.DataFrame({"fill_col": [str(i) for i in range(15)]})
    result = _wrap_legend(frame, fill="fill_col", color=None)
    assert isinstance(result, FeatureSpec)


def test_wrap_legend_large_color():
    frame = pl.DataFrame({"color_col": [str(i) for i in range(15)]})
    result = _wrap_legend(frame, fill=None, color="color_col")
    assert isinstance(result, FeatureSpec)


# ---- _is_variable_key / _are_variables / _is_observation_key / _are_observations ----


def test_is_variable_key_true(adata):
    assert _is_variable_key(adata, "CD3D") is True


def test_is_variable_key_false(adata):
    assert _is_variable_key(adata, "leiden") is False


def test_is_variable_key_none(adata):
    assert _is_variable_key(adata, None) is False


def test_are_variables_true(adata):
    assert _are_variables(adata, ["CD3D", "LYZ"]) is True


def test_are_variables_false(adata):
    assert _are_variables(adata, ["CD3D", "leiden"]) is False


def test_are_variables_none(adata):
    assert _are_variables(adata, None) is False


def test_is_observation_key_true(adata):
    assert _is_observation_key(adata, "leiden") is True


def test_is_observation_key_false(adata):
    assert _is_observation_key(adata, "CD3D") is False


def test_is_observation_key_none(adata):
    assert _is_observation_key(adata, None) is False


def test_are_observations_true(adata):
    assert _are_observations(adata, ["leiden"]) is True


def test_are_observations_false(adata):
    assert _are_observations(adata, ["CD3D"]) is False


def test_are_observations_none(adata):
    assert _are_observations(adata, None) is False


# ---- _is_observation_feature / _are_observation_features ----


def test_is_observation_feature_obs_col(adata):
    assert _is_observation_feature(adata, "leiden") is True


def test_is_observation_feature_gene(adata):
    assert _is_observation_feature(adata, "CD3D") is True


def test_is_observation_feature_false(adata):
    assert _is_observation_feature(adata, "NOT_REAL") is False


def test_is_observation_feature_none(adata):
    assert _is_observation_feature(adata, None) is False


def test_are_observation_features_true(adata):
    assert _are_observation_features(adata, ["leiden", "CD3D"]) is True


def test_are_observation_features_false(adata):
    assert _are_observation_features(adata, ["NOT_REAL"]) is False


def test_are_observation_features_none(adata):
    assert _are_observation_features(adata, None) is False


# ---- _is_variable_feature / _are_variable_features ----


def test_is_variable_feature_true(adata):
    # var columns typically include things like 'n_cells_by_counts'
    var_cols = list(adata.var.columns)
    if var_cols:
        assert _is_variable_feature(adata, var_cols[0]) is True


def test_is_variable_feature_false(adata):
    assert _is_variable_feature(adata, "NOT_REAL") is False


def test_is_variable_feature_none(adata):
    assert _is_variable_feature(adata, None) is False


def test_are_variable_features_true(adata):
    var_cols = list(adata.var.columns)
    if len(var_cols) >= 2:
        assert _are_variable_features(adata, var_cols[:2]) is True


def test_are_variable_features_false(adata):
    assert _are_variable_features(adata, ["NOT_REAL"]) is False


def test_are_variable_features_none(adata):
    assert _are_variable_features(adata, None) is False


# ---- _select_variable_keys ----


def test_select_variable_keys_mixed(adata):
    result = _select_variable_keys(adata, ["CD3D", "leiden", "LYZ"])
    assert "CD3D" in result
    assert "LYZ" in result
    assert "leiden" not in result


def test_select_variable_keys_none(adata):
    assert _select_variable_keys(adata, None) == []


# ---- _determine_axis ----


def test_determine_axis_obs(adata):
    assert _determine_axis(adata, ["leiden"]) == 0


def test_determine_axis_var(adata):
    var_cols = list(adata.var.columns)
    if var_cols:
        assert _determine_axis(adata, var_cols[:1]) == 1


def test_determine_axis_string_key(adata):
    assert _determine_axis(adata, "CD3D") == 0


def test_determine_axis_invalid(adata):
    with pytest.raises(ValueError):
        _determine_axis(adata, ["NOT_REAL_KEY_xyz"])


# ---- get_mapping / retrieve error paths ----


def test_retrieve_invalid_type():
    with pytest.raises(TypeError):
        retrieve("not a plot")


# ---- _share_labels / _share_axis / _share_ticks ----


def test_share_labels(adata):
    import cellestial as cl

    plot = cl.umap(adata, "leiden")
    # i=0 is top-left, should keep y label, remove x label
    result = _share_labels(plot, 0, ["a", "b", "c", "d"], ncol=2)
    assert result is not None


def test_share_axis(adata):
    import cellestial as cl

    plot = cl.umap(adata, "leiden")
    result = _share_axis(plot, 1, ["a", "b", "c", "d"], ncol=2, axis_type="axis")
    assert result is not None


def test_share_axis_arrow(adata):
    import cellestial as cl

    plot = cl.umap(adata, "leiden")
    result = _share_axis(plot, 0, ["a", "b"], ncol=2, axis_type="arrow")
    assert result is not None


def test_share_axis_invalid_type(adata):
    import cellestial as cl

    plot = cl.umap(adata, "leiden")
    with pytest.raises(ValueError):
        _share_axis(plot, 0, ["a", "b"], ncol=2, axis_type="invalid")


def test_share_ticks(adata):
    import cellestial as cl

    plot = cl.umap(adata, "leiden")
    result = _share_ticks(plot, 0, ["a", "b", "c", "d"], ncol=2)
    assert result is not None
