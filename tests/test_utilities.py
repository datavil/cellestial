"""Tests for cellestial.util.utilities, internal helpers."""

import warnings

import numpy as np
import polars as pl
import pytest
from lets_plot.plot.core import FeatureSpec

from cellestial.util.errors import (
    CellestialWarning,
    KeyNotFoundError,
    UnsupportedDataTypeError,
)
from cellestial.util.operations import retrieve
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
    _resolve_embedding_key,
    _resolve_midpoint,
    _select_variable_keys,
    _share_axis,
    _share_labels,
    _share_ticks,
    _wrap_legend,
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


@pytest.mark.parametrize("mid_point", ["mean", "median", "mid"])
def test_resolve_midpoint_ignores_nonfinite_values(mid_point):
    """Calculated midpoints should use only finite values."""
    series = pl.Series("x", [None, np.nan, -np.inf, -2.0, 0.0, 2.0, np.inf])

    assert _resolve_midpoint(series, mid_point) == pytest.approx(0.0)


def test_resolve_midpoint_accepts_negative_value_within_range():
    """Negative numeric midpoints remain valid when they are within range."""
    series = pl.Series("x", [-3.0, -1.0, 2.0])

    assert _resolve_midpoint(series, -1.5) == -1.5


@pytest.mark.parametrize("mid_point", [-3.0, 3.0])
def test_resolve_midpoint_rejects_value_outside_range(mid_point):
    """Numeric midpoints must fall within the finite data range."""
    series = pl.Series("x", [-2.0, 0.0, 2.0])

    with pytest.raises(ValueError, match="within the finite data range"):
        _resolve_midpoint(series, mid_point)


@pytest.mark.parametrize("mid_point", [np.nan, np.inf, -np.inf])
def test_resolve_midpoint_rejects_nonfinite_value(mid_point):
    """Numeric midpoints must themselves be finite."""
    series = pl.Series("x", [-2.0, 0.0, 2.0])

    with pytest.raises(ValueError, match="must be finite"):
        _resolve_midpoint(series, mid_point)


def test_resolve_midpoint_rejects_boolean():
    """Booleans should not be interpreted as numeric midpoints."""
    series = pl.Series("x", [0.0, 1.0])

    with pytest.raises(TypeError, match="mid_point"):
        _resolve_midpoint(series, True)


def test_resolve_midpoint_rejects_series_without_finite_values():
    """Calculated midpoint modes require at least one finite value."""
    series = pl.Series("x", [None, np.nan, np.inf, -np.inf])

    with pytest.raises(ValueError, match="without finite data values"):
        _resolve_midpoint(series, "median")


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


# ---- _resolve_embedding_key ----


def test_resolve_embedding_use_key_uppercases(adata):
    assert _resolve_embedding_key(adata, "umap", "X_umap2d", xy=(1, 2)) == "X_UMAP2D"


def test_resolve_embedding_use_key_already_upper(adata):
    assert _resolve_embedding_key(adata, "umap", "X_UMAP", xy=(1, 2)) == "X_UMAP"


def test_resolve_embedding_use_key_no_warning(adata):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _resolve_embedding_key(adata, "umap", "anything", xy=(1, 2))
    cellestial_warnings = [w for w in caught if issubclass(w.category, CellestialWarning)]
    assert cellestial_warnings == []


def test_resolve_embedding_exact_match_silent(adata):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_embedding_key(adata, "umap", None, xy=(1, 2))
    assert result == "X_UMAP"
    cellestial_warnings = [w for w in caught if issubclass(w.category, CellestialWarning)]
    assert cellestial_warnings == []


@pytest.mark.parametrize("dimensions", ["umap", "pca", "tsne"])
def test_resolve_embedding_exact_match_all_families(adata, dimensions):
    result = _resolve_embedding_key(adata, dimensions, None, xy=(1, 2))
    assert result == f"X_{dimensions.upper()}"


def test_resolve_embedding_fallback_single_candidate_warns(adata):
    local = adata.copy()
    local.obsm["X_umap2d"] = local.obsm.pop("X_umap")
    with pytest.warns(CellestialWarning, match=r"X_umap2d"):
        result = _resolve_embedding_key(local, "umap", None, xy=(1, 2))
    assert result == "X_UMAP2D"


def test_resolve_embedding_fallback_shape_aware(adata):
    local = adata.copy()
    n_obs = local.n_obs
    local.obsm.pop("X_umap")
    local.obsm["X_umap2d"] = np.zeros((n_obs, 2))
    local.obsm["X_umap3d"] = np.zeros((n_obs, 3))
    # 3rd axis requested -> 2D candidate is insufficient, pick 3D
    with pytest.warns(CellestialWarning):
        result = _resolve_embedding_key(local, "umap", None, xy=(1, 3))
    assert result == "X_UMAP3D"


def test_resolve_embedding_fallback_tie_break_shortest(adata):
    local = adata.copy()
    n_obs = local.n_obs
    local.obsm.pop("X_umap")
    local.obsm["X_umap_long_name"] = np.zeros((n_obs, 2))
    local.obsm["X_umap_a"] = np.zeros((n_obs, 2))
    with pytest.warns(CellestialWarning):
        result = _resolve_embedding_key(local, "umap", None, xy=(1, 2))
    assert result == "X_UMAP_A"


def test_resolve_embedding_no_candidate_raises(adata):
    local = adata.copy()
    for key in list(local.obsm.keys()):
        if key.upper().startswith("X_UMAP"):
            local.obsm.pop(key)
    with pytest.raises(KeyNotFoundError, match="X_UMAP"):
        _resolve_embedding_key(local, "umap", None, xy=(1, 2))


def test_resolve_embedding_unsupported_type():
    with pytest.raises(UnsupportedDataTypeError):
        _resolve_embedding_key("not anndata", "umap", None, xy=(1, 2))


def test_resolve_embedding_bare_form_exact_silent(adata):
    """Only `umap` in obsm (no scanpy `X_` prefix) -> silent, returns `UMAP`."""
    local = adata.copy()
    local.obsm["umap"] = local.obsm.pop("X_umap")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_embedding_key(local, "umap", None, xy=(1, 2))
    assert result == "UMAP"
    cellestial_warnings = [w for w in caught if issubclass(w.category, CellestialWarning)]
    assert cellestial_warnings == []


def test_resolve_embedding_bare_form_prefix_warns(adata):
    """Only `UMAP_2D` in obsm -> matches bare prefix, warns and picks it."""
    local = adata.copy()
    n_obs = local.n_obs
    local.obsm.pop("X_umap")
    local.obsm["UMAP_2D"] = np.zeros((n_obs, 2))
    with pytest.warns(CellestialWarning, match="UMAP_2D"):
        result = _resolve_embedding_key(local, "umap", None, xy=(1, 2))
    assert result == "UMAP_2D"


def test_resolve_embedding_canonical_wins_over_bare(adata):
    """When both `X_UMAP` and `UMAP` exist, canonical `X_UMAP` is preferred."""
    local = adata.copy()
    n_obs = local.n_obs
    local.obsm["UMAP"] = np.zeros((n_obs, 2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = _resolve_embedding_key(local, "umap", None, xy=(1, 2))
    assert result == "X_UMAP"
    cellestial_warnings = [w for w in caught if issubclass(w.category, CellestialWarning)]
    assert cellestial_warnings == []


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
