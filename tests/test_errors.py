"""Error-path tests for public misuse: singular/plural, missing keys, wrong types."""

import numpy as np
import pytest

import cellestial as cl
from cellestial.util.errors import (
    EmptySelectionError,
    InvalidPlotSpecError,
    KeyNotFoundError,
    MissingAestheticError,
    NonCategoricalColumnError,
    UnsupportedDataTypeError,
)
from tests.conftest import GROUP_KEY

# ---- Category 1: singular functions guide users to the plural ----


@pytest.mark.parametrize(
    ("fn", "plural"),
    [
        (cl.umap, "umaps"),
        (cl.tsne, "tsnes"),
        (cl.pca, "pcas"),
        (cl.dimensional, "dimensionals"),
        (cl.expression, "expressions"),
    ],
)
def test_singular_sequence_key_points_to_plural(adata, fn, plural):
    with pytest.raises(TypeError, match=plural):
        fn(adata, ["CD14", "MS4A1"])


def test_ridge_sequence_key_points_to_plural(adata, group_key):
    with pytest.raises(TypeError, match="ridges"):
        cl.ridge(adata, ["CD14", "MS4A1"], group_key)


# ---- Category 2: missing key says where Cellestial looked ----


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne, cl.dimensional])
def test_missing_key_reports_where_it_looked(adata, fn):
    with pytest.raises(KeyNotFoundError, match="observation metadata and variable"):
        fn(adata, "NOT_A_REAL_KEY")


# ---- Category 3: wrong data type says expected and received ----


@pytest.mark.parametrize("fn", [cl.umap, cl.dimensional, cl.heatmap, cl.elbow])
def test_wrong_data_type_reports_expected_and_received(fn):
    with pytest.raises(UnsupportedDataTypeError, match="Expected AnnData.*received `str`"):
        fn("not anndata")


def test_volcano_wrong_data_type():
    with pytest.raises(UnsupportedDataTypeError, match="Expected AnnData.*received `str`"):
        cl.volcano("not anndata", "B Cells")


def test_build_frame_wrong_type_mentions_spatialdata():
    with pytest.raises(UnsupportedDataTypeError, match="AnnData or SpatialData"):
        cl.build_frame("not anndata", axis=0)


# ---- Category 4: unknown `groups` / `drop` names name what is available ----


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (cl.umap, {"key": GROUP_KEY}),
        (cl.pca, {"key": GROUP_KEY}),
        (cl.tsne, {"key": GROUP_KEY}),
        (cl.dimensional, {"key": GROUP_KEY}),
        (cl.violin, {"key": "CD3D", "group_by": GROUP_KEY}),
        (cl.boxplot, {"key": "CD3D", "group_by": GROUP_KEY}),
        (cl.histogram, {"key": "CD3D", "group_by": GROUP_KEY}),
        (cl.ridge, {"key": "CD3D", "group_by": GROUP_KEY}),
    ],
)
def test_unknown_group_reports_available(adata, fn, kwargs):
    with pytest.raises(KeyNotFoundError, match=r"`groups` \['Nope'\].*Available:"):
        fn(adata, groups=["Nope"], **kwargs)


def test_unknown_group_caught_alongside_a_valid_one(adata, group_key):
    """A partial typo must not quietly produce a plot missing that condition."""
    with pytest.raises(KeyNotFoundError, match=r"\['NoSuchGroup'\]"):
        cl.umap(adata, key=group_key, groups=["Lymphocytes", "NoSuchGroup"])


def test_unknown_drop_reports_available(adata, group_key):
    with pytest.raises(KeyNotFoundError, match=r"`drop` \['Lymphcytes'\].*Available:"):
        cl.violin(adata, key="CD3D", group_by=group_key, drop=["Lymphcytes"])


def test_group_removed_by_nonfinite_coordinates_is_not_unknown(adata, group_key):
    """A group whose cells all lack coordinates still exists, so naming it must not raise."""
    missing = sorted(adata.obs[group_key].dropna().unique())[0]
    adata = adata.copy()
    umap = adata.obsm["X_umap"].copy()
    umap[(adata.obs[group_key] == missing).to_numpy()] = np.nan
    adata.obsm["X_umap"] = umap
    plot = cl.umap(adata, key=group_key, groups=[missing])
    assert cl.retrieve(plot).height == 0


def test_highlight_unknown_group_reports_available(adata, cluster_key):
    with pytest.raises(
        KeyNotFoundError, match=r"`groups` \['leidn'\] not found in color `leiden`.*Available:"
    ):
        cl.umap(adata, key=cluster_key) + cl.highlight(["0", "leidn"])


def test_highlight_without_color_key_says_how_to_fix(adata):
    with pytest.raises(MissingAestheticError, match=r"no color mapping.*key='leiden'"):
        cl.umap(adata) + cl.highlight("0")


def test_highlight_continuous_color_says_how_to_fix(adata):
    with pytest.raises(
        NonCategoricalColumnError, match=r"`CD14` holds numeric values.*categorical key"
    ):
        cl.umap(adata, key="CD14") + cl.highlight("0")


def test_highlight_empty_groups_fails_early():
    with pytest.raises(EmptySelectionError, match=r"`groups` is empty"):
        cl.highlight([])


def test_highlight_on_a_grid_points_to_singular(adata, cluster_key):
    with pytest.raises(InvalidPlotSpecError, match=r"adds to a single plot.*SupPlotsSpec"):
        cl.umaps(adata, [cluster_key]) + cl.highlight("0")
