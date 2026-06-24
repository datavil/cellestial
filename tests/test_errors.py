"""Error-path tests for public misuse: singular/plural, missing keys, wrong types."""

import pytest

import cellestial as cl
from cellestial.util.errors import KeyNotFoundError, UnsupportedDataTypeError

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
