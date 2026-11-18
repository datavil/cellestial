"""Follow-up fixes to the 2026-11-18 audit of `cellestial/frames`, `cellestial/util` and datasets."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import cellestial as cl
from cellestial.frames._container import _container
from cellestial.frames.operations import _pca_variance_frame
from cellestial.util.dendrogram import _dendrogram_info
from cellestial.util.errors import (
    ConflictingLengthError,
    GridIndexError,
    InvalidArgumentError,
    InvalidArgumentTypeError,
    InvalidPlotSpecError,
    KeyNotFoundError,
    UnsupportedDataTypeError,
    VariableNotFoundError,
    _did_you_mean,
)
from cellestial.util.utilities import _determine_axis, _require_feature_key, _validate_tooltips

# --- errors ------------------------------------------------------------------


@pytest.mark.parametrize("error_class", [KeyNotFoundError, VariableNotFoundError])
def test_key_errors_subclass_key_error_and_print_unquoted(error_class):
    """Key errors are `KeyError`s whose message prints without quotes."""
    error = error_class("key 'x' not found")
    assert isinstance(error, KeyError)
    assert str(error) == "key 'x' not found"


def test_did_you_mean_names_close_matches():
    """The hint names a close match and is empty when nothing is close."""
    assert _did_you_mean("leidn", ["leiden", "louvain"]) == "Did you mean 'leiden'?"
    assert _did_you_mean("zzz", ["leiden"]) == ""


def test_require_feature_key_suggests_close_match(adata):
    """An unknown color key names the closest metadata column."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        _require_feature_key(adata, "leidn")


def test_unknown_variable_suggests_close_match(adata):
    """An unknown variable names the closest variable."""
    with pytest.raises(VariableNotFoundError, match="CD14"):
        cl.build_frame(adata, axis=0, variable_keys=["CD1A4"])


def test_unknown_metadata_column_is_key_not_found_with_hint(adata):
    """An unknown metadata column raises `KeyNotFoundError` with a hint."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        cl.build_frame(adata, axis=0, metadata_columns=["leidn"])


def test_unknown_modality_suggests_close_match(mudata):
    """An unknown modality names the closest modality."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'rna'"):
        _container(mudata).select_modality("rnaa")


# --- bare builtin raises become errors.py classes ------------------------------


def _small_plot():
    from ggrust import aes, geom_point, ggplot

    return ggplot({"x": [1], "y": [1]}, aes("x", "y")) + geom_point()


def test_layout_accepts_numpy_widths():
    """A flat or nested numpy `widths` works like the matching list."""
    plot = _small_plot()
    flat = cl.layout([[plot, plot]], widths=np.array([1, 2]))
    nested = cl.layout([[plot, plot], [plot]], widths=[np.array([1, 2]), np.array([1])])
    matrix = cl.layout([[plot, plot], [plot, plot]], widths=np.array([[1, 2], [2, 1]]))
    assert flat.as_dict() == cl.layout([[plot, plot]], widths=[1, 2]).as_dict()
    assert nested.as_dict() == cl.layout([[plot, plot], [plot]], widths=[[1, 2], [1]]).as_dict()
    assert matrix.as_dict() == cl.layout([[plot, plot]] * 2, widths=[[1, 2], [2, 1]]).as_dict()


@pytest.mark.parametrize(
    ("call", "error_class", "message"),
    [
        (lambda plot: cl.layout(plot), InvalidArgumentTypeError, "plots"),
        (lambda plot: cl.layout([]), InvalidArgumentError, "at least one row"),
        (lambda plot: cl.layout([[]]), InvalidArgumentError, "row 0 is empty"),
        (lambda plot: cl.layout([[plot]], heights=[1, 2]), ConflictingLengthError, "heights"),
        (lambda plot: cl.layout([[plot]], widths="1"), InvalidArgumentTypeError, "widths"),
        (lambda plot: cl.layout([[plot]], widths=[1, [1]]), InvalidArgumentError, "mixes"),
        (lambda plot: cl.layout([[plot]], widths=[1, 2]), ConflictingLengthError, "broadcast"),
        (lambda plot: cl.layout([[plot]], widths=[[1], [1]]), ConflictingLengthError, "rows"),
        (lambda plot: cl.layout([[plot]], widths=[[1, 2]]), ConflictingLengthError, "widths"),
    ],
)
def test_layout_raises_specific_errors(call, error_class, message):
    """Invalid `layout` arguments raise the matching `errors.py` class."""
    with pytest.raises(error_class, match=message):
        call(_small_plot())


def test_grid_lookups_raise_specific_errors():
    """`get_figure` and `get_figures` raise specific classes for bad input."""
    grid = cl.layout([[_small_plot(), _small_plot()]])
    with pytest.raises(InvalidPlotSpecError):
        cl.get_figure(_small_plot(), 0)
    with pytest.raises(InvalidArgumentTypeError, match="index"):
        cl.get_figure(grid, "0")
    with pytest.raises(GridIndexError, match="out of range"):
        cl.get_figure(grid, 5)
    with pytest.raises(GridIndexError, match="out of range"):
        cl.get_figures(grid, [0, 5])
    with pytest.raises(InvalidPlotSpecError):
        cl.retrieve("not a plot")


@pytest.mark.parametrize(
    ("kwargs", "error_class"),
    [
        ({"axis": 0, "include_dimensions": -1}, InvalidArgumentError),
        ({"axis": 0, "include_dimensions": 1.5}, InvalidArgumentTypeError),
        ({}, InvalidArgumentError),
    ],
)
def test_build_frame_invalid_arguments(adata, kwargs, error_class):
    """Invalid `build_frame` arguments raise the matching `errors.py` class."""
    with pytest.raises(error_class):
        cl.build_frame(adata, **kwargs)


def test_determine_axis_unknown_key_suggests_close_match(adata):
    """A key on neither axis raises `KeyNotFoundError` with a hint."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        _determine_axis(adata, ["leidn"])


def test_unknown_tooltip_field_raises_key_not_found(adata):
    """A tooltip naming no column raises `KeyNotFoundError`."""
    frame = pl.DataFrame({"leiden": ["0"]})
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        _validate_tooltips(["leidn"], frame)


def test_missing_pca_variance_raises_key_not_found(adata):
    """Plotting variance without a stored PCA names the missing key."""
    with pytest.raises(KeyNotFoundError, match="PCA variance ratio"):
        _pca_variance_frame(adata, use_key="missing_pca")


def test_too_many_marker_genes_is_invalid_argument(mudata):
    """Asking for more marker genes than ranked raises `InvalidArgumentError`."""
    with pytest.raises(InvalidArgumentError, match="n_genes"):
        cl.marker_genes(mudata, modality="rna", n_genes=10**6)


def test_dendrogram_rejects_unsupported_data():
    """The dendrogram helper rejects a non-AnnData object with the shared class."""
    with pytest.raises(UnsupportedDataTypeError):
        _dendrogram_info("not data", "leiden")


def test_from_cellxgene_rejects_source_without_uuid():
    """A source holding no dataset UUID raises `InvalidArgumentError`."""
    with pytest.raises(InvalidArgumentError, match="UUID"):
        cl.datasets.from_cellxgene("not-a-uuid")
