"""Follow-up fixes to the 2026-11-18 audit of `cellestial/frames`, `cellestial/util` and datasets."""

from __future__ import annotations

import pytest

import cellestial as cl
from cellestial.frames._container import _container
from cellestial.util.errors import KeyNotFoundError, VariableNotFoundError, _did_you_mean
from cellestial.util.utilities import _require_feature_key

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
