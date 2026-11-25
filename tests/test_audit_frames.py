"""Regression tests from the 2026-11-18 audit of `cellestial/frames`, `cellestial/util` and datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from mudata import MuData
from scipy import sparse

import cellestial as cl
from cellestial.datasets import datasets
from cellestial.util.errors import DuplicateKeysError, InvalidArgumentError
from cellestial.util.utilities import _container_column, _modality_source


@pytest.fixture
def numeric_categories() -> AnnData:
    """Four cells and three variables with numeric categorical metadata holding a missing value."""
    data = AnnData(
        X=np.arange(12, dtype=np.float32).reshape(4, 3),
        obs=pd.DataFrame(index=["a", "b", "c", "d"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    data.obs["cluster"] = pd.Categorical([1, 2, None, 1])
    data.var["bin"] = pd.Categorical([0.5, None, 1.5])
    return data


@pytest.fixture
def single_modality(mudata) -> MuData:
    """A container with only the `rna` modality, so `modality=None` auto-selects it."""
    data = MuData({"rna": mudata["rna"].copy()})
    data.update()
    return data


# --- build_frame -------------------------------------------------------------


@pytest.mark.parametrize("axis", [2, -1, "0"])
def test_build_frame_rejects_invalid_axis(adata, axis):
    """An `axis` other than 0 or 1 raises `InvalidArgumentError`."""
    with pytest.raises(InvalidArgumentError, match="axis"):
        cl.build_frame(adata, axis=axis)


def test_build_frame_rejects_repeated_variable_keys(adata):
    """Repeated `variable_keys` raise `DuplicateKeysError` naming the repeat."""
    with pytest.raises(DuplicateKeysError, match=r"variable_keys.*CD14"):
        cl.build_frame(adata, variable_keys=["CD14", "MS4A1", "CD14"])


def test_build_frame_rejects_repeated_metadata_columns(adata):
    """Repeated `metadata_columns` raise `DuplicateKeysError` naming the repeat."""
    with pytest.raises(DuplicateKeysError, match=r"metadata_columns.*leiden"):
        cl.build_frame(adata, axis=0, metadata_columns=["leiden", "leiden"])
    with pytest.raises(DuplicateKeysError, match="metadata_columns"):
        cl.build_frame(adata, axis=1, metadata_columns=["n_cells", "n_cells"])


def test_build_frame_rejects_repeated_dimension_keys(adata):
    """Case variants of one embedding in `dimension_keys` raise `DuplicateKeysError`."""
    with pytest.raises(DuplicateKeysError, match="dimension_keys"):
        cl.build_frame(
            adata,
            axis=0,
            include_dimensions=2,
            metadata_columns=[],
            dimension_keys=["X_umap", "x_UMAP"],
        )


def test_numeric_categories_keep_missing_values_null(numeric_categories):
    """Missing values of a numeric categorical stay null, not a `"nan"` category."""
    observations = cl.build_frame(numeric_categories, axis=0)
    assert observations["cluster"].to_list() == ["1", "2", None, "1"]
    assert observations["cluster"].dtype == pl.Enum(["1", "2"])

    variables = cl.build_frame(numeric_categories, axis=1)
    assert variables["bin"].to_list() == ["0.5", None, "1.5"]
    assert variables["bin"].dtype == pl.Enum(["0.5", "1.5"])


def test_sparse_embedding_becomes_columns(numeric_categories):
    """A sparse embedding becomes dense dimension columns."""
    data = numeric_categories.copy()
    dense = np.arange(8, dtype=np.float32).reshape(4, 2)
    data.obsm["X_sparse"] = sparse.csr_matrix(dense)
    data.varm["V_sparse"] = sparse.csr_matrix(np.ones((3, 2)))

    observations = cl.build_frame(data, axis=0, include_dimensions=True)
    assert observations["X_SPARSE2"].to_list() == dense[:, 1].tolist()

    variables = cl.build_frame(data, axis=1, include_dimensions=True)
    assert variables["V_SPARSE1"].to_list() == [1.0, 1.0, 1.0]


def test_integer_matrix_keeps_integer_values():
    """An integer matrix keeps its integer dtype and values."""
    data = AnnData(
        X=np.arange(6, dtype=np.int64).reshape(2, 3),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    frame = cl.build_frame(data, variable_keys=["g3", "g1"])
    assert frame.select("g3", "g1").rows() == [(2, 0), (5, 3)]
    assert frame["g3"].dtype == pl.Int64


def test_variable_values_match_matrix_for_unsorted_keys(adata):
    """Unsorted `variable_keys` map to the matching matrix columns."""
    keys = ["MS4A1", "CD14", "CD3E"]
    frame = cl.build_frame(adata, variable_keys=keys)
    expected = adata[:, keys].X.toarray()
    np.testing.assert_array_equal(frame.select(keys).to_numpy(), expected)


# --- single-modality multimodal containers ------------------------------------


def test_modality_source_translates_column_for_auto_selected_modality(single_modality):
    """A qualified group column is translated for the sole modality."""
    holder, column = _modality_source(single_modality, None, "rna:leiden")
    assert column == "leiden"
    assert column in holder.obs.columns


def test_container_column_resolves_auto_selected_modality(single_modality):
    """A modality column maps back to the container for the sole modality."""
    assert _container_column(single_modality, None, "leiden") == "rna:leiden"


def test_multimodal_variables_align_with_modality_matrix(mudata):
    """Multimodal variables align to container observations through `obsmap`."""
    keys = {"rna": "CD14", "prot": "CD3_TotalSeqB"}
    frame = cl.build_frame(mudata, variable_keys=[f"{name}:{key}" for name, key in keys.items()])
    for name, key in keys.items():
        part = mudata[name]
        values = part[:, key].X
        values = values.toarray() if sparse.issparse(values) else np.asarray(values)
        aligned = pd.Series(values.reshape(-1), index=part.obs_names)
        expected = aligned.reindex(mudata.obs_names).to_numpy()
        np.testing.assert_allclose(frame[f"{name}:{key}"].to_numpy(), expected)


# --- datasets -----------------------------------------------------------------


def test_from_url_reads_extensionless_url_as_requested_format(tmp_path, mudata):
    """A URL without a file suffix is cached and read as `extension`."""
    source = tmp_path / "source"
    source.mkdir()
    payload = source / "download"
    mudata.write(payload)

    cache = tmp_path / "cache"
    data = datasets.from_url(
        payload.as_uri(), cache_directory=cache, bring=False, extension=".h5mu"
    )

    assert isinstance(data, MuData)
    assert (cache / "download.h5mu").exists()


def test_pbmc_cite_is_multimodal(mudata):
    """`pbmc_cite` loads as a two-modality object with 411 cells."""
    assert isinstance(mudata, MuData)
    assert sorted(mudata.mod) == ["prot", "rna"]
    assert mudata.n_obs == 411
