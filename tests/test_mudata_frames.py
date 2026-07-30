import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from mudata import MuData, set_options
from scipy import sparse

from cellestial._mudata import _resolve_mudata_variable
from cellestial.frames.mudata import (
    mudata_observations_frame,
    mudata_variable_columns,
    mudata_variables_frame,
)
from cellestial.util.errors import VariableNotFoundError


def _mudata_fixture() -> MuData:
    rna = AnnData(
        X=np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
        obs=pd.DataFrame(
            {"private_rna": ["x", "y", "z"]},
            index=["cell_a", "cell_b", "cell_c"],
        ),
        var=pd.DataFrame(
            {"private_rna_variable": [1, 2, 3]},
            index=["shared", "rna_gene", "chr1:10-20"],
        ),
    )
    protein = AnnData(
        X=np.array(
            [
                [30.0, 40.0],
                [10.0, 20.0],
            ]
        ),
        obs=pd.DataFrame(
            {"private_protein": ["z", "x"]},
            index=["cell_c", "cell_a"],
        ),
        var=pd.DataFrame(
            {"private_protein_variable": [4, 5]},
            index=["shared", "protein_marker"],
        ),
    )

    with (
        set_options(pull_on_update=False),
        pytest.warns(UserWarning, match="var_names are not unique"),
    ):
        data = MuData({"rna": rna, "protein": protein}, axis=0)

    data.obs["celltype"] = pd.Categorical(["A", "B", "C"])
    data.var["feature_type"] = ["gene", "gene", "peak", "protein", "protein"]
    data.obsm["X_joint"] = np.array(
        [
            [0.0, 0.5],
            [1.0, 1.5],
            [2.0, 2.5],
        ]
    )
    data.varm["LFs"] = np.arange(10, dtype=float).reshape(5, 2)
    return data


def test_resolve_mudata_variable_uses_first_separator():
    """Keep colons within a variable name after resolving its modality."""
    data = _mudata_fixture()

    assert _resolve_mudata_variable(data, "rna:chr1:10-20") == (
        "rna",
        "chr1:10-20",
        2,
    )
    assert _resolve_mudata_variable(data, "protein:shared") == (
        "protein",
        "shared",
        0,
    )


@pytest.mark.parametrize("key", ["shared", ":shared", "rna:", "unknown:shared"])
def test_resolve_mudata_variable_rejects_invalid_qualified_key(key):
    """Reject variable keys without a known, nonempty modality prefix."""
    data = _mudata_fixture()

    with pytest.raises(VariableNotFoundError):
        _resolve_mudata_variable(data, key)


def test_mudata_variable_columns_align_with_obsmap():
    """Align reordered and missing modality observations to global rows."""
    data = _mudata_fixture()

    columns = mudata_variable_columns(
        data,
        column_names=[],
        keys=["rna:rna_gene", "protein:shared"],
    )

    assert columns[0].to_list() == [2.0, 5.0, 8.0]
    assert columns[1].to_list() == [10.0, None, 30.0]


def test_mudata_variable_columns_support_sparse_matrices():
    """Read qualified variables from sparse modality matrices."""
    data = _mudata_fixture()
    data.mod["protein"].X = sparse.csr_matrix(data.mod["protein"].X)

    (column,) = mudata_variable_columns(
        data,
        column_names=[],
        keys="protein:protein_marker",
    )

    assert column.to_list() == [20.0, None, 40.0]


def test_mudata_observations_frame_uses_only_container_annotations():
    """Exclude metadata that exists only inside individual modalities."""
    data = _mudata_fixture()

    frame = mudata_observations_frame(
        data,
        variable_keys="protein:shared",
        include_dimensions=2,
        dimension_keys=["x_joint"],
    )

    assert frame.columns == [
        "Barcode",
        "celltype",
        "X_JOINT1",
        "X_JOINT2",
        "protein:shared",
    ]
    assert frame["Barcode"].to_list() == ["cell_a", "cell_b", "cell_c"]
    assert frame["protein:shared"].to_list() == [10.0, None, 30.0]
    assert frame["celltype"].dtype == pl.Categorical
    assert "private_rna" not in frame.columns
    assert "private_protein" not in frame.columns


def test_mudata_variables_frame_qualifies_identifiers_and_uses_global_metadata():
    """Namespace variables while reading only container-level annotations."""
    data = _mudata_fixture()

    frame = mudata_variables_frame(
        data,
        include_dimensions=1,
        dimension_keys=["lfs"],
    )

    assert frame["Variable"].to_list() == [
        "rna:shared",
        "rna:rna_gene",
        "rna:chr1:10-20",
        "protein:shared",
        "protein:protein_marker",
    ]
    assert frame.columns == ["Variable", "feature_type", "LFS1"]
    assert "private_rna_variable" not in frame.columns
    assert "private_protein_variable" not in frame.columns


@pytest.mark.parametrize("axis", [1, -1])
def test_mudata_frames_reject_non_observation_shared_layouts(axis):
    """Reject MuData layouts that do not use shared observations."""
    modality = AnnData(
        X=np.ones((2, 2)),
        obs=pd.DataFrame(index=["cell_a", "cell_b"]),
        var=pd.DataFrame(index=["gene_a", "gene_b"]),
    )
    with set_options(pull_on_update=False):
        data = MuData({"modality": modality}, axis=axis)

    with pytest.raises(ValueError, match=r"data\.axis == 0"):
        mudata_observations_frame(data)
    with pytest.raises(ValueError, match=r"data\.axis == 0"):
        mudata_variables_frame(data)
