import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from scipy import sparse
from spatialdata import SpatialData

import cellestial as cl
from cellestial.frames.build import anndata_observations_frame, anndata_variables_frame
from cellestial.frames.operations import _highest_expressed_genes_frame, _pca_variance_frame


def test_build_frame_requires_axis(adata):
    with pytest.raises(ValueError):
        cl.build_frame(adata)


def test_build_frame_obs_axis_with_dims(adata):
    frame = cl.build_frame(adata, axis=0, include_dimensions=2)
    assert isinstance(frame, pl.DataFrame)
    # Should include UMAP/PCA/TSNE component columns
    cols = frame.columns
    assert any("umap" in c.lower() or "UMAP" in c for c in cols)


def test_build_frame_variable_keys_infers_axis(adata):
    frame = cl.build_frame(adata, variable_keys=["CD14", "MS4A1"], include_dimensions=2)
    assert isinstance(frame, pl.DataFrame)
    assert "CD14" in frame.columns
    assert "MS4A1" in frame.columns


def test_build_frame_axis_1(adata):
    frame = cl.build_frame(adata, axis=1)
    assert isinstance(frame, pl.DataFrame)
    # axis=1 -> one row per var
    assert frame.height == adata.n_vars


def test_build_frame_invalid_key(adata):
    with pytest.raises(Exception):
        cl.build_frame(adata, variable_keys=["NOT_A_GENE_xyz"])


def test_build_frame_include_dimensions_int(adata):
    frame = cl.build_frame(adata, axis=0, include_dimensions=3)
    # 3 components requested
    cols_upper = [c for c in frame.columns if "UMAP" in c]
    assert len(cols_upper) >= 1


def test_anndata_observations_frame_dimension_validation():
    data = AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {"numeric_category": pd.Categorical([1, 2, 1])},
            index=["a", "b", "c"],
        ),
    )
    data.var_names = ["gene_a", "gene_b"]
    data.obsm["X_demo"] = np.ones((3, 2))

    frame = anndata_observations_frame(
        data,
        variable_keys="gene_a",
        include_dimensions=True,
    )

    assert "X_DEMO1" in frame.columns
    assert "gene_a" in frame.columns
    with pytest.raises(ValueError, match="negative"):
        anndata_observations_frame(data, include_dimensions=-1)
    with pytest.raises(TypeError, match="bool"):
        anndata_observations_frame(data, include_dimensions="yes")
    with pytest.raises(Exception, match="Expected"):
        anndata_observations_frame("not adata")


def test_anndata_variables_frame_dimension_validation():
    data = AnnData(
        X=np.ones((2, 3)),
        var=pd.DataFrame(
            {"numeric_category": pd.Categorical([1, 2, 1])},
            index=["gene_a", "gene_b", "gene_c"],
        ),
    )
    data.varm["PCs"] = np.ones((3, 2))

    frame = anndata_variables_frame(data, include_dimensions=1)

    assert "PCS1" in frame.columns
    with pytest.raises(ValueError, match="negative"):
        anndata_variables_frame(data, include_dimensions=-1)
    with pytest.raises(TypeError, match="bool"):
        anndata_variables_frame(data, include_dimensions="yes")
    with pytest.raises(Exception, match="Expected"):
        anndata_variables_frame("not adata")


def test_frame_builders_select_metadata_dimensions_and_identifiers():
    data = AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame({"keep": [1, 2, 3], "skip": [4, 5, 6]}, index=["a", "b", "c"]),
        var=pd.DataFrame({"keep_var": [7, 8], "skip_var": [9, 10]}, index=["g1", "g2"]),
    )
    data.obsm["X_demo"] = np.ones((3, 2))
    data.obsm["X_other"] = np.ones((3, 2))
    data.varm["PCs"] = np.ones((2, 2))

    observations = anndata_observations_frame(
        data,
        observations_name=None,
        metadata_columns=["keep"],
        include_dimensions=1,
        dimension_keys=["x_DEMO"],
    )
    variables = anndata_variables_frame(
        data,
        variables_name=None,
        metadata_columns=["keep_var"],
        include_dimensions=True,
        dimension_keys=["pcs"],
    )

    assert observations.columns == ["keep", "X_DEMO1"]
    assert variables.columns == ["keep_var", "PCS1", "PCS2"]


def test_frame_builders_allow_empty_embedding_selection():
    data = AnnData(X=np.ones((2, 2)))
    data.obsm["X_demo"] = np.ones((2, 2))
    data.varm["PCs"] = np.ones((2, 2))

    observations = anndata_observations_frame(
        data,
        metadata_columns=[],
        include_dimensions=True,
        dimension_keys=[],
    )
    variables = anndata_variables_frame(
        data,
        metadata_columns=[],
        include_dimensions=True,
        dimension_keys=[],
    )

    assert observations.columns == ["Barcode"]
    assert variables.columns == ["Variable"]


@pytest.mark.parametrize(
    ("builder", "kwargs", "message"),
    [
        (anndata_observations_frame, {"metadata_columns": ["missing"]}, "observations"),
        (anndata_variables_frame, {"metadata_columns": ["missing"]}, "variables"),
        (
            anndata_observations_frame,
            {"include_dimensions": True, "dimension_keys": ["missing"]},
            "embeddings",
        ),
        (
            anndata_variables_frame,
            {"include_dimensions": True, "dimension_keys": ["missing"]},
            "embeddings",
        ),
    ],
)
def test_frame_builders_reject_unknown_selections(builder, kwargs, message):
    data = AnnData(X=np.ones((2, 2)))

    with pytest.raises(KeyError, match=message):
        builder(data, **kwargs)


def test_build_frame_rejects_spatialdata_without_tables():
    with pytest.raises(ValueError, match="No annotation tables"):
        cl.build_frame(SpatialData(), axis=0)


def test_highest_expressed_genes_frame_dense_and_sparse():
    data = AnnData(X=np.array([[1.0, 3.0, 0.0], [0.0, 2.0, 2.0]]))
    data.var_names = ["gene_a", "gene_b", "gene_c"]

    dense = _highest_expressed_genes_frame(data, n=2)
    sparse_frame = _highest_expressed_genes_frame(
        AnnData(X=sparse.csr_matrix(data.X), obs=data.obs, var=data.var),
        n=2,
    )

    assert dense.columns == ["gene_b", "gene_c"]
    assert sparse_frame.columns == dense.columns


def test_highest_expressed_genes_frame_errors():
    data = AnnData(X=np.ones((2, 2)))

    with pytest.raises(ValueError, match="only 2 genes"):
        _highest_expressed_genes_frame(data, n=3)
    with pytest.raises(Exception, match="Expected"):
        _highest_expressed_genes_frame("not adata")


def test_pca_variance_frame_and_elbow_label():
    data = AnnData(X=np.ones((2, 3)))
    data.uns["pca"] = {"variance_ratio": np.array([0.5, 0.3, 0.2])}

    frame = _pca_variance_frame(data, n_pcs=2)
    plot = cl.elbow(data, n_pcs=3, label=True, label_every=1)

    assert frame["Principal Component"].to_list() == [1, 2]
    assert plot is not None


def test_pca_variance_frame_errors():
    data = AnnData(X=np.ones((2, 3)))

    with pytest.raises(ValueError, match="PCA variance ratio"):
        _pca_variance_frame(data)

    data.uns["pca"] = {"variance_ratio": np.array([0.5])}
    with pytest.raises(ValueError, match="only 1 components"):
        _pca_variance_frame(data, n_pcs=2)
    with pytest.raises(Exception, match="Expected"):
        _pca_variance_frame("not adata")


def test_pca_variance_frame_custom_use_key():
    data = AnnData(X=np.ones((2, 3)))
    data.uns["pca_harmony"] = {"variance_ratio": np.array([0.6, 0.3, 0.1])}
    frame = _pca_variance_frame(data, n_pcs=2, use_key="pca_harmony")
    assert frame["Variance Ratio"].to_list() == [0.6, pytest.approx(0.3)]


def test_pca_variance_frame_missing_custom_key_names_it():
    data = AnnData(X=np.ones((2, 3)))
    with pytest.raises(ValueError, match="`pca_harmony`"):
        _pca_variance_frame(data, use_key="pca_harmony")


def test_elbow_pca_key_threads_through():
    data = AnnData(X=np.ones((2, 3)))
    data.uns["pca_harmony"] = {"variance_ratio": np.array([0.5, 0.3, 0.2])}
    plot = cl.elbow(data, pca_key="pca_harmony")
    assert plot is not None
