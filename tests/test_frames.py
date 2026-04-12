import polars as pl
import pytest

import cellestial as cl


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
    frame = cl.build_frame(
        adata, variable_keys=["CD14", "MS4A1"], include_dimensions=2
    )
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
