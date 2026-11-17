"""Regression tests for defects found in the 2026-11-18 audit of core, common, layers and spatial."""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from ggrust import aes
from shapely.geometry import Point
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

import cellestial as cl
from cellestial.util import retrieve
from cellestial.util.errors import (
    CellestialWarning,
    InvalidArgumentError,
    KeyNotFoundError,
)


def _embedding_data() -> AnnData:
    """Small data with a categorical key, a gene and a 2-component UMAP."""
    rng = np.random.default_rng(0)
    n = 30
    data = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(["a", "b", "c"] * 10)},
            index=[f"cell_{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["GENE_A", "GENE_B"]),
    )
    data.obsm["X_umap"] = rng.random((n, 2))
    return data


# ---- dimensional: xy components ----


@pytest.mark.parametrize("xy", [(0, 1), (-1, 2)])
def test_dimensional_rejects_zero_based_components(xy):
    data = _embedding_data()
    with pytest.raises(InvalidArgumentError, match=r"(?s)X_UMAP.*1-based"):
        cl.umap(data, "cluster", xy=xy)


def test_dimensional_rejects_components_beyond_embedding():
    data = _embedding_data()
    with pytest.raises(InvalidArgumentError, match=r"(?s)X_UMAP.*2 components.*1-based"):
        cl.umap(data, "cluster", xy=(1, 3))


def test_dimensionals_rejects_components_outside_embedding():
    data = _embedding_data()
    with pytest.raises(InvalidArgumentError, match="1-based"):
        cl.umaps(data, ["cluster", "GENE_A"], xy=(0, 1))


# ---- plural dimensional: unknown key ----


@pytest.mark.parametrize("function", [cl.dimensionals, cl.umaps])
def test_dimensional_plurals_unknown_key_raises_key_error(function):
    data = _embedding_data()
    with pytest.raises(KeyNotFoundError, match="clustr"):
        function(data, ["cluster", "clustr"])


# ---- distributions / ridges: a bare string for `keys` ----


@pytest.mark.parametrize("function", [cl.violins, cl.boxplots])
def test_distribution_plurals_reject_string_keys(function):
    data = _embedding_data()
    with pytest.raises(TypeError, match="Sequence"):
        function(data, "GENE_A", group_by="cluster")


def test_ridges_reject_string_keys():
    data = _embedding_data()
    with pytest.raises(TypeError, match="Sequence"):
        cl.ridges(data, "GENE_A", "cluster")


@pytest.mark.parametrize("keys", [np.array(["GENE_A", "GENE_B"]), pd.Index(["GENE_A", "GENE_B"])])
def test_distribution_plurals_accept_array_like_keys(keys):
    data = _embedding_data()
    cl.violins(data, keys, group_by="cluster")
    cl.ridges(data, keys, "cluster")


# ---- distribution: groups / drop without group_by ----


@pytest.mark.parametrize("parameter", ["groups", "drop"])
def test_violin_groups_without_group_by_warns(parameter):
    data = _embedding_data()
    with pytest.warns(CellestialWarning, match=f"`{parameter}`.*`group_by`"):
        cl.violin(data, "GENE_A", **{parameter: ["a"]})


# ---- xyplot: missing axis column ----


def test_xyplot_missing_embedding_component_raises_key_error():
    data = _embedding_data()
    with pytest.raises(KeyNotFoundError, match=r"X_UMAP3.*did you mean"):
        cl.xyplot(data, "X_UMAP3", "X_UMAP1")


def test_xyplot_mapping_override_skips_missing_column_check():
    data = _embedding_data()
    data.obs["score"] = np.arange(data.n_obs, dtype=float)
    plot = cl.xyplot(data, "not_a_column", "GENE_A", mapping=aes(x="score"))
    assert plot.as_dict()["layers"][0]["mapping"]["x"] == "score"


# ---- highest_expressed_genes / elbow: non-positive counts ----


@pytest.mark.parametrize("n", [0, -1])
def test_highest_expressed_genes_rejects_non_positive_n(n):
    data = _embedding_data()
    with pytest.raises(InvalidArgumentError, match="`n`"):
        cl.highest_expressed_genes(data, n=n)


@pytest.mark.parametrize("n_pcs", [0, -1])
def test_elbow_rejects_non_positive_n_pcs(n_pcs):
    data = _embedding_data()
    data.uns["pca"] = {"variance_ratio": np.array([0.5, 0.3, 0.2])}
    with pytest.raises(InvalidArgumentError, match="`n_pcs`"):
        cl.elbow(data, n_pcs=n_pcs)


# ---- stream: embeddings without the `X_` prefix ----


def test_stream_derives_velocity_for_unprefixed_embedding():
    data = _embedding_data()
    data.obsm["wnn_umap"] = data.obsm["X_umap"]
    # constant rightward velocity, so every arrow must point right
    data.obsm["velocity_wnn_umap"] = np.tile([1.0, 0.0], (data.n_obs, 1))
    plot = cl.dimensional(data, "cluster", use_key="wnn_umap") + cl.stream(n_neighbors=5)
    arrows = plot.as_dict()["layers"][-1]["data"]
    dx = np.asarray(arrows["xend"]) - np.asarray(arrows["x"])
    dy = np.asarray(arrows["yend"]) - np.asarray(arrows["y"])
    assert len(dx) > 0
    assert np.all(dx > 0)
    assert np.allclose(dy, 0, atol=1e-6 * np.abs(dx).max())


def test_stream_unprefixed_embedding_without_velocity_raises():
    data = _embedding_data()
    data.obsm["wnn_umap"] = data.obsm["X_umap"]
    plot = cl.dimensional(data, "cluster", use_key="wnn_umap")
    with pytest.raises(KeyError, match="VELOCITY_WNN_UMAP1"):
        plot + cl.stream(n_neighbors=5)


# ---- spatial: SpatialData point shapes align on the instance key ----


def _spatialdata_reordered(table_ids: list[int]) -> SpatialData:
    """Shapes at x = 10 * instance id; the table lists `table_ids` in its own order."""
    n_shapes = 4
    table = AnnData(
        X=np.zeros((len(table_ids), 1), dtype="float32"),
        obs=pd.DataFrame(
            {
                "label": pd.Categorical([f"spot{i}" for i in table_ids]),
                "instance_id": table_ids,
                "region": ["spots"] * len(table_ids),
            },
            index=[f"c{i}" for i in table_ids],
        ),
        var=pd.DataFrame(index=["G"]),
    )
    table = TableModel.parse(
        table, region="spots", region_key="region", instance_key="instance_id"
    )
    shapes = gpd.GeoDataFrame(
        {"radius": [1.0] * n_shapes},
        geometry=[Point(i * 10.0, 0.0) for i in range(n_shapes)],
        index=np.arange(n_shapes),
    )
    return SpatialData(shapes={"spots": ShapesModel.parse(shapes)}, tables={"table": table})


@pytest.mark.parametrize("table_ids", [[3, 2, 1, 0], [2, 0]])
def test_spatialdata_points_align_on_instance_key(table_ids):
    data = _spatialdata_reordered(table_ids)
    frame = retrieve(cl.spatial(data, "label", image=False))
    positions = dict(zip(frame["label"].to_list(), frame["spatial_x"].to_list(), strict=True))
    assert positions == {f"spot{i}": i * 10.0 for i in table_ids}
