import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec
from shapely.geometry import MultiPolygon, Point
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, set_transformation
from xarray import DataArray

import cellestial as cl
from cellestial.spatial.utilities import (
    _image_to_yxc,
    _resolve_coordinate_system,
    _resolve_image_name,
    _resolve_table,
    _select_one,
    _spatial_components,
)

# ---- fixtures ----


@pytest.fixture
def data_minimal():
    """A small SpatialData with one image, one shapes element, one table."""
    n = 6
    rng = np.random.default_rng(0)

    table = AnnData(
        X=rng.random((n, 3)).astype("float32"),
        obs=pd.DataFrame(
            {
                "cluster": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
                "n_counts": rng.integers(1, 100, n).astype("float32"),
                "instance_id": np.arange(n),
                "region": ["spots"] * n,
            },
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["GENE_A", "GENE_B", "GENE_C"]),
    )
    table = TableModel.parse(
        table, region="spots", region_key="region", instance_key="instance_id"
    )

    geometry = [Point(i * 5.0, i * 7.0) for i in range(n)]
    shapes = gpd.GeoDataFrame({"radius": [1.0] * n}, geometry=geometry, index=np.arange(n))
    shapes = ShapesModel.parse(shapes)

    image = Image2DModel.parse(rng.random((3, 32, 32)).astype("float32"), dims=("c", "y", "x"))

    return SpatialData(
        images={"img": image},
        shapes={"spots": shapes},
        tables={"table": table},
    )


@pytest.fixture
def data_multi():
    """SpatialData with multiple tables, images, and shapes — forces selection."""
    n = 4
    rng = np.random.default_rng(1)

    def _table():
        a = AnnData(
            X=rng.random((n, 2)).astype("float32"),
            obs=pd.DataFrame(
                {
                    "instance_id": np.arange(n),
                    "region": ["spots1"] * n,
                },
                index=[f"c{i}" for i in range(n)],
            ),
            var=pd.DataFrame(index=["G1", "G2"]),
        )
        return TableModel.parse(
            a, region="spots1", region_key="region", instance_key="instance_id"
        )

    shapes_gdf = gpd.GeoDataFrame(
        {"radius": [1.0] * n},
        geometry=[Point(i, i) for i in range(n)],
        index=np.arange(n),
    )
    shapes_gdf = ShapesModel.parse(shapes_gdf)
    image = Image2DModel.parse(rng.random((3, 8, 8)).astype("float32"), dims=("c", "y", "x"))
    image2 = Image2DModel.parse(rng.random((3, 8, 8)).astype("float32"), dims=("c", "y", "x"))
    return SpatialData(
        images={"img1": image, "img2": image2},
        shapes={"spots1": shapes_gdf, "spots2": shapes_gdf.copy()},
        tables={"t1": _table(), "t2": _table()},
    )


@pytest.fixture
def data_polygons():
    """SpatialData whose shapes element holds polygons (rendered via geom_polygon)."""
    n = 3
    rng = np.random.default_rng(2)
    table = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame(
            {
                "cluster": pd.Categorical(["a", "b", "a"]),
                "score": rng.random(n).astype("float32"),
                "instance_id": np.arange(n),
                "region": ["polys"] * n,
            },
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    table = TableModel.parse(
        table, region="polys", region_key="region", instance_key="instance_id"
    )

    polys = gpd.GeoDataFrame(
        geometry=[Point(i * 5, i * 5).buffer(1.0) for i in range(n)],
        index=np.arange(n),
    )
    polys = ShapesModel.parse(polys)
    return SpatialData(shapes={"polys": polys}, tables={"table": table})


# ---- spatial() with SpatialData ----


def test_spatial_sdata_no_key(data_minimal):
    plot = cl.spatial(data_minimal)
    assert isinstance(plot, PlotSpec)


def test_spatial_sdata_categorical_key(data_minimal):
    plot = cl.spatial(data_minimal, key="cluster")
    assert isinstance(plot, PlotSpec)


def test_spatial_sdata_numeric_key(data_minimal):
    plot = cl.spatial(data_minimal, key="n_counts")
    assert isinstance(plot, PlotSpec)


def test_spatial_sdata_gene_key(data_minimal):
    plot = cl.spatial(data_minimal, key="GENE_A")
    assert isinstance(plot, PlotSpec)


def test_spatial_sdata_no_image(data_minimal):
    plot = cl.spatial(data_minimal, key="cluster", image=False)
    assert isinstance(plot, PlotSpec)


def test_spatial_sdata_ambiguous_table_raises(data_multi):
    with pytest.raises(ValueError, match="Multiple table"):
        cl.spatial(data_multi)


def test_spatial_sdata_ambiguous_resolved_with_kwargs(data_multi):
    plot = cl.spatial(
        data_multi,
        table_name="t1",
        image_name="img1",
        shapes_name="spots1",
    )
    assert isinstance(plot, PlotSpec)


def test_spatial_data_polygons_default_renders_centroids(data_polygons):
    """polygon=False (default) collapses Polygons to centroids."""
    plot = cl.spatial(data_polygons, key="cluster")
    assert isinstance(plot, PlotSpec)


def test_spatial_data_polygons_no_key(data_polygons):
    plot = cl.spatial(data_polygons, polygon=True)
    assert isinstance(plot, PlotSpec)


def test_spatial_data_polygons_categorical_key(data_polygons):
    plot = cl.spatial(data_polygons, key="cluster", polygon=True)
    assert isinstance(plot, PlotSpec)


def test_spatial_data_polygons_numeric_key(data_polygons):
    plot = cl.spatial(data_polygons, key="score", polygon=True)
    assert isinstance(plot, PlotSpec)


def test_spatial_data_polygons_gene_key(data_polygons):
    plot = cl.spatial(data_polygons, key="G1", polygon=True)
    assert isinstance(plot, PlotSpec)


def test_spatial_data_smart_shapes_from_table(data_minimal):
    """No shapes_name: should fall back to the table's annotated region."""
    plot = cl.spatial(data_minimal, key="cluster")
    assert isinstance(plot, PlotSpec)


def test_spatial_data_smart_resolution_with_image_name(data_multi):
    """Selection collapses when only image_name + table_name are given."""
    plot = cl.spatial(
        data_multi,
        table_name="t1",
        image_name="img1",
    )
    assert isinstance(plot, PlotSpec)


def test_spatial_sdata_unknown_table_raises(data_minimal):
    with pytest.raises(KeyError):
        cl.spatial(data_minimal, table_name="does_not_exist")


def test_spatial_sdata_unknown_coordinate_system_raises(data_minimal):
    with pytest.raises(KeyError):
        cl.spatial(data_minimal, coordinate_system="not_a_cs")


def test_spatial_invalid_data_type_raises():
    with pytest.raises(TypeError):
        cl.spatial("not a real object")  # type: ignore[arg-type]


# ---- spatials() with SpatialData ----


def test_spatials_sdata_returns_subplots(data_minimal):
    plot = cl.spatials(data_minimal, ["cluster", "GENE_A"])
    assert isinstance(plot, SupPlotsSpec)


def test_spatials_sdata_with_selection_kwargs(data_multi):
    plot = cl.spatials(
        data_multi,
        ["G1", "G2"],
        table_name="t1",
        image_name="img1",
        shapes_name="spots1",
    )
    assert isinstance(plot, SupPlotsSpec)


def test_spatials_with_layers_share_labels_and_interactive(data_minimal):
    from lets_plot import geom_point

    plot = cl.spatials(
        data_minimal,
        ["cluster", "GENE_A"],
        image=False,
        layers=geom_point(size=0.1),
        share_labels=True,
        interactive=True,
        ncol=1,
    )

    assert isinstance(plot, SupPlotsSpec)


def test_spatials_invalid_keys_type(data_minimal):
    with pytest.raises(TypeError, match="keys"):
        cl.spatials(data_minimal, "cluster")


# ---- AnnData regression: existing path still works ----


def test_spatial_anndata_generic_no_uns():
    """AnnData without spatial uns metadata should still produce a plot."""
    n = 5
    rng = np.random.default_rng(3)
    adata = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame(
            {"cluster": pd.Categorical(["a"] * n)},
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    adata.obsm["spatial"] = rng.random((n, 2)).astype("float32")
    plot = cl.spatial(adata, key="cluster")
    assert isinstance(plot, PlotSpec)


def test_spatial_scale_axis_constant_key_returns_zero():
    """Constant numeric spatial keys should not scale to NaN."""
    n = 5
    rng = np.random.default_rng(12)
    data = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame(
            {"score": [3.0] * n},
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    data.obsm["spatial"] = rng.random((n, 2)).astype("float32")
    plot = cl.spatial(data, key="score", scale_axis=0)
    assert plot.as_dict()["data"]["score"].to_list() == [0.0] * n


def test_spatial_anndata_visium_metadata_paths():
    n = 4
    rng = np.random.default_rng(13)
    data = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame(
            {
                "cluster": pd.Categorical(["a", "b", "a", "b"]),
                "score": [1.0, 2.0, 3.0, 4.0],
            },
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    data.obsm["spatial"] = rng.random((n, 2)).astype("float32")
    data.uns["spatial"] = {
        "lib": {
            "images": {"hires": rng.random((8, 8, 4)).astype("float32")},
            "scalefactors": {"tissue_hires_scalef": 2.0},
        }
    }

    plot = cl.spatial(
        data,
        key="score",
        greyscale=True,
        image_alpha=0.5,
        groups=["a"],
        crop=[0, 10, 0, 10],
        scale_axis=0,
        interactive=True,
    )

    assert isinstance(plot, PlotSpec)


def test_spatial_anndata_visium_errors_and_warnings():
    n = 3
    rng = np.random.default_rng(14)
    data = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame({"score": [1.0, 2.0, 3.0]}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    data.obsm["spatial"] = rng.random((n, 2)).astype("float32")
    data.uns["spatial"] = {
        "a": {"images": {}, "scalefactors": {}},
        "b": {"images": {}, "scalefactors": {}},
    }

    with pytest.raises(ValueError, match="Multiple spatial libraries"):
        cl.spatial(data)
    with pytest.raises(KeyError, match="library_id"):
        cl.spatial(data, library_id="missing")
    with pytest.warns(cl.util.errors.CellestialWarning, match="image"):
        cl.spatial(data, library_id="a", image=True)
    with pytest.warns(cl.util.errors.CellestialWarning, match="not categorical"):
        cl.spatial(data, key="score", library_id="a", image=False, groups=["a"])
    with pytest.raises(ValueError, match="crop"):
        cl.spatial(data, library_id="a", image=False, crop=[1, 2, 3])


def _visium_like(scalefactors: dict) -> AnnData:
    n = 3
    rng = np.random.default_rng(15)
    data = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame({"score": [1.0, 2.0, 3.0]}, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    data.obsm["spatial"] = rng.random((n, 2)).astype("float32")
    data.uns["spatial"] = {
        "lib": {
            "images": {"lowres": rng.random((4, 4, 4)).astype("float32")},
            "scalefactors": scalefactors,
        }
    }
    return data


import warnings as _warnings_module


def _scalef_warnings(caught) -> list:
    return [
        w
        for w in caught
        if issubclass(w.category, cl.util.errors.CellestialWarning)
        and "Scale factor" in str(w.message)
    ]


def test_spatial_scalefactor_present_silent():
    data = _visium_like({"tissue_lowres_scalef": 0.5})
    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        cl.spatial(data, image=True)
    assert _scalef_warnings(caught) == []


def test_spatial_scalefactor_missing_with_image_warns():
    data = _visium_like({})  # scalefactor absent
    with pytest.warns(cl.util.errors.CellestialWarning, match="tissue_lowres_scalef"):
        cl.spatial(data, image=True)


def test_spatial_scalefactor_missing_without_image_silent():
    data = _visium_like({})
    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always")
        cl.spatial(data, image=False)
    assert _scalef_warnings(caught) == []


def test_spatial_components_anndata_errors():
    data = AnnData(X=np.ones((2, 2)), var=pd.DataFrame(index=["G1", "G2"]))

    with pytest.raises(KeyError, match="spot coordinates"):
        _spatial_components(
            data,
            library_id=None,
            image_key="hires",
            image=False,
            spatial_key="spatial",
        )

    data.obsm["spatial"] = np.ones((2, 2))
    with pytest.raises(KeyError, match="no spatial"):
        _spatial_components(
            data,
            library_id="lib",
            image_key="hires",
            image=False,
            spatial_key="spatial",
        )
    with pytest.raises(Exception, match="Unsupported data type"):
        _spatial_components(
            "not data",
            library_id=None,
            image_key="hires",
            image=False,
            spatial_key="spatial",
        )


# ---- build_frame() with SpatialData ----


def test_build_frame_sdata_single_table(data_minimal):
    frame = cl.build_frame(data_minimal, axis=0)
    assert "cluster" in frame.columns


def test_build_frame_data_multi_table_raises(data_multi):
    with pytest.raises(ValueError, match="Multiple annotation tables"):
        cl.build_frame(data_multi, axis=0)


def test_spatial_component_selection_helpers(data_minimal, data_multi):
    assert _select_one("img", ["img"], "image") == "img"
    assert _select_one(None, ["img"], "image") == "img"

    with pytest.raises(KeyError, match="image"):
        _select_one("missing", ["img"], "image")
    with pytest.raises(ValueError, match="No image"):
        _select_one(None, [], "image")
    with pytest.raises(ValueError, match="Multiple image"):
        _select_one(None, ["a", "b"], "image")
    with pytest.raises(TypeError, match="SpatialData"):
        _resolve_table("not sdata", None)
    with pytest.raises(KeyError, match="image"):
        _resolve_image_name(data_minimal, "missing", coordinate_system="global")
    with pytest.raises(ValueError, match="Multiple image"):
        _resolve_image_name(data_multi, None, coordinate_system="global")
    assert (
        _resolve_coordinate_system(
            data_multi,
            None,
            shapes_name="spots1",
            image_name=None,
            want_image=True,
        )
        == "global"
    )


def test_image_to_yxc_dataarray_channel_handling():
    rgb = DataArray(np.ones((3, 4, 5)), dims=("c", "y", "x"))
    single_channel = DataArray(np.ones((1, 4, 5)), dims=("c", "y", "x"))
    already_yx = DataArray(np.ones((4, 5)), dims=("y", "x"))

    assert _image_to_yxc(rgb).shape == (4, 5, 3)
    assert _image_to_yxc(single_channel).shape == (4, 5)
    assert _image_to_yxc(already_yx).shape == (4, 5)


# ---- Visium HD-style multi-coordinate-system layout ----


@pytest.fixture
def data_visium_hd_like():
    """
    Mimics the Visium HD layout: 2 images (hires/lowres), shared shapes,
    one global CS holding all elements plus per-resolution CSes that hold
    only the matching image. Forces the CS tiebreaker to fire.
    """
    n = 4
    rng = np.random.default_rng(7)

    table = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame(
            {
                "score": rng.random(n).astype("float32"),
                "cell_id": np.arange(n),
                "region": ["cells"] * n,
            },
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    table = TableModel.parse(table, region="cells", region_key="region", instance_key="cell_id")

    cells = gpd.GeoDataFrame(
        {"radius": [1.0] * n},
        geometry=[Point(i * 10.0, i * 10.0) for i in range(n)],
        index=np.arange(n),
    )
    cells = ShapesModel.parse(cells)

    hires = Image2DModel.parse(rng.random((3, 32, 32)).astype("float32"), dims=("c", "y", "x"))
    lowres = Image2DModel.parse(rng.random((3, 8, 8)).astype("float32"), dims=("c", "y", "x"))

    set_transformation(hires, Identity(), to_coordinate_system="hires_cs")
    set_transformation(lowres, Identity(), to_coordinate_system="lowres_cs")
    set_transformation(cells, Identity(), to_coordinate_system="hires_cs")
    set_transformation(cells, Identity(), to_coordinate_system="lowres_cs")

    return SpatialData(
        images={"hires": hires, "lowres": lowres},
        shapes={"cells": cells},
        tables={"table": table},
    )


def test_smart_cs_tiebreaker_prefers_image_only_cs(data_visium_hd_like):
    """
    When multiple CSes contain the chosen image+shapes, prefer the one
    where the image is the only image. Mirrors Visium HD's _downscaled_lowres
    pattern.
    """
    plot = cl.spatial(
        data_visium_hd_like,
        key="score",
        image_name="lowres",
    )
    assert isinstance(plot, PlotSpec)


def test_smart_image_auto_resolves_from_coordinate_system(data_visium_hd_like):
    """When coordinate_system pins down a single image, image_name is auto."""
    plot = cl.spatial(
        data_visium_hd_like,
        key="score",
        coordinate_system="lowres_cs",
    )
    assert isinstance(plot, PlotSpec)


def test_smart_no_image_pin_still_ambiguous(data_visium_hd_like):
    """
    No image_name + no coordinate_system + multiple CSes each with one
    image → genuine ambiguity, must raise.
    """
    with pytest.raises(ValueError, match="coordinate system"):
        cl.spatial(data_visium_hd_like, key="score")


# ---- geom dispatch verification ----


def _layer_geoms(plot: PlotSpec) -> list[str]:
    return [layer.get("geom") for layer in plot.as_dict().get("layers", [])]


def test_polygon_true_emits_geom_polygon(data_polygons):
    plot = cl.spatial(data_polygons, key="cluster", polygon=True)
    assert "polygon" in _layer_geoms(plot)
    assert "point" not in _layer_geoms(plot)


def test_polygon_false_on_polygon_shapes_emits_geom_point(data_polygons):
    plot = cl.spatial(data_polygons, key="cluster")
    assert "point" in _layer_geoms(plot)
    assert "polygon" not in _layer_geoms(plot)


# ---- unsupported geometries ----


def test_multipolygon_raises():
    n = 2
    rng = np.random.default_rng(11)
    table = AnnData(
        X=rng.random((n, 2)).astype("float32"),
        obs=pd.DataFrame(
            {"instance_id": np.arange(n), "region": ["mp"] * n},
            index=[f"c{i}" for i in range(n)],
        ),
        var=pd.DataFrame(index=["G1", "G2"]),
    )
    table = TableModel.parse(table, region="mp", region_key="region", instance_key="instance_id")
    geoms = [
        MultiPolygon([Point(i, i).buffer(1.0), Point(i + 5, i + 5).buffer(1.0)]) for i in range(n)
    ]
    multi = ShapesModel.parse(gpd.GeoDataFrame(geometry=geoms, index=np.arange(n)))
    data = SpatialData(shapes={"mp": multi}, tables={"table": table})
    with pytest.raises(NotImplementedError, match="MultiPolygon"):
        cl.spatial(data, polygon=True)


# ---- spatials() with polygon=True ----


def test_spatials_polygon_true_returns_subplots(data_polygons):
    plot = cl.spatials(data_polygons, ["cluster", "G1"], polygon=True)
    assert isinstance(plot, SupPlotsSpec)
