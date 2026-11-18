"""Regression tests for the spatial follow-up fixes of the 2026-11-18 audit."""

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from shapely.geometry import Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel, TableModel

import cellestial as cl
from cellestial.spatial import utilities
from cellestial.util.errors import (
    DuplicateKeysError,
    EmptySelectionError,
    InvalidArgumentError,
    KeyNotFoundError,
)


def _frame(plot) -> pl.DataFrame:
    return plot.as_dict()["data"]


def _table(obs: dict, n: int, regions: str | list[str]) -> AnnData:
    table = AnnData(
        X=np.ones((n, 1), dtype="float32"),
        obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=["G"]),
    )
    return TableModel.parse(table, region=regions, region_key="region", instance_key="cell_id")


# ---- 1. polygon holes ----


def _signed_area(x: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def _rings(x: list[float], y: list[float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a vertex list into rings, each ending on its first vertex."""
    rings, start = [], 0
    for index in range(len(x)):
        if index > start and (x[index], y[index]) == (x[start], y[start]):
            rings.append((np.array(x[start : index + 1]), np.array(y[start : index + 1])))
            start = index + 1
    return rings


@pytest.fixture
def data_hole() -> SpatialData:
    """One square with a square hole wound the same way as the exterior, and a plain square."""
    shell = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(3, 3), (7, 3), (7, 7), (3, 7)]  # counter-clockwise, like the shell
    geometry = [Polygon(shell, [hole]), Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])]
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=geometry, index=[0, 1]))
    table = _table(
        {"label": pd.Categorical(["ring", "plain"]), "cell_id": [0, 1], "region": "cells"},
        2,
        "cells",
    )
    return SpatialData(shapes={"cells": shapes}, tables={"table": table})


def test_polygon_hole_winds_against_exterior(data_hole):
    frame = _frame(cl.spatial(data_hole, "label", polygon=True)).filter(pl.col("label") == "ring")
    rings = _rings(frame["polygon_x"].to_list(), frame["polygon_y"].to_list())
    assert len(rings) == 2
    exterior, hole = (_signed_area(x, y) for x, y in rings)
    assert exterior > 0 > hole


# ---- 2. instance key column kept in polygon mode ----


def test_polygon_keeps_instance_key_column(data_hole):
    plot = cl.spatial(data_hole, "label", polygon=True, add_keys="cell_id", tooltips=["cell_id"])
    frame = _frame(plot)
    assert "cell_id" in frame.columns
    assert frame.group_by("label").agg(pl.col("cell_id").first()).sort("label").rows() == [
        ("plain", 1),
        ("ring", 0),
    ]


# ---- 3. polygon mode filters by region ----


@pytest.fixture
def data_two_regions() -> SpatialData:
    """Two shapes elements with the same instance ids, one table annotating both."""

    def square(x: float) -> Polygon:
        return Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)])

    cells = ShapesModel.parse(gpd.GeoDataFrame(geometry=[square(0), square(5)], index=[0, 1]))
    nuclei = ShapesModel.parse(gpd.GeoDataFrame(geometry=[square(50), square(55)], index=[0, 1]))
    table = _table(
        {
            "label": pd.Categorical(["cell0", "cell1", "nucleus0", "nucleus1"]),
            "cell_id": [0, 1, 0, 1],
            "region": pd.Categorical(["cells", "cells", "nuclei", "nuclei"]),
        },
        4,
        ["cells", "nuclei"],
    )
    return SpatialData(shapes={"cells": cells, "nuclei": nuclei}, tables={"table": table})


@pytest.mark.parametrize(
    ("shapes_name", "expected"),
    [("cells", {"cell0": 0.0, "cell1": 5.0}), ("nuclei", {"nucleus0": 50.0, "nucleus1": 55.0})],
)
def test_polygon_joins_only_rows_of_the_chosen_region(data_two_regions, shapes_name, expected):
    plot = cl.spatial(data_two_regions, "label", shapes_name=shapes_name, polygon=True)
    frame = _frame(plot).group_by("label").agg(pl.col("polygon_x").min(), pl.len())
    assert dict(frame.select("label", "polygon_x").rows()) == expected
    assert frame["len"].to_list() == [5, 5]  # one closed square each, no duplicates


def test_polygon_frame_length_mismatch_raises(data_hole):
    frame = cl.build_frame(data_hole, axis=0).head(1)
    with pytest.raises(cl.util.errors.ConflictingLengthError, match="rows"):
        cl.spatial(data_hole, "label", polygon=True, frame=frame)


# ---- 5. mapping that sets the aesthetic the function sets itself ----


def _aes(**kwargs):
    from ggrust import aes

    return aes(**kwargs)


@pytest.mark.parametrize(
    ("polygon", "aesthetic"), [(False, "color"), (False, "x"), (True, "fill"), (True, "group")]
)
def test_mapping_conflict_raises(data_hole, polygon, aesthetic):
    with pytest.raises(InvalidArgumentError, match=f"`{aesthetic}`"):
        cl.spatial(data_hole, "label", polygon=polygon, mapping=_aes(**{aesthetic: "label"}))


def test_mapping_conflict_raises_in_spatials(data_hole):
    with pytest.raises(InvalidArgumentError, match="`color`"):
        cl.spatials(data_hole, ["label"], mapping=_aes(color="label"))


def test_mapping_color_without_key_is_used(data_hole):
    plot = cl.spatial(data_hole, mapping=_aes(color="label"))
    layer = next(layer for layer in plot.as_dict()["layers"] if layer["geom"] == "point")
    assert layer["mapping"]["color"] == "label"


# ---- 6. selection helpers raise errors.py classes with hints ----


def test_select_one_unknown_key_hints_close_match():
    with pytest.raises(KeyNotFoundError, match="did you mean 'cells'"):
        utilities._select_one("cels", ["cells", "nuclei"], "shapes")


def test_select_one_empty_and_ambiguous():
    with pytest.raises(EmptySelectionError, match="No image"):
        utilities._select_one(None, [], "image")
    with pytest.raises(InvalidArgumentError, match="Multiple image"):
        utilities._select_one(None, ["a", "b"], "image")


def test_spatialdata_unknown_names_hint(data_two_regions):
    with pytest.raises(KeyNotFoundError, match="did you mean 'nuclei'"):
        cl.spatial(data_two_regions, shapes_name="nucleus")
    with pytest.raises(KeyNotFoundError, match="did you mean 'table'"):
        cl.spatial(data_two_regions, table_name="tabel")
    with pytest.raises(KeyNotFoundError, match="did you mean 'global'"):
        cl.spatial(data_two_regions, shapes_name="cells", coordinate_system="globl")
    with pytest.raises(InvalidArgumentError, match="shapes_name"):
        cl.spatial(data_two_regions)


def _visium(libraries: list[str]) -> AnnData:
    data = AnnData(
        X=np.ones((2, 1), dtype="float32"),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["G"]),
    )
    data.obsm["spatial"] = np.ones((2, 2))
    data.uns["spatial"] = {name: {"images": {}, "scalefactors": {}} for name in libraries}
    return data


def test_anndata_library_errors_hint():
    data = _visium(["V1_Lymph_Node", "V1_Kidney"])
    with pytest.raises(InvalidArgumentError, match="library_id"):
        cl.spatial(data, image=False)
    with pytest.raises(KeyNotFoundError, match="did you mean 'V1_Kidney'"):
        cl.spatial(data, library_id="V1_Kidny", image=False)
    with pytest.raises(KeyNotFoundError, match="did you mean 'spatial'"):
        cl.spatial(data, library_id="V1_Kidney", spatial_key="spatail", image=False)


# ---- 7. spatials resolves the components once ----


def test_spatials_resolves_components_once(data_hole, monkeypatch):
    calls = []
    original = utilities._spatial_components

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    for module in ("cellestial.spatial.spatials", "cellestial.spatial.spatial"):
        monkeypatch.setattr(sys.modules[module], "_spatial_components", counting)
    grid = cl.spatials(data_hole, ["label", "G"], polygon=True)
    assert len(calls) == 1
    first = cl.spatial(data_hole, "label", polygon=True)
    first_frame = first.as_dict()["data"]
    assert grid.as_dict()["figures"][0]["data"].select(first_frame.columns).equals(first_frame)


@pytest.mark.parametrize("polygon", [False, True])
def test_spatials_subplots_match_spatial(data_hole, polygon):
    grid = cl.spatials(data_hole, ["label", "G"], polygon=polygon)
    for figure, key in zip(grid.as_dict()["figures"], ["label", "G"], strict=True):
        single = cl.spatial(data_hole, key, polygon=polygon).as_dict()
        assert figure["data"].select(single["data"].columns).equals(single["data"])
        assert figure["layers"] == single["layers"]


# ---- duplicate shape ids ----


@pytest.mark.parametrize("polygon", [False, True])
def test_duplicate_shape_ids_raise(polygon):
    squares = [Polygon([(x, 0), (x + 1, 0), (x + 1, 1), (x, 1)]) for x in (0, 5, 10)]
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=squares, index=[0, 0, 1]))
    table = _table(
        {"label": pd.Categorical(["a", "b"]), "cell_id": [1, 0], "region": "cells"}, 2, "cells"
    )
    data = SpatialData(shapes={"cells": shapes}, tables={"table": table})
    with pytest.raises(DuplicateKeysError, match=r"cells.*\[0\]"):
        cl.spatial(data, "label", polygon=polygon)
