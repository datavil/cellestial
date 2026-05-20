from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import shapely
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation
from xarray import DataArray

from cellestial.util import _warn
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _select_one(name: str | None, available: list[str], kind: str) -> str:
    """Return `name` if given, else the single available entry, else raise."""
    if name is not None:
        if name not in available:
            msg = f"{kind} `{name}` not found. Available: {available}"
            raise KeyError(msg)
        return name
    if len(available) == 1:
        return available[0]
    if len(available) == 0:
        msg = f"No {kind} elements found in the SpatialData object."
        raise ValueError(msg)
    msg = (
        f"Multiple {kind} elements found: {available}. "
        f"Pass the corresponding `*_name` argument to select one."
    )
    raise ValueError(msg)


def _element_in_coordinate_system(elem, coordinate_system: str) -> bool:
    """True when `elem` has a transformation registered for `coordinate_system`."""
    return coordinate_system in get_transformation(elem, get_all=True)


def _resolve_shapes_name(data: SpatialData, shapes_name: str | None, table: AnnData) -> str:
    """Pick a shapes element, falling back to the table's annotated region."""
    if shapes_name is not None:
        if shapes_name not in data.shapes:
            msg = f"shapes `{shapes_name}` not found. Available: {list(data.shapes)}"
            raise KeyError(msg)
        return shapes_name

    region = table.uns.get("spatialdata_attrs", {}).get("region")
    candidates = [region] if isinstance(region, str) else list(region or [])
    candidates = [r for r in candidates if r in data.shapes]
    if len(candidates) == 1:
        return candidates[0]

    return _select_one(None, list(data.shapes), "shapes")


def _resolve_coordinate_system(
    data: SpatialData,
    coordinate_system: str | None,
    *,
    shapes_name: str,
    image_name: str | None,
    want_image: bool,
) -> str:
    """Pick a coordinate system that contains the chosen shapes (and image)."""
    systems = list(data.coordinate_systems)

    if coordinate_system is not None:
        if coordinate_system not in systems:
            msg = f"coordinate_system `{coordinate_system}` not found. Available: {systems}"
            raise KeyError(msg)
        return coordinate_system

    if len(systems) == 1:
        return systems[0]
    if "global" in systems:
        return "global"

    shapes_elem = data.shapes[shapes_name]

    def _has_image(cs: str) -> bool:
        if not want_image:
            return True
        if image_name is not None:
            return _element_in_coordinate_system(data.images[image_name], cs)
        return any(_element_in_coordinate_system(img, cs) for img in data.images.values())

    candidates = [
        cs for cs in systems if _element_in_coordinate_system(shapes_elem, cs) and _has_image(cs)
    ]
    if len(candidates) == 1:
        return candidates[0]

    # Tiebreaker: when an image is pinned, prefer the CS that holds only that
    # image (i.e., the most specific one). Catches Visium HD-style layouts
    # with parallel `_downscaled_<resolution>` coordinate systems.
    if want_image and image_name is not None and len(candidates) > 1:
        specialised = [
            cs
            for cs in candidates
            if [
                name for name, img in data.images.items() if _element_in_coordinate_system(img, cs)
            ]
            == [image_name]
        ]
        if len(specialised) == 1:
            return specialised[0]

    msg = (
        f"Multiple coordinate systems present: {systems}. Pass `coordinate_system=` to select one."
    )
    raise ValueError(msg)


def _resolve_image_name(
    data: SpatialData, image_name: str | None, *, coordinate_system: str
) -> str | None:
    """Pick the image element that lives in the chosen coordinate system."""
    if image_name is not None:
        if image_name not in data.images:
            msg = f"image `{image_name}` not found. Available: {list(data.images)}"
            raise KeyError(msg)
        return image_name

    in_cs = [
        name
        for name, img in data.images.items()
        if _element_in_coordinate_system(img, coordinate_system)
    ]
    if len(in_cs) == 1:
        return in_cs[0]
    if len(in_cs) == 0:
        return None
    msg = (
        f"Multiple image elements in coordinate system `{coordinate_system}`: "
        f"{in_cs}. Pass `image_name=` to select one."
    )
    raise ValueError(msg)


def _resolve_table(data: SpatialData, table_name: str | None) -> AnnData:
    """Pick the annotation table from a SpatialData object."""
    if not isinstance(data, SpatialData):
        msg = f"Expected a SpatialData object, got {type(data)}."
        raise TypeError(msg)
    name = _select_one(table_name, list(data.tables.keys()), "table")
    return data.tables[name]


def _image_to_yxc(elem) -> NDArray:
    """
    Convert a SpatialData image element to a (y, x[, c]) NumPy array.

    Accepts either an `xarray.DataArray` or a multiscale `DataTree`.
    """
    if isinstance(elem, DataArray):
        array = np.asarray(elem.values)
        dims = elem.dims
    else:
        head_name = next(iter(elem.children.keys()))
        head = elem[head_name]
        data_var = next(iter(head.data_vars))
        array = np.asarray(head[data_var].values)
        dims = head[data_var].dims

    if "c" in dims:
        c_axis = dims.index("c")
        array = np.moveaxis(array, c_axis, -1)
        if array.shape[-1] == 1:
            array = array[..., 0]
    return array


def _polygon_vertex_frame(shapes_geo) -> pl.DataFrame:
    """
    Long-format vertex frame for `geom_polygon`.

    Columns: `instance_id`, `polygon_x`, `polygon_y`.
    """
    geoms = shapes_geo.geometry.values
    coords = shapely.get_coordinates(geoms)
    per_geom_counts = np.fromiter(
        (shapely.count_coordinates(g) for g in geoms),
        dtype=np.int64,
        count=len(geoms),
    )
    instance_ids = np.repeat(np.asarray(shapes_geo.index), per_geom_counts)
    return pl.DataFrame(
        {
            "instance_id": instance_ids,
            "polygon_x": coords[:, 0],
            "polygon_y": coords[:, 1],
        }
    )


def _spatialdata_components(
    data: SpatialData,
    *,
    table_name: str | None,
    image_name: str | None,
    shapes_name: str | None,
    coordinate_system: str | None,
    image: bool,
    polygon: bool = False,
) -> tuple[NDArray | None, NDArray | None, pl.DataFrame | None, AnnData]:
    """
    Resolve image, geometry, and the annotation table.

    Returns
    -------
    image_array : NDArray | None
        The chosen image transformed into the target coordinate system,
        in `(y, x[, c])` order, or None when `image=False` or no image
        element exists.
    spot_coordinates : NDArray | None
        Shape `(n, 2)` array of `(x, y)` per spot. None when polygon
        rendering is requested.
    polygon_frame : polars.DataFrame | None
        Long-format vertex frame, populated only when `polygon=True` and
        the chosen shapes element holds Polygons.
    table : AnnData
        The chosen annotation table.

    Notes
    -----
    When the chosen shapes element holds Polygons, `polygon=True` returns
    a long-format vertex frame for `geom_polygon`. `polygon=False`
    (the default) reduces them to centroids and returns point coordinates
    instead.
    """
    if not isinstance(data, SpatialData):
        msg = f"Expected a SpatialData object, got {type(data)}."
        raise TypeError(msg)

    table = _resolve_table(data, table_name)
    chosen_shapes_name = _resolve_shapes_name(data, shapes_name, table)
    target_cs = _resolve_coordinate_system(
        data,
        coordinate_system,
        shapes_name=chosen_shapes_name,
        image_name=image_name,
        want_image=image,
    )
    chosen_image_name = (
        _resolve_image_name(data, image_name, coordinate_system=target_cs) if image else None
    )

    # spatialdata's transform mutates the underlying GeoDataFrame via chained
    # assignment, which trips a geopandas FutureWarning we don't control.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*ChainedAssignmentError.*",
            category=FutureWarning,
            module=r"geopandas\..*",
        )
        shapes_geo = data.transform_element_to_coordinate_system(chosen_shapes_name, target_cs)

    geom_types = set(shapes_geo.geometry.geom_type.unique())
    spot_coordinates: NDArray | None = None
    polygon_frame: pl.DataFrame | None = None
    if geom_types.issubset({"Point"}):
        spot_coordinates = np.column_stack(
            [shapes_geo.geometry.x.to_numpy(), shapes_geo.geometry.y.to_numpy()]
        )
    elif geom_types.issubset({"Polygon"}):
        if polygon:
            polygon_frame = _polygon_vertex_frame(shapes_geo)
        else:
            centroids = shapes_geo.geometry.centroid
            spot_coordinates = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    else:
        msg = (
            f"Shapes element `{chosen_shapes_name}` contains "
            f"{sorted(geom_types)}; only Point or Polygon are supported."
        )
        raise NotImplementedError(msg)

    image_array = None
    if image and chosen_image_name is not None:
        image_elem = data.transform_element_to_coordinate_system(chosen_image_name, target_cs)
        image_array = _image_to_yxc(image_elem)

    return image_array, spot_coordinates, polygon_frame, table


def _spatial_components(
    data,
    *,
    library_id: str | None,
    image_key: str,
    image: bool,
    spatial_key: str,
    table_name: str | None = None,
    image_name: str | None = None,
    shapes_name: str | None = None,
    coordinate_system: str | None = None,
    polygon: bool = False,
) -> tuple[NDArray | None, NDArray | None, pl.DataFrame | None, AnnData]:
    """
    Extract image, geometry, and the annotation table.

    Returns `(image, point_coords, polygon_frame, table)`. Exactly one of
    `point_coords` and `polygon_frame` is non-None. For AnnData input
    the table is the input itself; for SpatialData it is the resolved table.
    """
    if isinstance(data, SpatialData):
        return _spatialdata_components(
            data,
            table_name=table_name,
            image_name=image_name,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            image=image,
            polygon=polygon,
        )

    if isinstance(data, AnnData):
        if spatial_key not in data.obsm:
            msg = f"spot coordinates not found under `{spatial_key}`"
            raise KeyError(msg)

        spatial_uns = data.uns.get("spatial", {})

        # Generic mode: no tissue image / scalefactor metadata available.
        if not spatial_uns:
            if library_id is not None:
                msg = (
                    f"library_id `{library_id}` was provided but no spatial "
                    "library metadata is present"
                )
                raise KeyError(msg)
            return None, data.obsm[spatial_key], None, data

        # Visium mode.
        if library_id is None:
            if len(spatial_uns) == 1:
                library_id = next(iter(spatial_uns))
            else:
                msg = (
                    f"Multiple spatial libraries found: {list(spatial_uns)}. "
                    "Pass `library_id=` to select one."
                )
                raise ValueError(msg)
        elif library_id not in spatial_uns:
            msg = f"library_id `{library_id}` not found in spatial libraries"
            raise KeyError(msg)

        library = spatial_uns[library_id]
        scale_factors = library.get("scalefactors", {})
        scalef_key = f"tissue_{image_key}_scalef"
        scale_factor = scale_factors.get(scalef_key)
        if scale_factor is None:
            if image:
                _warn(f"Scale factor `{scalef_key}` missing; spots may not align with the image.")
            scale_factor = 1.0
        spot_coordinates = data.obsm[spatial_key] * scale_factor

        image_array = None
        if image:
            images = library.get("images", {})
            image_array = images.get(image_key)
            if image_array is None:
                msg = f"image `{image_key}` not found for library `{library_id}`"
                _warn(msg)
        return image_array, spot_coordinates, None, data

    msg = f"Unsupported data type: `{type(data)}`"
    raise UnsupportedDataTypeError(msg)
