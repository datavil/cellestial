from __future__ import annotations

__lazy_modules__ = ["anndata", "numpy"]

import warnings
from difflib import get_close_matches
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from anndata import AnnData

from cellestial.util import _is_spatialdata, _warn
from cellestial.util.errors import (
    ConflictingLengthError,
    DuplicateKeysError,
    EmptySelectionError,
    InvalidArgumentError,
    KeyNotFoundError,
    _unsupported_data_type,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from spatialdata import SpatialData


def _not_found(kind: str, name: str, available: list[str], parameter: str) -> KeyNotFoundError:
    """Build a `KeyNotFoundError` for an unknown `name`, with a close match when one exists."""
    matches = get_close_matches(name, available, n=1)
    hint = f"; did you mean '{matches[0]}'?" if matches else "."
    return KeyNotFoundError(
        f"{kind} `{name}` not found{hint} Available: {available}.\n"
        f"Pass one of them as `{parameter}=`."
    )


def _select_one(name: str | None, available: list[str], kind: str) -> str:
    """Return `name` if given, else the single available entry, else raise."""
    if name is not None:
        if name not in available:
            error = _not_found(kind, name, available, f"{kind}_name")
            raise error
        return name
    if len(available) == 1:
        return available[0]
    if len(available) == 0:
        msg = f"No {kind} elements found in the SpatialData object."
        raise EmptySelectionError(msg)
    msg = f"Multiple {kind} elements found: {available}. Pass `{kind}_name=` to select one."
    raise InvalidArgumentError(msg)


def _element_in_coordinate_system(element, coordinate_system: str) -> bool:
    """True when `element` has a transformation registered for `coordinate_system`."""
    from spatialdata.transformations import get_transformation

    return coordinate_system in get_transformation(element, get_all=True)


def _resolve_shapes_name(data: SpatialData, shapes_name: str | None, table: AnnData) -> str:
    """Pick a shapes element, falling back to the table's annotated region."""
    if shapes_name is not None:
        if shapes_name not in data.shapes:
            error = _not_found("shapes", shapes_name, list(data.shapes), "shapes_name")
            raise error
        return shapes_name

    region = table.uns.get("spatialdata_attrs", {}).get("region")
    candidates = [region] if isinstance(region, str) else list(region or [])
    candidates = [r for r in candidates if r in data.shapes]
    if len(candidates) == 1:
        return candidates[0]

    return _select_one(None, list(data.shapes), "shapes")


def _resolve_instance_key(table: AnnData) -> str | None:
    """Return the metadata column joining a table to its shapes element, if present."""
    if isinstance(table, AnnData):
        instance_key = table.uns.get("spatialdata_attrs", {}).get("instance_key")
        if instance_key is not None and instance_key in table.obs.columns:
            return instance_key
    return None


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
            error = _not_found(
                "coordinate_system", coordinate_system, systems, "coordinate_system"
            )
            raise error
        return coordinate_system

    if len(systems) == 1:
        return systems[0]
    if "global" in systems:
        return "global"

    shapes_element = data.shapes[shapes_name]

    def _has_image(cs: str) -> bool:
        if not want_image:
            return True
        if image_name is not None:
            return _element_in_coordinate_system(data.images[image_name], cs)
        return any(_element_in_coordinate_system(img, cs) for img in data.images.values())

    candidates = [
        cs
        for cs in systems
        if _element_in_coordinate_system(shapes_element, cs) and _has_image(cs)
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
    raise InvalidArgumentError(msg)


def _resolve_image_name(
    data: SpatialData, image_name: str | None, *, coordinate_system: str
) -> str | None:
    """Pick the image element that lives in the chosen coordinate system."""
    if image_name is not None:
        if image_name not in data.images:
            error = _not_found("image", image_name, list(data.images), "image_name")
            raise error
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
    raise InvalidArgumentError(msg)


def _resolve_table(data: SpatialData, table_name: str | None) -> AnnData:
    """Pick the annotation table from a SpatialData object."""
    from spatialdata import SpatialData

    if not isinstance(data, SpatialData):
        raise _unsupported_data_type(data, "SpatialData")
    name = _select_one(table_name, list(data.tables.keys()), "table")
    return data.tables[name]


def _image_to_yxc(element) -> NDArray:
    """
    Convert a SpatialData image element to a (y, x[, c]) NumPy array.

    Notes
    -----
    Accepts either an `xarray.DataArray` or a multiscale `DataTree`.
    """
    from xarray import DataArray

    if isinstance(element, DataArray):
        array = np.asarray(element.values)
        dimensions = element.dims
    else:
        head_name = next(iter(element.children.keys()))
        head = element[head_name]
        data_variable = next(iter(head.data_vars))
        array = np.asarray(head[data_variable].values)
        dimensions = head[data_variable].dims

    if "c" in dimensions:
        channel_axis = dimensions.index("c")
        array = np.moveaxis(array, channel_axis, -1)
        if array.shape[-1] == 1:
            array = array[..., 0]
    return array


def _polygon_vertex_frame(shapes_geo, table_positions: NDArray) -> pl.DataFrame:
    """
    Long-format vertex frame for `geom_polygon`, one polygon per matched table row.

    Notes
    -----
    Columns: `table_row` (position of the table row the polygon annotates),
    `instance_id`, `polygon_x`, `polygon_y`. `table_positions` holds, per table
    row, the position of its shape (-1 when it has none here).

    Each polygon lists its exterior ring, then each hole, every ring ending on
    its first vertex. Rings are oriented so the exterior winds counter-clockwise
    and holes clockwise, which leaves holes unfilled under the nonzero rule.
    """
    import shapely

    geoms = shapely.orient_polygons(shapes_geo.geometry.values)
    coordinates = shapely.get_coordinates(geoms)
    per_geom_counts = np.fromiter(
        (shapely.count_coordinates(g) for g in geoms),
        dtype=np.int64,
        count=len(geoms),
    )
    vertices = pl.DataFrame(
        {
            "shape_position": np.repeat(np.arange(len(geoms)), per_geom_counts),
            "instance_id": np.repeat(np.asarray(shapes_geo.index), per_geom_counts),
            "polygon_x": coordinates[:, 0],
            "polygon_y": coordinates[:, 1],
        }
    )
    found = np.flatnonzero(table_positions >= 0)
    rows = pl.DataFrame({"table_row": found, "shape_position": table_positions[found]})
    return rows.join(vertices, on="shape_position", how="inner", maintain_order="left").drop(
        "shape_position"
    )


def _table_shape_positions(
    table: AnnData, shapes_index, shapes_name: str
) -> NDArray[np.intp] | None:
    """
    Position of each table row's shape in `shapes_index`, -1 when it has none.

    Notes
    -----
    Rows are matched on the table's instance key and limited to rows whose
    region is `shapes_name`. None when the table has no instance key. Raises
    `DuplicateKeysError` when the shapes repeat an instance id.
    """
    instance_key = _resolve_instance_key(table)
    if instance_key is None:
        return None
    if not shapes_index.is_unique:
        duplicated = sorted(set(shapes_index[shapes_index.duplicated()].tolist()))
        msg = (
            f"Shapes element `{shapes_name}` repeats the instance ids {duplicated[:10]}, "
            "so table rows cannot be matched to one shape each.\n"
            "Give every shape a unique id before plotting."
        )
        raise DuplicateKeysError(msg)
    positions = shapes_index.get_indexer(table.obs[instance_key])
    region_key = table.uns.get("spatialdata_attrs", {}).get("region_key")
    if region_key in table.obs.columns:
        positions[np.asarray(table.obs[region_key] != shapes_name)] = -1
    return positions


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
    from spatialdata import SpatialData

    if not isinstance(data, SpatialData):
        raise _unsupported_data_type(data, "SpatialData")

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

    table_positions = _table_shape_positions(table, shapes_geo.index, chosen_shapes_name)
    geom_types = set(shapes_geo.geometry.geom_type.unique())
    spot_coordinates: NDArray | None = None
    polygon_frame: pl.DataFrame | None = None
    if geom_types.issubset({"Point"}):
        spot_coordinates = np.column_stack(
            [shapes_geo.geometry.x.to_numpy(), shapes_geo.geometry.y.to_numpy()]
        )
    elif geom_types.issubset({"Polygon"}):
        if polygon:
            if table_positions is None:
                msg = (
                    "`polygon=True` needs the table to name the shape each row annotates, "
                    "but the table has no instance key column.\n"
                    "Parse the table with `TableModel.parse(..., instance_key=...)`, "
                    "or pass `polygon=False` to draw the polygon centroids."
                )
                raise InvalidArgumentError(msg)
            polygon_frame = _polygon_vertex_frame(shapes_geo, table_positions)
        else:
            centroids = shapes_geo.geometry.centroid
            spot_coordinates = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    else:
        msg = (
            f"Shapes element `{chosen_shapes_name}` contains "
            f"{sorted(geom_types)}; only Point or Polygon are supported."
        )
        raise NotImplementedError(msg)

    if spot_coordinates is not None:
        # Shapes and table rows need not share order or length: place each table row
        # at its own instance's coordinates (NaN when it annotates no shape here).
        if table_positions is not None:
            aligned = np.full((table.n_obs, 2), np.nan)
            found = table_positions >= 0
            aligned[found] = spot_coordinates[table_positions[found]]
            spot_coordinates = aligned
        elif len(spot_coordinates) != table.n_obs:
            msg = (
                f"Shapes element `{chosen_shapes_name}` has {len(spot_coordinates)} shapes "
                f"but the table has {table.n_obs} rows, and the table has no instance key "
                "to match them.\nParse the table with `TableModel.parse(..., instance_key=...)`."
            )
            raise ConflictingLengthError(msg)

    image_array = None
    if image and chosen_image_name is not None:
        image_element = data.transform_element_to_coordinate_system(chosen_image_name, target_cs)
        image_array = _image_to_yxc(image_element)

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

    Notes
    -----
    Returns `(image, point_coords, polygon_frame, table)`. Exactly one of
    `point_coords` and `polygon_frame` is non-None. For AnnData input
    the table is the input itself; for SpatialData it is the resolved table.
    """
    if _is_spatialdata(data):
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
            error = _not_found(
                "spot coordinates", spatial_key, list(data.obsm.keys()), "spatial_key"
            )
            raise error

        spatial_uns = data.uns.get("spatial", {})

        # Generic mode: no tissue image / scalefactor metadata available.
        if not spatial_uns:
            if library_id is not None:
                msg = (
                    f"library_id `{library_id}` was provided but no spatial "
                    "library metadata is present.\n"
                    "Pass `library_id=None` to plot the spot coordinates without an image."
                )
                raise KeyNotFoundError(msg)
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
                raise InvalidArgumentError(msg)
        elif library_id not in spatial_uns:
            error = _not_found("library_id", library_id, list(spatial_uns), "library_id")
            raise error

        library = spatial_uns[library_id]
        images = library.get("images", {})

        # Resolve which image variant to render. When the requested one is
        # absent, fall back to `hires`, then to whatever else is available, so
        # the image and the scale factor below stay matched to one resolution.
        resolved_image_key = image_key
        if image and image_key not in images:
            if "hires" in images:
                resolved_image_key = "hires"
            elif images:
                resolved_image_key = next(iter(images))
            if resolved_image_key in images:
                _warn(
                    f"image `{image_key}` not found for library `{library_id}`; "
                    f"using `{resolved_image_key}` instead"
                )

        scale_factors = library.get("scalefactors", {})
        scalef_key = f"tissue_{resolved_image_key}_scalef"
        scale_factor = scale_factors.get(scalef_key)
        if scale_factor is None:
            if image:
                _warn(f"Scale factor `{scalef_key}` missing; spots may not align with the image.")
            scale_factor = 1.0
        spot_coordinates = data.obsm[spatial_key] * scale_factor

        image_array = None
        if image:
            image_array = images.get(resolved_image_key)
            if image_array is None:
                _warn(f"no images found for library `{library_id}`")
        return image_array, spot_coordinates, None, data

    raise _unsupported_data_type(data, AnnData, "SpatialData")
