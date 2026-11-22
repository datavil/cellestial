from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "ggrust.plot.core", "mudata"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl
from anndata import AnnData
from ggrust import (
    aes,
    coord_fixed,
    geom_imshow,
    geom_point,
    geom_polygon,
    ggplot,
    ggtb,
    scale_color_brewer,
    scale_fill_brewer,
    scale_y_reverse,
)
from ggrust.plot.core import FeatureSpec, PlotSpec
from mudata import MuData

from cellestial.frames import build_frame
from cellestial.layers.utilities import (
    _reject_constant_color_fill,
    _reject_constant_over_mapping,
)
from cellestial.single.core.dimensional import _KEY_COLOR_HINT
from cellestial.spatial.utilities import _reject_renamed_greyscale, _spatial_components
from cellestial.themes import _THEME_SPATIAL
from cellestial.util import (
    _collect_aes_columns,
    _color_gradient,
    _drop_nonfinite_rows,
    _fill_gradient,
    _filter_groups,
    _is_spatialdata,
    _reject_sequence_key,
    _resolve_tooltips,
    _validate_tooltips,
)
from cellestial.util.errors import (
    ConflictingLengthError,
    InvalidArgumentError,
    UnsupportedDataTypeError,
    _unsupported_data_type,
)

if TYPE_CHECKING:
    from typing import Literal

    from ggrust.plot.core import PlotSpec
    from numpy.typing import NDArray
    from polars import DataFrame
    from spatialdata import SpatialData

    _Components = tuple[NDArray | None, NDArray | None, DataFrame | None, AnnData]


def spatial(
    data: AnnData | SpatialData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    library_id: str | None = None,
    image: bool = True,
    image_key: Literal["hires", "lowres"] | str = "lowres",
    grayscale: bool = False,
    image_alpha: float | None = None,
    cmap: str | list | None = None,
    norm: bool | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    spatial_key: str = "spatial",
    table_name: str | None = None,
    image_name: str | None = None,
    shapes_name: str | None = None,
    coordinate_system: str | None = None,
    polygon: bool = False,
    crop: Sequence[int] | None = None,
    mapping: FeatureSpec | None = None,
    size: float | None = 1.5,
    alpha: float = 1.0,
    geom_color: str | None = None,
    geom_fill: str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    include_dimensions: bool | int = False,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#f6f6f6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    midpoint: Literal["mean", "median", "mid"] | float = "mid",
    **geom_kwargs,
) -> PlotSpec:
    """
    Spatial plot.

    Parameters
    ----------
    data : AnnData | SpatialData
        Spatial data to plot. AnnData inputs use Visium-style metadata stored
        on the object. SpatialData inputs are resolved via the `table_name`,
        `image_name`, `shapes_name`, and `coordinate_system` arguments below.
    key : str, default=None
        The key (cell feature or gene name) to color the spots by.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the `key` column.
    library_id : str | None, default=None
        The library identifier. If None and only one library is present, it is
        auto-selected; if multiple libraries are present, this must be provided.
        Leave as None when no library metadata is present (generic spatial data).
        Ignored for SpatialData inputs.
    image : bool, default=True
        Whether to render the tissue image as a background layer.
    image_key : str, default='lowres'
        Which image variant to render. Visium ships with 'hires' and 'lowres'.
        Falls back to 'hires', then to any available variant, when the requested
        one is missing. Ignored for SpatialData inputs.
    grayscale : bool, default=False
        Whether to convert an RGB(A) image to grayscale (Rec.709 luminance).
    image_alpha : float | None, default=None
        Alpha (transparency) of the tissue image.
        Distinct from `alpha`, which controls spot transparency.
    cmap : str | list | None, default=None
        Colormap name or list of colors. Grayscale images only.
    norm : bool | None, default=None
        Whether to linearly scale grayscale luminance values to [0, 255].
        Grayscale images only.
    vmin : float | None, default=None
        Lower bound for grayscale luminance normalization.
    vmax : float | None, default=None
        Upper bound for grayscale luminance normalization.
    spatial_key : str, default='spatial'
        The embedding key containing spot coordinates in fullres pixel space.
        Ignored for SpatialData inputs (coordinates come from the chosen
        spot geometries).
    table_name : str | None, default=None
        Selects which annotation table to use when multiple are present.
        If None and exactly one table is present, it is auto-selected.
        SpatialData inputs only.
    image_name : str | None, default=None
        Selects which background image to render when multiple are present.
        If None and exactly one image is present, it is auto-selected.
        SpatialData inputs only.
    shapes_name : str | None, default=None
        Selects which spot geometries to use when multiple are present.
        If None and exactly one geometry is present, it is auto-selected.
        SpatialData inputs only.
    coordinate_system : str | None, default=None
        Target coordinate system. Image and geometries are aligned into this
        system before plotting. If None, the single available system is used,
        falling back to 'global' when multiple are defined. SpatialData
        inputs only.
    polygon : bool, default=False
        Render polygon-shaped geometries as filled polygons. When False
        (default), polygon geometries are reduced to their centroids and
        rendered as points. SpatialData inputs only.
    crop : Sequence[int] | None, default=None
        Crop the plot to a region given as `(left, right, top, bottom)`.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings, the result of `aes()`.
    size : float | None, default=1.5
        Spot size.
    alpha : float, default=1.0
        Alpha (transparency) of the spots.
    groups : str | Sequence[str] | None, default=None
        Select specific groups to show.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out rows where `key` matches any of
        them. Categorical keys only.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    add_keys : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the frame,
        on top of those inferred from `key`, `mapping`, and `tooltips`.
    include_dimensions : bool | int
        Whether to include dimensions from embeddings in the DataFrame, default is False.
        Providing an integer will limit the number of dimensions to given number.
    tooltips : {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over a spot. None auto-generates from
        mapped keys; 'none' disables tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observation_id_column : str, default='Barcode'
        The name to give the barcode column in the DataFrame.
    color_low : str, default='#f6f6f6'
        Color for low values in the continuous gradient.
    color_mid : str | None, default=None
        Color for mid values in the continuous gradient. When None, the scale
        falls back to a 2-color gradient between color_low and color_high.
    color_high : str, default='#377eb8'
        Color for high values in the continuous gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='mid'
        Midpoint for the continuous color gradient.
    geom_color : str | None, default=None
        Color for all spots (border color for polygons). To color by a column, use `key`.
    geom_fill : str | None, default=None
        Fill color for all polygons (or filled point shapes 21 to 25).
    **geom_kwargs
        Additional parameters forwarded to `geom_point`.

    Returns
    -------
    PlotSpec
        Spatial plot of tissue spots over the H&E image.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported spatial data object.
    TypeError
        If `key` is a sequence of keys; use `spatials` for multiple keys.
    KeyNotFoundError
        If a requested table, image, shapes element, coordinate system, library
        or `spatial_key` is missing. The message suggests a close match.
    EmptySelectionError
        If the spatial data holds no table, shapes or image to choose from.
    InvalidArgumentError
        If spatial components are ambiguous, `crop` is not 4 values, `mapping`
        sets an aesthetic the plot maps itself, `color` or `fill` is passed
        as a constant (use `geom_color` / `geom_fill`), a `geom_color` /
        `geom_fill` would hide the coloring by `key` or `mapping`, or `polygon=True` meets a table
        without an instance key.
    ConflictingLengthError
        If the shapes cannot be matched to the table rows, or a given `frame`
        has a different number of rows than the table in polygon mode.
    DuplicateKeysError
        If the shapes element repeats an instance id.
    NotImplementedError
        If polygon rendering encounters unsupported MultiPolygon geometry.

    Notes
    -----
    If no tissue image metadata is present, the plot falls back to a plain
    spatial scatter using the raw coordinates.

    With several libraries, `library_id` picks the tissue image and its scale
    factor, but every spot in `data` is drawn on that image. Subset `data` to
    the spots of one library first.

    `polygon=True` draws holes in polygon geometries unfilled. The outline also
    draws a line from each polygon's exterior to its holes.

    `cmap`, `norm`, `vmin` and `vmax` are passed to `geom_imshow`.

    Examples
    --------
    An example interactive spatial plot with a categorical key.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.human_lymph_node()

        cl.spatial(data,key="clusters",interactive=True)


    With a gene name as the key.

    .. jupyter-execute::

        cl.spatial(data, key="MS4A1",interactive=True)

    Customize the plot

    .. jupyter-execute::

        (
            cl.spatial(data, key="MS4A1", size=2.4, interactive=True)
            + ggsize(600, 600)
            + scale_color_viridis()
        )

    With another dataset.

    .. jupyter-execute::

        import squidpy as sq

        data_hne = sq.datasets.visium_hne_adata()

        cl.spatial(data_hne,key="cluster",interactive=True)

    Make the background image grayscale.

    .. jupyter-execute::

        cl.spatial(data_hne,key="cluster",grayscale=True,interactive=True)

    Select specific groups to show.

    .. jupyter-execute::

        cl.spatial(data_hne,key="cluster", groups=["Hippocampus", "Hypothalamus_1", "Striatum"], interactive=True)
    """
    return _spatial(
        data,
        key,
        components=None,
        frame=frame,
        library_id=library_id,
        image=image,
        image_key=image_key,
        grayscale=grayscale,
        image_alpha=image_alpha,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        spatial_key=spatial_key,
        table_name=table_name,
        image_name=image_name,
        shapes_name=shapes_name,
        coordinate_system=coordinate_system,
        polygon=polygon,
        crop=crop,
        mapping=mapping,
        size=size,
        alpha=alpha,
        groups=groups,
        drop=drop,
        variable_keys=variable_keys,
        add_keys=add_keys,
        include_dimensions=include_dimensions,
        tooltips=tooltips,
        interactive=interactive,
        observation_id_column=observation_id_column,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        midpoint=midpoint,
        geom_color=geom_color,
        geom_fill=geom_fill,
        **geom_kwargs,
    )


def _spatial(
    data: AnnData | SpatialData,
    key: str | None = None,
    *,
    components: _Components | None,
    frame: DataFrame | None = None,
    library_id: str | None = None,
    image: bool = True,
    image_key: Literal["hires", "lowres"] | str = "lowres",
    grayscale: bool = False,
    image_alpha: float | None = None,
    cmap: str | list | None = None,
    norm: bool | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    spatial_key: str = "spatial",
    table_name: str | None = None,
    image_name: str | None = None,
    shapes_name: str | None = None,
    coordinate_system: str | None = None,
    polygon: bool = False,
    crop: Sequence[int] | None = None,
    mapping: FeatureSpec | None = None,
    size: float | None = 1.5,
    alpha: float = 1.0,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    include_dimensions: bool | int = False,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#f6f6f6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    midpoint: Literal["mean", "median", "mid"] | float = "mid",
    geom_color: str | None = None,
    geom_fill: str | None = None,
    **geom_kwargs,
) -> PlotSpec:
    """
    Build the spatial plot, reusing `components` when they are already resolved.

    Notes
    -----
    Parameters are those of `spatial`. `components` is the result of
    `_spatial_components` for the same arguments; None resolves them here.
    """
    # HANDLE: data type
    if isinstance(data, MuData):
        msg = (
            "Spatial plots do not accept a multimodal object, because spot "
            "coordinates and tissue images live in a modality rather than on the "
            "container.\nPass a single modality instead, e.g. `data['rna']`.\n"
            "To color by a variable from another modality, pass that modality "
            "together with a container frame:\n"
            "    spatial(data['rna'], key='prot:CD3', "
            "frame=build_frame(data, variable_keys=['prot:CD3']))"
        )
        raise UnsupportedDataTypeError(msg)

    if not isinstance(data, AnnData) and not _is_spatialdata(data):
        raise _unsupported_data_type(data, AnnData, "SpatialData")

    _reject_constant_color_fill(geom_kwargs, function="spatial", hint=_KEY_COLOR_HINT)
    _reject_renamed_greyscale(geom_kwargs)
    _reject_sequence_key(key, singular="spatial", plural="spatials")

    if components is None:
        components = _spatial_components(
            data,
            library_id=library_id,
            image_key=image_key,
            image=image,
            spatial_key=spatial_key,
            table_name=table_name,
            image_name=image_name,
            shapes_name=shapes_name,
            coordinate_system=coordinate_system,
            polygon=polygon,
        )
    image_array, spot_coordinates, polygon_frame, data = components
    is_polygon = polygon_frame is not None

    # HANDLE: mapping
    mapping = mapping or aes()
    _reject_mapping_conflicts(mapping, key=key, polygon=is_polygon)

    # HANDLE: variable_keys
    if variable_keys is None:
        variable_keys = []
    elif isinstance(variable_keys, str):
        variable_keys = [variable_keys]
    elif isinstance(variable_keys, Sequence):
        variable_keys = list(variable_keys)

    # Collect the frame columns from the color key, aes refs, and `add_keys`.
    if isinstance(add_keys, str):
        add_keys = [add_keys]
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[key, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=0,
    )
    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[observation_id_column, *([key] if key is not None else [])],
        metadata_columns=metadata_columns,
    )

    # BUILD: dataframe
    if frame is None:
        observation_column_name = None if tooltips == "none" else observation_id_column
        frame = build_frame(
            data=data,
            variable_keys=variable_keys,
            axis=0,
            observation_id_column=observation_column_name,
            include_dimensions=include_dimensions,
            metadata_columns=metadata_columns,
        )

    if is_polygon:
        # Long-format vertices joined to the table rows they annotate.
        if frame.height != data.n_obs:
            msg = (
                f"`frame` has {frame.height} rows but the spatial table has {data.n_obs}; "
                "polygons are matched to frame rows by position.\n"
                "Build `frame` from the same table without filtering its rows."
            )
            raise ConflictingLengthError(msg)
        frame = polygon_frame.join(
            frame.with_row_index("table_row"), on="table_row", how="inner", maintain_order="left"
        ).drop("table_row")
    else:
        frame = frame.with_columns(
            pl.Series("spatial_x", spot_coordinates[:, 0]),
            pl.Series("spatial_y", spot_coordinates[:, 1]),
        )

    # HANDLE: groups / drop filters (categorical-only)
    if key is not None:
        frame = _filter_groups(frame, column=key, groups=groups, drop=drop, column_parameter="key")

    if is_polygon:
        finite_vertex = pl.all_horizontal(
            pl.col("polygon_x").is_not_null(),
            pl.col("polygon_x").is_finite(),
            pl.col("polygon_y").is_not_null(),
            pl.col("polygon_y").is_finite(),
        )
        frame = frame.filter(finite_vertex.all().over("instance_id"))
    else:
        frame = _drop_nonfinite_rows(frame, ["spatial_x", "spatial_y"])

    _validate_tooltips(tooltips, frame)

    # BUILD: plot
    sptl = ggplot(data=frame)

    if image_array is not None:
        if grayscale and image_array.ndim == 3:
            # Rec.709 luminance; slice the first 3 channels so RGBA inputs work too.
            image_array = (image_array[..., :3] @ [0.2126, 0.7152, 0.0722]).astype(
                image_array.dtype,
            )
        sptl += geom_imshow(
            image_data=image_array,
            cmap=cmap,
            norm=norm,
            alpha=image_alpha,
            vmin=vmin,
            vmax=vmax,
            show_legend=False,
        )

    mapped_by = {
        aesthetic: f"`mapping` column {column!r}"
        for aesthetic, column in mapping.as_dict().items()
        if aesthetic in ("color", "fill")
    }
    if key is not None:
        mapped_by["fill" if is_polygon else "color"] = f"`key` {key!r}"
    _reject_constant_over_mapping(geom_color=geom_color, geom_fill=geom_fill, mapped_by=mapped_by)

    if "size" in mapping.as_dict():
        size = None

    if is_polygon:
        sptl += geom_polygon(
            mapping=aes(
                x="polygon_x",
                y="polygon_y",
                group="instance_id",
                **({} if key is None else {"fill": key}),
                **mapping.as_dict(),
            ),
            size=size,
            alpha=alpha,
            tooltips=tooltips,
            color=geom_color,
            fill=geom_fill,
            **geom_kwargs,
        )
        if key is not None:
            if isinstance(frame[key].dtype, (pl.Categorical, pl.Enum)):
                sptl += scale_fill_brewer(palette="Set2")
            elif frame[key].dtype.is_numeric():
                sptl += _fill_gradient(
                    frame[key],
                    color_low=color_low,
                    color_mid=color_mid,
                    color_high=color_high,
                    midpoint=midpoint,
                )
    else:
        sptl += geom_point(
            mapping=aes(
                x="spatial_x",
                y="spatial_y",
                **({} if key is None else {"color": key}),
                **mapping.as_dict(),
            ),
            size=size,
            alpha=alpha,
            tooltips=tooltips,
            color=geom_color,
            fill=geom_fill,
            **geom_kwargs,
        )
        if key is not None:
            if isinstance(frame[key].dtype, (pl.Categorical, pl.Enum)):
                sptl += scale_color_brewer(palette="Set2")
            elif frame[key].dtype.is_numeric():
                sptl += _color_gradient(
                    frame[key],
                    color_low=color_low,
                    color_mid=color_mid,
                    color_high=color_high,
                    midpoint=midpoint,
                )

    # Image rows grow downward in pixel space; reverse y so spots align with the image.
    if crop is not None:
        if len(crop) != 4:
            msg = (
                f"`crop` must be a sequence of 4 integers (left, right, top, bottom), "
                f"got {len(crop)} values."
            )
            raise InvalidArgumentError(msg)
        left, right, top, bottom = crop
        coordinates = coord_fixed(xlim=[left, right], ylim=[top, bottom])
    else:
        coordinates = coord_fixed()
    sptl += scale_y_reverse() + coordinates + _THEME_SPATIAL

    if interactive:
        sptl += ggtb(size_zoomin=-1)

    return sptl


def _reject_mapping_conflicts(mapping: FeatureSpec, *, key: str | None, polygon: bool) -> None:
    """Raise when `mapping` sets an aesthetic that `spatial` maps itself."""
    owned = {"x": "the spot coordinates", "y": "the spot coordinates"}
    if polygon:
        owned["group"] = "the polygon outlines"
    if key is not None:
        owned["fill" if polygon else "color"] = f"`key` ('{key}')"
    for aesthetic in mapping.as_dict():
        if aesthetic in owned:
            msg = (
                f"`mapping` sets `{aesthetic}`, which `spatial` already maps to "
                f"{owned[aesthetic]}.\nRemove `{aesthetic}` from `mapping`"
                + (
                    "; to map it to another column, pass `key=None` and set it in `mapping`."
                    if aesthetic in ("color", "fill")
                    else "."
                )
            )
            raise InvalidArgumentError(msg)
