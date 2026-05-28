from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl
from anndata import AnnData
from lets_plot import (
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
from lets_plot.plot.core import FeatureSpec, PlotSpec
from spatialdata import SpatialData

from cellestial.frames import build_frame
from cellestial.spatial.utilities import _resolve_instance_key, _spatial_components
from cellestial.themes import _THEME_SPATIAL
from cellestial.util import (
    _collect_aes_columns,
    _color_gradient,
    _fill_gradient,
    _resolve_tooltips,
    _validate_tooltips,
    _warn,
)
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from typing import Literal

    from lets_plot.plot.core import PlotSpec
    from polars import DataFrame


def spatial(
    data: AnnData | SpatialData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    library_id: str | None = None,
    image: bool = True,
    image_key: Literal["hires", "lowres"] | str = "lowres",
    greyscale: bool = False,
    image_alpha: float | None = None,
    cmap: str | list | None = None,
    norm: bool | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    scale_axis: Literal[0, 1] | None = None,
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
    add_columns: Sequence[str] | str | None = None,
    include_dimensions: bool | int = False,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
    color_low: str = "#f6f6f6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "mid",
    **point_kwargs,
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
    greyscale : bool, default=False
        Whether to convert an RGB(A) image to greyscale (Rec.709 luminance).
    image_alpha : float | None, default=None
        Alpha (transparency) of the tissue image.
        Distinct from `alpha`, which controls spot transparency.
    cmap : str | list | None, default=None
        Colormap name or list of colors. Greyscale images only.
    norm : bool | None, default=None
        Whether to linearly scale greyscale luminance values to [0, 255].
        Greyscale images only.
    vmin : float | None, default=None
        Lower bound for greyscale luminance normalization.
    vmax : float | None, default=None
        Upper bound for greyscale luminance normalization.
    scale_axis : {0, 1} | None, default=None
        Whether to standardize `key` values between 0 and 1 with min-max
        scaling. Constant values are set to 0.
        Only applied when `key` is numeric.
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
    size : float | None, default=0.8
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
    add_columns : str | Sequence[str] | None, default=None
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
    observations_name : str, default='Barcode'
        The name to give the barcode column in the DataFrame.
    color_low : str, default='#f6f6f6'
        Color for low values in the continuous gradient.
    color_mid : str | None, default=None
        Color for mid values in the continuous gradient. When None, the scale
        falls back to a 2-color gradient between color_low and color_high.
    color_high : str, default='#377eb8'
        Color for high values in the continuous gradient.
    mid_point : {'mean', 'median', 'mid'} | float, default='mid'
        Midpoint for the continuous color gradient.
    **point_kwargs
        Additional parameters forwarded to `geom_point`.

    Returns
    -------
    PlotSpec
        Spatial plot of tissue spots over the H&E image.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported spatial data object.
    KeyError
        If a requested table, image, shape, or coordinate system is missing.
    ValueError
        If spatial components are ambiguous, `crop` is invalid, or polygon
        rendering cannot join shapes back to the table.
    NotImplementedError
        If polygon rendering encounters unsupported MultiPolygon geometry.

    Notes
    -----
    If no tissue image metadata is present, the plot falls back to a plain
    spatial scatter using the raw coordinates.

    Examples
    --------
    An example interactive spatial plot with a categorical key.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.human_lymph_node(cache_directory="data")

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

    Make the background image greyscale.

    .. jupyter-execute::

        cl.spatial(data_hne,key="cluster",greyscale=True,interactive=True)

    Select specific groups to show.

    .. jupyter-execute::

        cl.spatial(data_hne,key="cluster", groups=["Hippocampus", "Hypothalamus_1", "Striatum"], interactive=True)
    """
    # HANDLE: data type
    if not isinstance(data, (AnnData, SpatialData)):
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    image_array, spot_coordinates, polygon_frame, data = _spatial_components(
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

    # HANDLE: mapping
    mapping = mapping or aes()

    # HANDLE: variable_keys
    if variable_keys is None:
        variable_keys = []
    elif isinstance(variable_keys, str):
        variable_keys = [variable_keys]
    elif isinstance(variable_keys, Sequence):
        variable_keys = list(variable_keys)

    # Collect the frame columns from the colour key, aes refs, and `add_columns`.
    if isinstance(add_columns, str):
        add_columns = [add_columns]
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[key, *(add_columns or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=0,
    )
    # Polygon mode joins on an instance key; include it up front.
    instance_key = _resolve_instance_key(data)
    if (
        polygon_frame is not None
        and instance_key is not None
        and instance_key not in metadata_columns
    ):
        metadata_columns.append(instance_key)

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[observations_name, *([key] if key is not None else [])],
        metadata_columns=metadata_columns,
    )

    # BUILD: dataframe
    if frame is None:
        observation_column_name = None if tooltips == "none" else observations_name
        frame = build_frame(
            data=data,
            variable_keys=variable_keys,
            axis=0,
            observations_name=observation_column_name,
            include_dimensions=include_dimensions,
            metadata_columns=metadata_columns,
        )

    is_polygon = polygon_frame is not None
    if is_polygon:
        # Long-format vertices joined to the table on instance_key.
        if instance_key is None or instance_key not in frame.columns:
            msg = (
                "Polygon shapes require the table to have an `instance_key` "
                "column joining it to the shapes element."
            )
            raise ValueError(msg)
        frame = polygon_frame.join(
            frame, left_on="instance_id", right_on=instance_key, how="inner"
        )
    else:
        frame = frame.with_columns(
            pl.Series("spatial_x", spot_coordinates[:, 0]),
            pl.Series("spatial_y", spot_coordinates[:, 1]),
        )

    _validate_tooltips(tooltips, frame)

    # HANDLE: groups filter (categorical-only)
    if groups is not None and key is not None:
        if isinstance(groups, str):
            groups = [groups]
        if frame[key].dtype == pl.Categorical:
            frame = frame.filter(pl.col(key).is_in(list(groups)))
        else:
            msg = f"key `{key}` is not categorical, `groups` filter ignored"
            _warn(msg)

    # HANDLE: drop filter (categorical-only)
    if drop is not None and key is not None:
        if isinstance(drop, str):
            drop = [drop]
        if frame[key].dtype == pl.Categorical:
            frame = frame.filter(~pl.col(key).is_in(list(drop)).fill_null(False))
        else:
            msg = f"key `{key}` is not categorical, `drop` filter ignored"
            _warn(msg)

    # HANDLE: standard scaling (numeric keys only)
    if scale_axis is not None and key is not None and frame[key].dtype.is_numeric():
        value = pl.col(key)
        value_min = value.min()
        value_max = value.max()
        value_range = value_max - value_min
        frame = frame.with_columns(
            pl.when(value_range == 0)
            .then(0.0)
            .otherwise((value - value_min) / value_range)
            .alias(key)
        )

    # BUILD: plot
    sptl = ggplot(data=frame)

    if image_array is not None:
        if greyscale and image_array.ndim == 3:
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

    if "size" in mapping.as_dict():
        size = None

    if is_polygon:
        sptl += geom_polygon(
            mapping=aes(
                x="polygon_x", y="polygon_y", group="instance_id", fill=key, **mapping.as_dict()
            ),
            size=size,
            alpha=alpha,
            tooltips=tooltips,
            **point_kwargs,
        )
        if key is not None:
            if frame[key].dtype == pl.Categorical:
                sptl += scale_fill_brewer(palette="Set2")
            elif frame[key].dtype.is_numeric():
                sptl += _fill_gradient(
                    frame[key],
                    color_low=color_low,
                    color_mid=color_mid,
                    color_high=color_high,
                    mid_point=mid_point,
                )
    else:
        sptl += geom_point(
            mapping=aes(x="spatial_x", y="spatial_y", color=key, **mapping.as_dict()),
            size=size,
            alpha=alpha,
            tooltips=tooltips,
            **point_kwargs,
        )
        if key is not None:
            if frame[key].dtype == pl.Categorical:
                sptl += scale_color_brewer(palette="Set2")
            elif frame[key].dtype.is_numeric():
                sptl += _color_gradient(
                    frame[key],
                    color_low=color_low,
                    color_mid=color_mid,
                    color_high=color_high,
                    mid_point=mid_point,
                )

    # Image rows grow downward in pixel space; reverse y so spots align with the image.
    if crop is not None:
        if len(crop) != 4:
            msg = "crop must be a sequence of 4 integers (left, right, top, bottom)"
            raise ValueError(msg)
        left, right, top, bottom = crop
        coord = coord_fixed(xlim=[left, right], ylim=[top, bottom])
    else:
        coord = coord_fixed()
    sptl += scale_y_reverse() + coord + _THEME_SPATIAL

    if interactive:
        sptl += ggtb(size_zoomin=-1)

    return sptl
