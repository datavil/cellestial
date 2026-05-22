from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from anndata import AnnData
from lets_plot import gggrid, ggtb
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.frames import build_frame
from cellestial.spatial.spatial import spatial
from cellestial.spatial.utilities import _spatial_components
from cellestial.util import _is_variable_key, _share_labels

if TYPE_CHECKING:
    from lets_plot.plot.subplots import SupPlotsSpec
    from spatialdata import SpatialData


def spatials(
    data: AnnData | SpatialData,
    keys: Sequence[str],
    *,
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
    variable_keys: Sequence[str] | str | None = None,
    include_dimensions: bool | int = False,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
    color_low: str = "#f6f6f6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "mid",
    # multi plot args
    share_labels: bool = False,
    layers: Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None = None,
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list[float] | None = None,
    heights: list[float] | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    guides: str = "auto",
    **point_kwargs,
) -> SupPlotsSpec:
    """
    Grid of spatial plots.

    Parameters
    ----------
    data : AnnData | SpatialData
        Spatial data to plot. AnnData inputs use Visium-style metadata stored
        on the object. SpatialData inputs are resolved via the `table_name`,
        `image_name`, `shapes_name`, and `coordinate_system` arguments below.
    keys : Sequence[str]
        The keys (cell features or gene names) to color the spots by.
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
        Whether to standardize `key` values between 0 and 1 (subtracts the
        minimum and divides by the maximum).
        Only applied when `key` is numeric.
    spatial_key : str, default='spatial'
        The embedding key containing spot coordinates in fullres pixel space.
        Ignored for SpatialData inputs.
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
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
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
    share_labels : bool, default=False
        If True, only show axis labels at the grid edges.
    layers : Sequence[FeatureSpec|LayerSpec] | FeatureSpec | LayerSpec | None, default=None
        Layers to add to every plot in the grid.
    ncol : int, default=None
        Number of columns in grid. If None, plots are arranged in one row.
    sharex, sharey : str | None, default=None
        Axis-sharing mode passed to `gggrid`.
    widths, heights : list[float] | None, default=None
        Relative column widths / row heights.
    hspace, vspace : float | None, default=None
        Cell spacing in px.
    fit : bool | None, default=None
        Whether to stretch each plot to its cell.
    align : bool | None, default=None
        Whether to align inner geom areas.
    guides : str, default='auto'
        How guides (legends, colorbars) are collected by `gggrid`.
    **point_kwargs
        Additional parameters forwarded to `geom_point`.

    Returns
    -------
    SupPlotsSpec
        Grid of spatial plots, one per key.

    Notes
    -----
    If no tissue image metadata is present, the plot falls back to a plain
    spatial scatter using the raw coordinates.

    Examples
    --------
    A grid of spatial plots with a sequnce of keys.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.human_lymph_node(cache_directory="data")

        cl.spatials(data,keys=["clusters","MS4A1"],interactive=True)

    With another dataset.

    .. jupyter-execute::

        import squidpy as sq

        data_hne = sq.datasets.visium_hne_adata()

        cl.spatials(data_hne,keys=["leiden","Mef2c"], interactive=True)

    Adjust the layout via setting a column number.

    .. jupyter-execute::

        cl.spatials(data_hne,keys=["leiden","Mef2c"], ncol=1, interactive=True) + ggsize(500,1000)
    """
    if not isinstance(keys, Sequence) or isinstance(keys, str):
        msg = "keys must be a Sequence of strings"
        raise TypeError(msg)

    # Resolve the annotation table once (mirrors `spatial`); SpatialData needs extraction
    # for the variable-key check below, while `build_frame` accepts either type.
    if isinstance(data, AnnData):
        table = data
    else:
        _, _, _, table = _spatial_components(
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

    # BUILD: one shared frame for all keys, instead of rebuilding per key.
    # spatial is always observations-axis; only gene keys are pulled from X.
    if variable_keys is None:
        variable_keys = []
    elif isinstance(variable_keys, str):
        variable_keys = [variable_keys]
    else:
        variable_keys = list(variable_keys)
    variable_keys.extend(key for key in keys if _is_variable_key(table, key))
    variable_keys = list(dict.fromkeys(variable_keys))
    frame = build_frame(
        data=table,
        variable_keys=variable_keys,
        axis=0,
        observations_name=observations_name,
        include_dimensions=include_dimensions,
    )

    plots = []
    for i, key in enumerate(keys):
        plot = spatial(
            data=data,
            key=key,
            frame=frame,
            library_id=library_id,
            image=image,
            image_key=image_key,
            greyscale=greyscale,
            image_alpha=image_alpha,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            scale_axis=scale_axis,
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
            variable_keys=variable_keys,
            include_dimensions=include_dimensions,
            tooltips=tooltips,
            observations_name=observations_name,
            color_low=color_low,
            color_mid=color_mid,
            color_high=color_high,
            mid_point=mid_point,
            **point_kwargs,
        )

        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        if share_labels:
            plot = _share_labels(plot, i, keys, ncol)

        plots.append(plot)

    sptls = gggrid(
        plots,
        ncol=ncol,  # ty:ignore[invalid-argument-type]
        sharex=sharex,  # ty:ignore[invalid-argument-type]
        sharey=sharey,  # ty:ignore[invalid-argument-type]
        widths=widths,  # ty:ignore[invalid-argument-type]
        heights=heights,  # ty:ignore[invalid-argument-type]
        hspace=hspace,  # ty:ignore[invalid-argument-type]
        vspace=vspace,  # ty:ignore[invalid-argument-type]
        fit=fit,  # ty:ignore[invalid-argument-type]
        align=align,  # ty:ignore[invalid-argument-type]
        guides=guides,
    )
    if interactive:
        sptls += ggtb(size_zoomin=-1)

    return sptls
