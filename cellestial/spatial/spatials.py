from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from lets_plot import gggrid
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.spatial.spatial import spatial
from cellestial.util import _share_labels

if TYPE_CHECKING:
    from anndata import AnnData
    from lets_plot.plot.subplots import SupPlotsSpec


# AI-GENERATED: Claude 4.7
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line, edge cases unverified
def spatials(
    data: AnnData,
    keys: Sequence[str],
    *,
    library_id: str | None = None,
    image: bool = True,
    image_key: Literal["hires", "lowres"] | str = "hires",
    greyscale: bool = False,
    image_alpha: float | None = None,
    cmap: str | list | None = None,
    norm: bool | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    spatial_key: str = "spatial",
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
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "median",
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
    Grid of spatial transcriptomics plots.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing Visium spatial data.
    keys : Sequence[str]
        The keys (cell features or gene names) to color the spots by.
    library_id : str | None, default=None
        The library identifier under `data.uns['spatial']`.
    image : bool, default=True
        Whether to render the tissue image as a background layer in each plot.
    image_key : str, default='hires'
        Which image to render and which scalefactor to use for spot alignment.
    greyscale, image_alpha, cmap, norm, vmin, vmax
        Image-rendering controls. See `cl.spatial`.
    spatial_key : str, default='spatial'
        The `obsm` key containing spot coordinates in fullres pixel space.
    crop : Sequence[int] | None, default=None
        Crop each plot to `(left, right, top, bottom)` in image-pixel space.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings.
    size : float | None, default=1.5
        Spot size.
    alpha : float, default=1.0
        Spot alpha.
    groups : str | Sequence[str] | None, default=None
        Restrict each plot to spots whose categorical `key` value is in `groups`.
        See `cl.spatial`.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to each DataFrame.
    include_dimensions : bool | int, default=False
        Whether to include `obsm` dimensions in each plot's frame.
    tooltips : {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over a spot.
    interactive : bool, default=False
        Whether to make each plot interactive.
    observations_name : str, default='Barcode'
        Name to give the barcode column.
    color_low, color_mid, color_high : str
        Continuous gradient colors. `color_mid=None` falls back to a 2-color scale.
    mid_point : {'mean', 'median', 'mid'} | float, default='median'
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

    Examples
    --------
    .. jupyter-execute::

        import scanpy as sc
        import cellestial as cl

        data = sc.datasets.visium_sge("V1_Mouse_Kidney")
        data.var_names_make_unique()

        cl.spatials(
            data,
            keys=["in_tissue", "Slc34a1"],
            ncol=2,
        )
    """
    if not isinstance(keys, Sequence) or isinstance(keys, str):
        msg = "keys must be a Sequence of strings"
        raise TypeError(msg)

    plots = []
    for i, key in enumerate(keys):
        plot = spatial(
            data=data,
            key=key,
            library_id=library_id,
            image=image,
            image_key=image_key,
            greyscale=greyscale,
            image_alpha=image_alpha,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            spatial_key=spatial_key,
            crop=crop,
            mapping=mapping,
            size=size,
            alpha=alpha,
            groups=groups,
            variable_keys=variable_keys,
            include_dimensions=include_dimensions,
            tooltips=tooltips,
            interactive=interactive,
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

    return gggrid(
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
