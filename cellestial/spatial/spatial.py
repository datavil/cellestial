from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    coord_fixed,
    geom_imshow,
    geom_point,
    ggplot,
    ggtb,
    layer_tooltips,
    scale_color_brewer,
    scale_y_reverse,
)
from lets_plot.plot.core import FeatureSpec, PlotSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_SPATIAL
from cellestial.util import (
    _color_gradient,
    _is_variable_key,
    _select_variable_keys,
)

if TYPE_CHECKING:
    from typing import Literal

    from lets_plot.plot.core import PlotSpec
    from numpy.typing import NDArray


# AI-GENERATED: Claude 4.7
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line
def _visium_components(
    data: AnnData,
    *,
    library_id: str | None,
    image_key: str,
    image: bool,
    spatial_key: str,
) -> tuple[NDArray | None, NDArray]:
    """
    Extract the Visium-specific bits cellestial needs to render a spatial plot.

    Returns
    -------
    tuple
        (image_array_or_None, scaled_spot_coords)
        - image_array_or_None: the H&E image to render under the spots, or None
          if image rendering was disabled or the image was missing.
        - scaled_spot_coords: an (N, 2) array of spot coordinates already scaled
          into the chosen image's pixel space.
    """
    if isinstance(data, AnnData):
        if spatial_key not in data.obsm:
            msg = f"`obsm['{spatial_key}']` not found; spatial plot requires spot coordinates"
            raise KeyError(msg)

        spatial_uns = data.uns.get("spatial", {})
        if library_id is None:
            if len(spatial_uns) == 1:
                library_id = next(iter(spatial_uns))
            elif len(spatial_uns) == 0:
                msg = "`uns['spatial']` is empty; cannot resolve image or scalefactors"
                raise KeyError(msg)
            else:
                msg = (
                    f"Multiple libraries found in `uns['spatial']`: {list(spatial_uns)}. "
                    "Pass `library_id=` to select one."
                )
                raise ValueError(msg)
        elif library_id not in spatial_uns:
            msg = f"library_id `{library_id}` not found in `uns['spatial']`"
            raise KeyError(msg)

        library = spatial_uns[library_id]
        scale_factors = library.get("scalefactors", {})
        scale_factor = scale_factors.get(f"tissue_{image_key}_scalef", 1.0)
        spot_coordinates = data.obsm[spatial_key] * scale_factor

        image_array = None
        if image:
            images = library.get("images", {})
            image_array = images.get(image_key)
            if image_array is None:
                msg = f"image `{image_key}` not found for library `{library_id}`"
                warnings.warn(msg, stacklevel=2)
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise TypeError(msg)

    return image_array, spot_coordinates


# AI-GENERATED: Claude 4.7
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line, edge cases unverified
def spatial(
    data: AnnData,
    key: str | None = None,
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
    scale_axis: Literal[0, 1] | None = None,
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
    color_low: str = "#f6f6f6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "median",
    **point_kwargs,
) -> PlotSpec:
    """
    Spatial plot.

    Renders tissue spots over an optional H&E image, colored by an observation
    or gene expression key. Spot coordinates are read from `data.obsm[spatial_key]`
    and aligned with the chosen image using the scalefactors stored in
    `data.uns['spatial'][library_id]`.

    Parameters
    ----------
    data : AnnData
        The AnnData object containing Visium spatial data.
    key : str, default=None
        The key (cell feature or gene name) to color the spots by.
    library_id : str | None, default=None
        The library identifier under `data.uns['spatial']`. If None and only one
        library is present, it is auto-selected; otherwise this must be provided.
    image : bool, default=True
        Whether to render the tissue image as a background layer.
    image_key : str, default='hires'
        Which image under `uns['spatial'][library_id]['images']` to render, and
        which `tissue_<image_key>_scalef` to use for spot alignment.
        Visium ships with 'hires' and 'lowres'.
    greyscale : bool, default=False
        Whether to convert an RGB(A) image to greyscale (Rec.709 luminance).
    image_alpha : float | None, default=None
        Alpha (transparency) of the tissue image. Distinct from `alpha`, which
        controls spot transparency.
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
        Whether to standardize ``key`` values between 0 and 1 (subtracts the
        minimum and divides by the maximum). A spatial plot has one variable
        per plot (the colored ``key``), so 0 and 1 produce the same result —
        the parameter is provided for signature consistency with ``heatmap``.
        Only applied when ``key`` is numeric.
    spatial_key : str, default='spatial'
        The `obsm` key containing spot coordinates in fullres pixel space.
    crop : Sequence[int] | None, default=None
        Crop the plot to a region given as `(left, right, top, bottom)` in the
        chosen image's pixel space (i.e. fullres coords scaled by
        `tissue_<image_key>_scalef`). `top` is the smaller pixel value (image
        origin is top-left); `bottom` is the larger.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings, the result of `aes()`.
    size : float | None, default=0.8
        Spot size.
    alpha : float, default=1.0
        Alpha (transparency) of the spots.
    groups : str | Sequence[str] | None, default=None
        When `key` is categorical, restrict the spots drawn to observations whose
        `key` value matches one of the listed groups. Ignored when `key` is None
        or numeric (a warning is emitted in the latter case).
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    include_dimensions : bool | int, default=False
        Whether to include dimensions from `obsm` (e.g. UMAP, PCA) in the
        DataFrame. An integer caps the number of dimensions per `obsm` entry.
        Spot coordinates are always added regardless of this flag.
    tooltips : {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over a spot. None auto-generates from
        mapped keys; 'none' disables tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observations_name : str, default='Barcode'
        The name to give the barcode column in the DataFrame.
    color_low, color_mid, color_high : str
        Continuous gradient colors. `color_mid=None` falls back to a 2-color scale.
    mid_point : {'mean', 'median', 'mid'} | float, default='median'
        Midpoint for the continuous color gradient.
    **point_kwargs
        Additional parameters forwarded to `geom_point`.

    Returns
    -------
    PlotSpec
        Spatial plot of tissue spots over the H&E image.

    Examples
    --------
    An example interactive spatial plot with a categorical key.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/V1_Human_Lymph_Node_pped.h5ad")

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
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # All AnnData-specific extraction lives in the helper, gated behind the
    # isinstance check above.
    image_array, spot_coordinates = _visium_components(
        data,
        library_id=library_id,
        image_key=image_key,
        image=image,
        spatial_key=spatial_key,
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

    if _is_variable_key(data, key):
        variable_keys.append(key)  # ty:ignore[invalid-argument-type]

    # HANDLE: tooltips
    if tooltips is None:
        tooltip_fields = [observations_name]
        if key is not None:
            tooltip_fields.append(key)
        tooltips_spec = layer_tooltips(tooltip_fields)
    elif tooltips == "none" or isinstance(tooltips, str):
        tooltips_spec = tooltips
    elif isinstance(tooltips, Sequence):
        tooltips = list(tooltips)
        tooltips_spec = layer_tooltips(tooltips)
        tooltips_variables = _select_variable_keys(data, tooltips)
        if not set(tooltips_variables).issubset(variable_keys):
            variable_keys.extend(tooltips_variables)
    elif isinstance(tooltips, FeatureSpec):
        tooltips_spec = tooltips

    # BUILD: dataframe
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=0,
        observations_name=observations_name,
        include_dimensions=include_dimensions,
    )
    frame = frame.with_columns(
        pl.Series("spatial_x", spot_coordinates[:, 0]),
        pl.Series("spatial_y", spot_coordinates[:, 1]),
    )

    # HANDLE: groups filter (categorical-only)
    if groups is not None and key is not None:
        if isinstance(groups, str):
            groups = [groups]
        if frame[key].dtype == pl.Categorical:
            frame = frame.filter(pl.col(key).is_in(list(groups)))
        else:
            msg = f"key `{key}` is not categorical, `groups` filter ignored"
            warnings.warn(msg, stacklevel=1)

    # HANDLE: standard scaling (numeric keys only)
    if scale_axis is not None and key is not None and frame[key].dtype.is_numeric():
        v = pl.col(key)
        frame = frame.with_columns(((v - v.min()) / (v.max() - v.min())).alias(key))

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
    sptl += geom_point(
        mapping=aes(x="spatial_x", y="spatial_y", color=key, **mapping.as_dict()),
        size=size,
        alpha=alpha,
        tooltips=tooltips_spec,
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
