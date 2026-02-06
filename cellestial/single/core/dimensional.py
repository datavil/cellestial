from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

# Core scverse libraries
import polars as pl
from anndata import AnnData

# Data retrieval
from lets_plot import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    labs,
    layer_tooltips,
    scale_color_brewer,
)
from lets_plot.plot.core import PlotSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_DIMENSION
from cellestial.util import (
    _add_arrow_axis,
    _color_gradient,
    _decide_tooltips,
    _is_variable_key,
    _legend_ondata,
    _select_variable_keys,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import PlotSpec


def dimensional(
    data: AnnData,
    key: str | None = None,
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float = 0.8,
    interactive: bool = False,
    observations_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "median",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    show_tooltips: bool = True,
    add_tooltips: Sequence[str] | str | None = None,
    custom_tooltips: Sequence[str] | str | None = None,
    legend_ondata: bool = False,
    ondata_size: float = 12,
    ondata_color: str = "#3f3f3f",
    ondata_fontface: str = "bold",
    ondata_family: str = "sans",
    ondata_alpha: float = 1,
    ondata_weighted: bool = True,
    **point_kwargs,
) -> PlotSpec:
    """
    Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    dimensions : Literal['umap', 'pca', 'tsne'], default='umap'
        The dimensional reduction method to use.
        e.g., 'umap' or 'pca' or 'tsne'.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
    size : float, default=0.8
        The size of the points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
        - Applies to continuous (non-categorical) data.

    color_mid : str, default=None
        The color to use for the middle part of the color gradient.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
        - Applies to continuous (non-categorical) data.

    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
        - Applies to continuous (non-categorical) data.

    mid_point : Literal["mean", "median", "mid"] | float, default="median"
        The midpoint (in data value) of the color gradient.
        Can be 'mean', 'median' and 'mid' or a number (float or int).
        - If 'mean', the midpoint is the mean of the data.
        - If 'median', the midpoint is the median of the data.
        - If 'mid', the midpoint is the mean of 'min' and 'max' of the data.

    axis_type : Literal["axis", "arrow"] | None
        Whether to use regular axis or arrows as the axis.
    arrow_length : float, default=0.25
        Length of the arrow head (px).
    arrow_size : float, default=1
        Size of the arrow.
    arrow_color : str, default='#3f3f3f'
        Color of the arrows.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
        - Applies to continuous (non-categorical) data.

    arrow_angle : float, default=10
        Angle of the arrow head in degrees.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    add_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Custom tooltips, will overwrite the base_tooltips.
    legend_ondata: bool, default=False
        whether to show legend on data
    ondata_size: float, default=12
        size of the legend (text) on data.
    ondata_color: str, default='#3f3f3f'
        color of the legend (text) on data
    ondata_fontface: str, default='bold'
        fontface of the legend (text) on data.
        https://lets-plot.org/python/pages/aesthetics.html#font-face
    ondata_family: str, default='sans'
        family of the legend (text) on data.
        https://lets-plot.org/python/pages/aesthetics.html#font-family
    ondata_alpha: float, default=1
        alpha (transparency) of the legend on data.
    ondata_weighted: bool, default=True
        whether to use weighted mean for the legend on data.
        If True, the weighted mean of the group means is used.
        If False, the arithmetic mean of the group means is used.
    **point_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensional reduction plot.

    """
    # HANDLE: Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    #  HANDLE: XY
    if len(xy) != 2:
        msg = f"xy MUST be of length 2, (len(xy)=={len(xy)})"
        raise KeyError(msg)
    if use_key is None:
        x = f"X_{dimensions.upper()}{xy[0]}"  # e.g. X_UMAP1
        y = f"X_{dimensions.upper()}{xy[1]}"  # e.g. X_UMAP2
    else:
        x = f"{use_key}{xy[0]}"  # e.g. X_UMAP1
        y = f"{use_key}{xy[1]}"  # e.g. X_UMAP2

    # HANDLE: tooltips #TODO: refactor this block
    if "tooltips" in point_kwargs:
        tooltips_layer = point_kwargs.pop("tooltips")
        variable_keys = None
    else:
        if key is None:
            base_tooltips = [observations_name]
        else:
            base_tooltips = [observations_name, key]

        # create the list of tooltip keys
        tooltips = _decide_tooltips(
            base_tooltips=base_tooltips,
            add_tooltips=add_tooltips,
            custom_tooltips=custom_tooltips,
            show_tooltips=show_tooltips,
        )
        tooltips_layer = layer_tooltips(tooltips) # create the object
        # extract the variable keys
        if key is not None and _is_variable_key(data, key):
            if tooltips =="none":
                variable_keys = key
            else:
                variable_keys = [key] + _select_variable_keys(data, tooltips)
        else:
            variable_keys = _select_variable_keys(data, tooltips)

    # BUILD: dataframe
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=0,
        observations_name=observations_name,
        include_dimensions=True,
    )

    # BUILD: scatter plot
    # BASE PLOT
    scttr = ggplot(data=frame) + geom_point(
        aes(x=x, y=y, color=key),
        size=size,
        tooltips=tooltips_layer, #TODO: remove from args
        **point_kwargs,
    ) + _THEME_DIMENSION

    if key is not None:
        # CASE1 ---------------------- CATEGORICAL DATA ----------------------
        if frame[key].dtype == pl.Categorical:
            scttr += scale_color_brewer(palette="Set2")

        # CASE2 ---------------------- CONTINUOUS DATA ----------------------
        elif frame[key].dtype.is_numeric():
            scttr += _color_gradient(
                frame[key],
                color_low=color_low,
                color_mid=color_mid,
                color_high=color_high,
                mid_point=mid_point,
            )
        # else: let letsplot handle it

    # HANDLE: tSNE label, a special case for labels
    if dimensions == "tsne":
        x_label = f"tSNE{xy[0]}"
        y_label = f"tSNE{xy[1]}"
        scttr += labs(x=x_label, y=y_label)
    else:
        # UMAP1 and UMAP2 rather than X_UMAP1 and X_UMAP2 etc.,
        scttr += labs(
            x=x.replace("X_", ""),
            y=y.replace("X_", ""),
        )

    # HANDLE: arrow axis
    scttr += _add_arrow_axis(
        frame=frame,
        x=x,
        y=y,
        axis_type=axis_type,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        arrow_length=arrow_length,
    )
    # HANDLE: interactive
    if interactive:
        scttr += ggtb(size_zoomin=-1)

    # HANDLE: legend on data
    if legend_ondata and key is not None:
        if frame[key].dtype == pl.Categorical:
            scttr += _legend_ondata(
                frame=frame,
                x=x,
                y=y,
                cluster_name=key,
                size=ondata_size,
                color=ondata_color,
                fontface=ondata_fontface,
                family=ondata_family,
                alpha=ondata_alpha,
                weighted=ondata_weighted,
            )
        else:
            msg = f"key `{key}` is not categorical, legend on data will not be added"
            warnings.warn(msg, stacklevel=1)

    return scttr
