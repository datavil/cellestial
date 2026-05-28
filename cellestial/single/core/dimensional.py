from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

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
    scale_color_brewer,
)
from lets_plot.plot.core import FeatureSpec, PlotSpec

from cellestial.frames import build_frame
from cellestial.layers import _modify_axis, ondata_legend
from cellestial.themes import _THEME_DIMENSION
from cellestial.util import (
    _collect_aes_columns,
    _color_gradient,
    _resolve_embedding_key,
    _resolve_tooltips,
    _validate_tooltips,
    _warn,
)
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec
    from polars import DataFrame


def dimensional(
    data: AnnData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    variable_keys: Sequence[str] | str | None = None,
    add_columns: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
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
    legend_ondata: bool = False,
    ondata_size: float = 12,
    ondata_color: str = "#3f3f3f",
    ondata_fontface: str = "bold",
    ondata_family: str = "sans",
    ondata_alpha: float = 1,
    ondata_label: bool = False,
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
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and key columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    dimensions : {'umap', 'pca', 'tsne'}, default='umap'
        The dimensionality reduction method to use.
        e.g., 'umap' or 'pca' or 'tsne'.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
    size : float | None, default=0.8
        The size of the points.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    add_columns : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the frame,
        on top of those inferred from `key`, `mapping`, and `tooltips`. Useful
        when an added layer reads a column the plot itself does not reference.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups, keeping points where `key` matches any of
        them. Categorical keys only.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups, filtering out points where `key` matches any of
        them. Categorical keys only.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
    color_mid : str, default=None
        The color to use for the middle part of the color gradient.
    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
    mid_point : {'mean', 'median', 'mid'} | float, default='median'
        The midpoint (in data value) of the color gradient.
        Can be 'mean', 'median' and 'mid' or a number (float or int).
        - If 'mean', the midpoint is the mean of the data.
        - If 'median', the midpoint is the median of the data.
        - If 'mid', the midpoint is the mean of 'min' and 'max' of the data.

    axis_type : {'axis', 'arrow'} | None
        Whether to use regular axis or arrows as the axis.
    arrow_length : float, default=0.25
        Span of each axis line as a fraction of its data range (0.25 covers 25%).
    arrow_size : float, default=1
        Width of the axis lines.
    arrow_color : str, default='#3f3f3f'
        Color of the arrows.
    arrow_angle : float, default=10
        Angle of the arrow head in degrees.
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
    ondata_label: bool, default=False
        Whether to draw on-data legends with a filled label background.
    **point_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyError
        If `xy` does not contain exactly two dimensions.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        cl.dimensional(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        cl.dimensional(data,key="CD14",axis_type="arrow",color_high="red")

    """
    # HANDLE: Data types
    if not isinstance(data, AnnData):
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    # HANDLE: mapping
    mapping = mapping or aes()

    #  HANDLE: XY
    if len(xy) != 2:
        msg = f"xy MUST be of length 2, (len(xy)=={len(xy)})"
        raise KeyError(msg)
    prefix = _resolve_embedding_key(data=data, dimensions=dimensions, use_key=use_key, xy=xy)
    x = f"{prefix}{xy[0]}"  # e.g. X_UMAP1
    y = f"{prefix}{xy[1]}"  # e.g. X_UMAP2

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

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=[observations_name, *([key] if key is not None else [])],
        metadata_columns=metadata_columns,
        axis=0,
    )

    # BUILD: dataframe
    # All embeddings are materialised (not just the plotted one) so deferred
    # layers like `stream` can read velocity embeddings from the frame.
    if frame is None:
        # Skip the observation identifier column when no tooltip can reference it.
        observation_column_name = None if tooltips == "none" else observations_name
        frame = build_frame(
            data=data,
            variable_keys=variable_keys,
            axis=0,
            observations_name=observation_column_name,
            include_dimensions=max(xy),
            metadata_columns=metadata_columns,
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

    # BUILD: scatter plot
    if "size" in mapping.as_dict():
        size = None
    scttr = (
        ggplot(data=frame)
        + geom_point(
            mapping=aes(x=x, y=y, color=key, **mapping.as_dict()),
            size=size,
            tooltips=tooltips,
            **point_kwargs,
        )
        + _THEME_DIMENSION
    )

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
    scttr += _modify_axis(
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
    if key is not None and legend_ondata:
        if frame[key].dtype == pl.Categorical:
            scttr += ondata_legend(
                size=ondata_size,
                color=ondata_color,
                fontface=ondata_fontface,
                family=ondata_family,
                alpha=ondata_alpha,
                label=ondata_label,
            )
        elif frame[key].dtype != pl.Categorical:
            msg = f"key `{key}` is not categorical, legend on data will not be added"
            _warn(msg)

    return scttr
