from __future__ import annotations

from difflib import get_close_matches
from typing import TYPE_CHECKING, Literal

from cellestial.layers.utilities import _reject_constant_color_fill
from cellestial.single.core.dimensional import _KEY_COLOR_HINT, dimensional
from cellestial.util import (
    _is_variable_key,
    _reject_renamed_embedding_kwargs,
    _reject_sequence_key,
)
from cellestial.util.errors import VariableNotFoundError
from cellestial.util.utilities import _container

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from ggrust.plot.core import FeatureSpec, PlotSpec
    from mudata import MuData
    from polars import DataFrame


def umap(
    data: AnnData | MuData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    embedding_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    geom_color: str | None = None,
    geom_fill: str | None = None,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    midpoint: Literal["mean", "median", "mid"] | float = "median",
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
    halo_width: float | None = 0.5,
    halo_color: str | None = None,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    **geom_kwargs,
) -> PlotSpec:
    """
    UMAP Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and `key` columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    embedding_key : str | None, default=None
        Name of the stored embedding to plot, e.g. `X_umap_harmony`.
        If None, the embedding matching the plot type is used.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float | None, default=0.8
        The size of the points.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    add_keys : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the frame,
        on top of those inferred from `key`, `mapping`, and `tooltips`.
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
    observation_id_column : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
    color_mid : str, default=None
        The color to use for the middle part of the color gradient.
    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='median'
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
    halo_width : float | None, default = 0.5
        Width of the text halo (text outline), not rendered when 0.
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    halo_color : str | None, default = None
        Color of the text halo (text outline).
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    outline : bool, default=False
        Whether to outline the point cloud: points are drawn on top of a gap
        layer and an outline layer, so only the rim of the cloud shows a border.
        Outlined points get `alpha=0.7` unless `alpha` is passed.
    outline_width : tuple[float, float], default=(0.3, 0.05)
        Widths of the outline and of the gap between outline and points, as
        fractions of the point size.
    outline_color : tuple[str, str], default=('black', 'white')
        Colors of the outline and of the gap.
    geom_color : str | None, default=None
        Color for all points in the plot. To color by a column, use `key`.
    geom_fill : str | None, default=None
        Fill color for all points in the plot (filled shapes 21 to 25 only).
    **geom_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Raises
    ------
    UnknownParameterError
        If the removed `use_key` or `basis` is passed; use `embedding_key`.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.umap(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.umap(data,key="CD14",axis_type="arrow",color_high="red")

    Highlight a few groups over grey cells.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.umap(data,key="leiden") + cl.highlight(["0","3"])

    """
    _reject_sequence_key(key, singular="umap", plural="umaps")
    _reject_constant_color_fill(geom_kwargs, function="umap", hint=_KEY_COLOR_HINT)
    _reject_renamed_embedding_kwargs(geom_kwargs)
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions="umap",
        embedding_key=embedding_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        groups=groups,
        drop=drop,
        tooltips=tooltips,
        interactive=interactive,
        observation_id_column=observation_id_column,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        midpoint=midpoint,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_label=ondata_label,
        halo_width=halo_width,
        halo_color=halo_color,
        outline=outline,
        outline_width=outline_width,
        outline_color=outline_color,
        geom_color=geom_color,
        geom_fill=geom_fill,
        **geom_kwargs,
    )


def tsne(
    data: AnnData | MuData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    embedding_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    geom_color: str | None = None,
    geom_fill: str | None = None,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    midpoint: Literal["mean", "median", "mid"] | float = "median",
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
    halo_width: float | None = 0.5,
    halo_color: str | None = None,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    **geom_kwargs,
) -> PlotSpec:
    """
    tSNE Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and `key` columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    embedding_key : str | None, default=None
        Name of the stored embedding to plot, e.g. `X_umap_harmony`.
        If None, the embedding matching the plot type is used.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float | None, default=0.8
        The size of the points.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    add_keys : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the frame,
        on top of those inferred from `key`, `mapping`, and `tooltips`.
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
    observation_id_column : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
    color_mid : str, default=None
        The color to use for the middle part of the color gradient.
    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='median'
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
    halo_width : float | None, default = 0.5
        Width of the text halo (text outline), not rendered when 0.
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    halo_color : str | None, default = None
        Color of the text halo (text outline).
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    outline : bool, default=False
        Whether to outline the point cloud: points are drawn on top of a gap
        layer and an outline layer, so only the rim of the cloud shows a border.
        Outlined points get `alpha=0.7` unless `alpha` is passed.
    outline_width : tuple[float, float], default=(0.3, 0.05)
        Widths of the outline and of the gap between outline and points, as
        fractions of the point size.
    outline_color : tuple[str, str], default=('black', 'white')
        Colors of the outline and of the gap.
    geom_color : str | None, default=None
        Color for all points in the plot. To color by a column, use `key`.
    geom_fill : str | None, default=None
        Fill color for all points in the plot (filled shapes 21 to 25 only).
    **geom_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Raises
    ------
    UnknownParameterError
        If the removed `use_key` or `basis` is passed; use `embedding_key`.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.tsne(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.tsne(data,key="CD14",axis_type="arrow",color_high="red")

    Highlight a few groups over grey cells.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.tsne(data,key="leiden") + cl.highlight(["0","3"])

    """  # noqa: D403
    _reject_sequence_key(key, singular="tsne", plural="tsnes")
    _reject_constant_color_fill(geom_kwargs, function="tsne", hint=_KEY_COLOR_HINT)
    _reject_renamed_embedding_kwargs(geom_kwargs)
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions="tsne",
        embedding_key=embedding_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        groups=groups,
        drop=drop,
        tooltips=tooltips,
        interactive=interactive,
        observation_id_column=observation_id_column,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        midpoint=midpoint,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_label=ondata_label,
        halo_width=halo_width,
        halo_color=halo_color,
        outline=outline,
        outline_width=outline_width,
        outline_color=outline_color,
        geom_color=geom_color,
        geom_fill=geom_fill,
        **geom_kwargs,
    )


def pca(
    data: AnnData | MuData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    embedding_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    geom_color: str | None = None,
    geom_fill: str | None = None,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    midpoint: Literal["mean", "median", "mid"] | float = "median",
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
    halo_width: float | None = 0.5,
    halo_color: str | None = None,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    **geom_kwargs,
) -> PlotSpec:
    """
    PCA Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and `key` columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    embedding_key : str | None, default=None
        Name of the stored embedding to plot, e.g. `X_umap_harmony`.
        If None, the embedding matching the plot type is used.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float | None, default=0.8
        The size of the points.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    add_keys : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the frame,
        on top of those inferred from `key`, `mapping`, and `tooltips`.
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
    observation_id_column : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
    color_mid : str, default=None
        The color to use for the middle part of the color gradient.
    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='median'
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
    halo_width : float | None, default = 0.5
        Width of the text halo (text outline), not rendered when 0.
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    halo_color : str | None, default = None
        Color of the text halo (text outline).
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    outline : bool, default=False
        Whether to outline the point cloud: points are drawn on top of a gap
        layer and an outline layer, so only the rim of the cloud shows a border.
        Outlined points get `alpha=0.7` unless `alpha` is passed.
    outline_width : tuple[float, float], default=(0.3, 0.05)
        Widths of the outline and of the gap between outline and points, as
        fractions of the point size.
    outline_color : tuple[str, str], default=('black', 'white')
        Colors of the outline and of the gap.
    geom_color : str | None, default=None
        Color for all points in the plot. To color by a column, use `key`.
    geom_fill : str | None, default=None
        Fill color for all points in the plot (filled shapes 21 to 25 only).
    **geom_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Raises
    ------
    UnknownParameterError
        If the removed `use_key` or `basis` is passed; use `embedding_key`.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.pca(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.pca(data,key="CD14",axis_type="arrow",color_high="red")

    Highlight a few groups over grey cells.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.pca(data,key="leiden") + cl.highlight(["0","3"])

    """
    _reject_sequence_key(key, singular="pca", plural="pcas")
    _reject_constant_color_fill(geom_kwargs, function="pca", hint=_KEY_COLOR_HINT)
    _reject_renamed_embedding_kwargs(geom_kwargs)
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions="pca",
        embedding_key=embedding_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        groups=groups,
        drop=drop,
        tooltips=tooltips,
        interactive=interactive,
        observation_id_column=observation_id_column,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        midpoint=midpoint,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_label=ondata_label,
        halo_width=halo_width,
        halo_color=halo_color,
        outline=outline,
        outline_width=outline_width,
        outline_color=outline_color,
        geom_color=geom_color,
        geom_fill=geom_fill,
        **geom_kwargs,
    )


def expression(
    data: AnnData | MuData,
    key: str,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    dimensions: Literal["umap", "pca", "tsne"] | str = "umap",
    embedding_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    geom_color: str | None = None,
    geom_fill: str | None = None,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observation_id_column: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    midpoint: Literal["mean", "median", "mid"] | float = "median",
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
    halo_width: float | None = 0.5,
    halo_color: str | None = None,
    outline: bool = False,
    outline_width: tuple[float, float] = (0.3, 0.05),
    outline_color: tuple[str, str] = ("black", "white"),
    **geom_kwargs,
) -> PlotSpec:
    """
    Dimensionality reduction plot of expression data.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    key : str
        The key (gene names) to color the points by.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and `key` columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    dimensions : str, default='umap'
        The dimensionality reduction to plot, named without the `X_` prefix.
        e.g., 'umap', 'pca', 'tsne', or a joint embedding such as 'wnn_umap'.
    embedding_key : str | None, default=None
        Name of the stored embedding to plot, e.g. `X_umap_harmony`.
        If None, the embedding matching the plot type is used.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float | None, default=0.8
        The size of the points.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    add_keys : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the frame,
        on top of those inferred from `key`, `mapping`, and `tooltips`.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observation_id_column : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    color_low : str, default='#e6e6e6'
        The color to use for the low end of the color gradient.
    color_mid : str, default=None
        The color to use for the middle part of the color gradient.
    color_high : str, default='#377EB8'
        The color to use for the high end of the color gradient.
    midpoint : {'mean', 'median', 'mid'} | float, default='median'
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
    halo_width : float | None, default = 0.5
        Width of the text halo (text outline), not rendered when 0.
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    halo_color : str | None, default = None
        Color of the text halo (text outline).
        Only applicable when on-data legend is used without background (i.e ondata_label=False).
    outline : bool, default=False
        Whether to outline the point cloud: points are drawn on top of a gap
        layer and an outline layer, so only the rim of the cloud shows a border.
        Outlined points get `alpha=0.7` unless `alpha` is passed.
    outline_width : tuple[float, float], default=(0.3, 0.05)
        Widths of the outline and of the gap between outline and points, as
        fractions of the point size.
    outline_color : tuple[str, str], default=('black', 'white')
        Colors of the outline and of the gap.
    geom_color : str | None, default=None
        Color for all points in the plot. To color by a column, use `key` (a gene).
    geom_fill : str | None, default=None
        Fill color for all points in the plot (filled shapes 21 to 25 only).
    **geom_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Raises
    ------
    VariableNotFoundError
        If `key` is not present in variable names.
    UnknownParameterError
        If the removed `use_key` or `basis` is passed; use `embedding_key`.

    Examples
    --------
    Dimensionality reduction plot with continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from ggrust import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.expression(data,key="CD14",axis_type="arrow",color_high="red")

    """
    _reject_sequence_key(key, singular="expression", plural="expressions")
    _reject_constant_color_fill(geom_kwargs, function="expression", hint=_KEY_COLOR_HINT)
    _reject_renamed_embedding_kwargs(geom_kwargs)
    if not _is_variable_key(data, key):
        close_matches = get_close_matches(key, _container(data).variable_names().astype(str), n=3)
        hint = f" Did you mean {close_matches}?" if close_matches else ""
        msg = (
            f"'{key}' is not present in `variable` names.{hint}\n"
            "`expression` colors by a gene (variable); for observation metadata use `dimensional`."
        )
        raise VariableNotFoundError(msg)
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions=dimensions,
        embedding_key=embedding_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        tooltips=tooltips,
        interactive=interactive,
        observation_id_column=observation_id_column,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        midpoint=midpoint,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_label=ondata_label,
        halo_width=halo_width,
        halo_color=halo_color,
        outline=outline,
        outline_width=outline_width,
        outline_color=outline_color,
        geom_color=geom_color,
        geom_fill=geom_fill,
        **geom_kwargs,
    )
