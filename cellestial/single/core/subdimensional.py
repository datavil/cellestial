from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from cellestial.single.core.dimensional import dimensional
from cellestial.util import _is_variable_key, _reject_sequence_key
from cellestial.util.errors import VariableNotFoundError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.core import FeatureSpec, PlotSpec
    from mudata import MuData
    from polars import DataFrame


def umap(
    data: AnnData | MuData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
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
    **point_kwargs,
) -> PlotSpec:
    """
    UMAP Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and `key` columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
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
    observations_name : str, default='Barcode'
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
    **point_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.umap(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.umap(data,key="CD14",axis_type="arrow",color_high="red")

    """
    _reject_sequence_key(key, singular="umap", plural="umaps")
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions="umap",
        use_key=use_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        groups=groups,
        drop=drop,
        tooltips=tooltips,
        interactive=interactive,
        observations_name=observations_name,
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
        **point_kwargs,
    )


def tsne(
    data: AnnData | MuData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
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
    **point_kwargs,
) -> PlotSpec:
    """
    tSNE Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and `key` columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
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
    observations_name : str, default='Barcode'
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
    **point_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.tsne(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.tsne(data,key="CD14",axis_type="arrow",color_high="red")

    """  # noqa: D403
    _reject_sequence_key(key, singular="tsne", plural="tsnes")
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions="tsne",
        use_key=use_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        groups=groups,
        drop=drop,
        tooltips=tooltips,
        interactive=interactive,
        observations_name=observations_name,
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
        **point_kwargs,
    )


def pca(
    data: AnnData | MuData,
    key: str | None = None,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
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
    **point_kwargs,
) -> PlotSpec:
    """
    PCA Dimensionality reduction plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str, default=None
        The key (cell feature) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    frame : DataFrame | None, default=None
        A prebuilt frame to plot from. If provided, the frame is used directly and
        building from `data` is skipped. Must contain the embedding and `key` columns.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
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
    observations_name : str, default='Barcode'
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
    **point_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensionality reduction plot.

    Examples
    --------
    Dimensionality reduction plot with categorical data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.pca(data,key="cell_type_lvl1",axis_type="arrow",legend_ondata=True)

    With continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.pca(data,key="CD14",axis_type="arrow",color_high="red")

    """
    _reject_sequence_key(key, singular="pca", plural="pcas")
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions="pca",
        use_key=use_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        groups=groups,
        drop=drop,
        tooltips=tooltips,
        interactive=interactive,
        observations_name=observations_name,
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
        **point_kwargs,
    )


def expression(
    data: AnnData | MuData,
    key: str,
    *,
    frame: DataFrame | None = None,
    mapping: FeatureSpec | None = None,
    dimensions: Literal["umap", "pca", "tsne"] | str = "umap",
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    size: float | None = 0.8,
    variable_keys: Sequence[str] | str | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
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
    **point_kwargs,
) -> PlotSpec:
    """
    Dimensionality reduction plot of expression data.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
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
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
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
    observations_name : str, default='Barcode'
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
    VariableNotFoundError
        If `key` is not present in variable names.

    Examples
    --------
    Dimensionality reduction plot with continuous data.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.expression(data,key="CD14",axis_type="arrow",color_high="red")

    """
    _reject_sequence_key(key, singular="expression", plural="expressions")
    if not _is_variable_key(data, key):
        msg = f"'{key}' is not present in `variable` names"
        raise VariableNotFoundError(msg)
    return dimensional(
        data=data,
        key=key,
        frame=frame,
        mapping=mapping,
        dimensions=dimensions,
        use_key=use_key,
        xy=xy,
        size=size,
        variable_keys=variable_keys,
        add_keys=add_keys,
        tooltips=tooltips,
        interactive=interactive,
        observations_name=observations_name,
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
        **point_kwargs,
    )
