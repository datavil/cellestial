from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from cellestial.single.core.dimensional import dimensional

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from lets_plot.plot.core import PlotSpec


def umap(
    data: AnnData,
    key: str | None = None,
    *,
    use_key: str | None = None,
    xy: tuple[int, int] | Iterable[int, int] = (1, 2),
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
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
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    tooltips_title: str | None = None,
    legend_ondata: bool = False,
    ondata_size: float = 12,
    ondata_color: str = "#3f3f3f",
    ondata_fontface: str = "bold",
    ondata_family: str = "sans",
    ondata_alpha: float = 1,
    ondata_weighted: bool = True,
    **point_kwargs: dict[str, Any],
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
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
    xy : tuple[int, int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float, default=0.8
        The size of the points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    cluster_name : str, default='Cluster'
        The name to overwrite the clustering key in the dataframe and the plot.
    barcode_name : str, default='Barcode'
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
    add_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Custom tooltips, will overwrite the base_tooltips.
    tooltips_title : str | None, default=None
        Title for the tooltips.
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
    **point_kwargs : dict[str, Any]
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensional reduction plot.

    """
    return dimensional(
        data=data,
        key=key,
        dimensions="umap",
        use_key=use_key,
        xy=xy,
        size=size,
        interactive=interactive,
        cluster_name=cluster_name,
        barcode_name=barcode_name,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        show_tooltips=show_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        tooltips_title=tooltips_title,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_weighted=ondata_weighted,
        **point_kwargs,
    )


def tsne(
    data: AnnData,
    key: str | None = None,
    *,
    use_key: str | None = None,
    xy: tuple[int, int] | Iterable[int, int] = (1, 2),
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
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
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    tooltips_title: str | None = None,
    legend_ondata: bool = False,
    ondata_size: float = 12,
    ondata_color: str = "#3f3f3f",
    ondata_fontface: str = "bold",
    ondata_family: str = "sans",
    ondata_alpha: float = 1,
    ondata_weighted: bool = True,
    **point_kwargs: dict[str, Any],
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
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
    xy : tuple[int, int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float, default=0.8
        The size of the points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    cluster_name : str, default='Cluster'
        The name to overwrite the clustering key in the dataframe and the plot.
    barcode_name : str, default='Barcode'
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
    add_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Custom tooltips, will overwrite the base_tooltips.
    tooltips_title : str | None, default=None
        Title for the tooltips.
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
    **point_kwargs : dict[str, Any]
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensional reduction plot.

    """
    return dimensional(
        data=data,
        key=key,
        dimensions="tsne",
        use_key=use_key,
        xy=xy,
        size=size,
        interactive=interactive,
        cluster_name=cluster_name,
        barcode_name=barcode_name,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        show_tooltips=show_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        tooltips_title=tooltips_title,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_weighted=ondata_weighted,
        **point_kwargs,
    )


def pca(
    data: AnnData,
    key: str | None = None,
    *,
    use_key: str | None = None,
    xy: tuple[int, int] | Iterable[int, int] = (1, 2),
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
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
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    tooltips_title: str | None = None,
    legend_ondata: bool = False,
    ondata_size: float = 12,
    ondata_color: str = "#3f3f3f",
    ondata_fontface: str = "bold",
    ondata_family: str = "sans",
    ondata_alpha: float = 1,
    ondata_weighted: bool = True,
    **point_kwargs: dict[str, Any],
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
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
    xy : tuple[int, int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float, default=0.8
        The size of the points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    cluster_name : str, default='Cluster'
        The name to overwrite the clustering key in the dataframe and the plot.
    barcode_name : str, default='Barcode'
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
    add_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Custom tooltips, will overwrite the base_tooltips.
    tooltips_title : str | None, default=None
        Title for the tooltips.
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
    **point_kwargs : dict[str, Any]
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensional reduction plot.

    """
    return dimensional(
        data=data,
        key=key,
        dimensions="pca",
        use_key=use_key,
        xy=xy,
        size=size,
        interactive=interactive,
        cluster_name=cluster_name,
        barcode_name=barcode_name,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        show_tooltips=show_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        tooltips_title=tooltips_title,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_weighted=ondata_weighted,
        **point_kwargs,
    )


def expression(
    data: AnnData,
    key: str,
    *,
    dimensions: Literal["umap", "pca", "tsne"] = "umap",
    use_key: str | None = None,
    xy: tuple[int, int] | Iterable[int, int] = (1, 2),
    size: float = 0.8,
    interactive: bool = False,
    cluster_name: str = "Cluster",
    barcode_name: str = "Barcode",
    color_low: str = "#e6e6e6",
    color_mid: str | None = None,
    color_high: str = "#377eb8",
    mid_point: Literal["mean", "median", "mid"] | float = "median",
    axis_type: Literal["axis", "arrow"] | None = "arrow",
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    show_tooltips: bool = True,
    add_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    custom_tooltips: list[str] | tuple[str] | Iterable[str] | None = None,
    tooltips_title: str | None = None,
    legend_ondata: bool = False,
    ondata_size: float = 12,
    ondata_color: str = "#3f3f3f",
    ondata_fontface: str = "bold",
    ondata_family: str = "sans",
    ondata_alpha: float = 1,
    ondata_weighted: bool = True,
    **point_kwargs: dict[str, Any],
) -> PlotSpec:
    """
    Dimensionality reduction plot of expression data.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str
        The key (genes) to color the points by.
        e.g., 'leiden' or 'louvain' to color by clusters or gene name for expression.
    dimensions : Literal['umap', 'pca', 'tsne'], default='umap'
        The dimensional reduction method to use.
        e.g., 'umap' or 'pca' or 'tsne'.
    use_key : str, default=None
        The specific key to use for the desired dimensions.
        e.g., 'X_umap_2d' or 'X_pca_2d'.
        Otherwise, the function will decide on the key based on the dimensions.
    xy : tuple[int, int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g., (1, 2) for UMAP1 and UMAP2.
    size : float, default=0.8
        The size of the points.
    interactive : bool, default=False
        Whether to make the plot interactive.
    cluster_name : str, default='Cluster'
        The name to overwrite the clustering key in the dataframe and the plot.
    barcode_name : str, default='Barcode'
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
    add_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str] | tuple[str] | Iterable[str] | None, default=None
        Custom tooltips, will overwrite the base_tooltips.
    tooltips_title : str | None, default=None
        Title for the tooltips.
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
    **point_kwargs : dict[str, Any]
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Dimensional reduction plot.

    """
    if key not in data.var_names:
        msg = f"'{key}' is not present in `variable` (gene) names"
        raise ValueError(msg)
    return dimensional(
        data=data,
        key=key,
        dimensions=dimensions,
        use_key=use_key,
        xy=xy,
        size=size,
        interactive=interactive,
        cluster_name=cluster_name,
        barcode_name=barcode_name,
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,
        axis_type=axis_type,
        arrow_length=arrow_length,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        show_tooltips=show_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        tooltips_title=tooltips_title,
        legend_ondata=legend_ondata,
        ondata_size=ondata_size,
        ondata_color=ondata_color,
        ondata_fontface=ondata_fontface,
        ondata_family=ondata_family,
        ondata_alpha=ondata_alpha,
        ondata_weighted=ondata_weighted,
        **point_kwargs,
    )
