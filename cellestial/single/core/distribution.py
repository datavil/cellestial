from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from typing import TYPE_CHECKING, Any, Literal

from anndata import AnnData
from lets_plot import (
    aes,
    geom_boxplot,
    geom_jitter,
    geom_violin,
    ggplot,
    ggtb,
    guide_legend,
    guides,
    layer_tooltips,
)
from lets_plot.plot.core import PlotSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_DIST
from cellestial.util import (
    _decide_tooltips,
    _determine_axis,
    _select_variable_keys,
)

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def violin(
    data: AnnData,
    key: str | Sequence[str],
    *,
    axis: Literal[0,1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    geom_fill: str = "#FF00FF",
    geom_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    barcode_name: str = "Barcode",
    variable_name: str = "Variable",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Sequence[str] | str | None = None,
    custom_tooltips: list[str] | tuple[str] | Sequence[str] | str | None = None,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs: dict[str, Any],
) -> PlotSpec:
    """
    Violin Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str
        The key to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    geom_fill : str, default="#FF00FF"
        Fill color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str, default="#2f2f2f"
        Border color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_tooltips :
    point_color : str, default="#1f1f1f"
        Color for the points in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the violin plot.
    point_size : float, default=0.5
        Size for the points in the violin plot.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variable_name : str, default="Variable"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Custom tooltips to show.
    interactive : bool, default=False
        Whether to make the plot interactive.
    variable_column : str, default="variable"
        The name of the variable column in the dataframe.
    value_column : str, default="value"
        The name of the value column in the dataframe.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **geom_kwargs : dict[str, Any]
        Additional parameters for the `geom_violin` layer.
        For more information on geom_violin parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_violin.html

    Returns
    -------
    PlotSpec
        Violin plot.
    """
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # convert to list if single string
    if isinstance(key, str):
        keys = [key]
    elif isinstance(key, Sequence):
        keys = list(key)

    # handle geom_kwargs
    if geom_kwargs:
        if "tooltips" in geom_kwargs:
            msg = "violin tooltips are non-customizable by `geom_kwargs`"
            raise KeyError(msg)

    # determine separator
    separator = None
    if fill is not None:
        separator = fill
    elif color is not None:
        separator = color

    # handle point_kwargs
    if point_kwargs is None:
        point_kwargs = {}
    else:
        if "tooltips" in point_kwargs:
            msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
            raise KeyError(msg)

    axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    #identifier = barcode_name if axis == 0 else variable_name

    # handle fill and color
    geom_fill = None if fill is not None else geom_fill
    geom_color = None if color is not None else geom_color

    # BUILD: the DataFrame
    variable_keys = _select_variable_keys(data=data, keys=keys)
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=barcode_name,
        variables_name=variable_name,
    )

    frame = frame.unpivot(
        on=keys, index=separator, value_name=value_column, variable_name=variable_column
    )
    if separator is None or len(keys) > 1:
        separator = variable_column

    # handle tooltips
    base_tooltips = [variable_column,value_column]
    """if color is not None:
        base_tooltips.append(color)
    if fill is not None:
        base_tooltips.append(fill)"""
    # determine tooltips
    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )

    # handle violin tooltips
    geom_tooltips = frame.columns

    # BUILD: the plot
    dst = ggplot(data=frame) + _THEME_DIST

    # add the violin layer
    dst += geom_violin(
        mapping=aes(x=separator, y=value_column, color=color, fill=fill),
        fill=geom_fill,
        color=geom_color,
        tooltips=layer_tooltips(geom_tooltips),
        **geom_kwargs,
    )

    # handle the point (jitter)
    if show_points:
        dst += geom_jitter(
            data=frame,
            mapping=aes(x=separator, y=value_column),
            color=point_color,
            alpha=point_alpha,
            size=point_size,
            tooltips=layer_tooltips(tooltips),
            **point_kwargs,
        )

    # handle interactive
    if interactive:
        dst += ggtb()

    ## wrap the legend (Deprecated as of lets_plot=4.8.0)
    # if fill is not None:
    #     n_distinct = frame.select(fill).unique().height
    #     if n_distinct > 10:
    #         ncol = ceil(n_distinct / 10)
    #         dst = dst + guides(fill=guide_legend(ncol=ncol))
    # if color is not None:
    #     n_distinct = frame.select(color).unique().height
    #     if n_distinct > 10:
    #         ncol = ceil(n_distinct / 10)
    #         dst = dst + guides(color=guide_legend(ncol=ncol))

    return dst


def boxplot(
    data: AnnData,
    key: str,
    *,
    axis: Literal[0,1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    geom_fill: str = "#FF00FF",
    geom_color: str = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    barcode_name: str = "Barcode",
    variable_name: str = "Variable",
    show_tooltips: bool = True,
    show_points: bool = True,
    add_tooltips: list[str] | tuple[str] | Sequence[str] | str | None = None,
    custom_tooltips: list[str] | tuple[str] | Sequence[str] | str | None = None,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs: dict[str, Any],
) -> PlotSpec:
    """
    Boxplot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str
        The key to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    geom_fill : str, default="#FF00FF"
        Fill color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str, default="#2f2f2f"
        Border color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_color : str, default="#1f1f1f"
        Color for the points in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    point_alpha : float, default=0.7
        Alpha (transparency) for the points in the boxplot.
    point_size : float, default=0.5
        Size for the points in the boxplot.
    barcode_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variable_name : str, default="Variable"
        The name to give to variable index column in the dataframe.
    show_tooltips : bool, default=True
        Whether to show tooltips.
    show_points : bool, default=True
        Whether to show points.
    add_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Additional tooltips to show.
    custom_tooltips : list[str] | tuple[str] | Sequence[str] | str | None, default=None
        Custom tooltips to show.
    interactive : bool, default=False
        Whether to make the plot interactive.
    variable_column : str, default="variable"
        The name of the variable column in the dataframe.
    value_column : str, default="value"
        The name of the value column in the dataframe.
    point_kwargs : dict[str, Any] | None, default=None
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html
    **geom_kwargs : dict[str, Any]
        Additional parameters for the `geom_boxplot` layer.
        For more information on geom_boxplot parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_boxplot.html

    Returns
    -------
    PlotSpec
        Boxplot.
    """
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # convert to list if single string
    if isinstance(key, str):
        keys = [key]
    elif isinstance(key, Sequence):
        keys = list(key)

    # handle geom_kwargs
    if geom_kwargs:
        if "tooltips" in geom_kwargs:
            msg = "boxplot tooltips are non-customizable by `geom_kwargs`"
            raise KeyError(msg)

    # determine separator
    separator = None
    if fill is not None:
        separator = fill # fill has higher priority
    elif color is not None:
        separator = color

    # handle point_kwargs
    if point_kwargs is None:
        point_kwargs = {}
    else:
        if "tooltips" in point_kwargs:
            msg = "use tooltips args within the function instead of adding `'tooltips' : 'value'` to `point_kwargs`\n"
            raise KeyError(msg)

    axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    #identifier = barcode_name if axis == 0 else variable_name

    # handle fill and color
    geom_fill = None if fill is not None else geom_fill
    geom_color = None if color is not None else geom_color

    # BUILD: the DataFrame
    variable_keys = _select_variable_keys(data=data, keys=keys)
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=barcode_name,
        variables_name=variable_name,
    )

    frame = frame.unpivot(
        on=keys, index=separator, value_name=value_column, variable_name=variable_column
    )
    if separator is None or len(keys) > 1:
        separator = variable_column

    # handle tooltips
    base_tooltips = [variable_column,value_column]
    """if color is not None:
        base_tooltips.append(color)
    if fill is not None:
        base_tooltips.append(fill)"""
    # determine tooltips
    tooltips = _decide_tooltips(
        base_tooltips=base_tooltips,
        add_tooltips=add_tooltips,
        custom_tooltips=custom_tooltips,
        show_tooltips=show_tooltips,
    )

    # handle boxplot tooltips
    geom_tooltips = frame.columns

    # BUILD: the plot
    dst = ggplot(data=frame) + _THEME_DIST

    # add the violin layer
    dst += geom_boxplot(
        mapping=aes(x=separator, y=value_column, color=color, fill=fill),
        fill=geom_fill,
        color=geom_color,
        tooltips=layer_tooltips(geom_tooltips),
        **geom_kwargs,
    )

    # handle the point (jitter)
    if show_points:
        dst += geom_jitter(
            data=frame,
            mapping=aes(x=separator, y=value_column),
            color=point_color,
            alpha=point_alpha,
            size=point_size,
            tooltips=layer_tooltips(tooltips),
            **point_kwargs,
        )

    # handle interactive
    if interactive:
        dst += ggtb()

    ## wrap the legend (Deprecated as of lets_plot=4.8.0)
    # if fill is not None:
    #     n_distinct = frame.select(fill).unique().height
    #     if n_distinct > 10:
    #         ncol = ceil(n_distinct / 10)
    #         dst = dst + guides(fill=guide_legend(ncol=ncol))
    # if color is not None:
    #     n_distinct = frame.select(color).unique().height
    #     if n_distinct > 10:
    #         ncol = ceil(n_distinct / 10)
    #         dst = dst + guides(color=guide_legend(ncol=ncol))

    return dst
