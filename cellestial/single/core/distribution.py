from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from anndata import AnnData
from lets_plot import (
    aes,
    geom_boxplot,
    geom_jitter,
    geom_point,
    geom_sina,
    geom_violin,
    ggplot,
    ggtb,
    layer_tooltips,
)
from lets_plot.plot.core import FeatureSpec, PlotSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_DIST
from cellestial.util import (
    _determine_axis,
    _select_variable_keys,
)

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec


def _distribution(
    data: AnnData,
    key: str | Sequence[str],
    *,
    geom: Literal["violin", "boxplot"] = "violin",
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str | None = "#FF00FF",
    geom_color: str | None = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    show_points: bool = True,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs,
) -> PlotSpec:
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # convert to list if single string
    if isinstance(key, str):
        keys = [key]
    elif isinstance(key, Sequence):
        keys = list(key)

    # handle value_column if single key is provided
    if len(keys) == 1:
        value_column = keys[0]

    # determine separator
    separator = None
    if fill is not None:
        separator = fill
    elif color is not None:
        separator = color
    
    if point_kwargs is None:
        point_kwargs = {}

    # determine index to unpivot
    index = [separator] if separator else []
    if add_keys is not None:
        if isinstance(add_keys, str):
            add_keys = [add_keys]
        index.extend(add_keys)

    # DETERMINE: axis if not provided
    axis = _determine_axis(data=data, keys=keys) if axis is None else axis

    # handle fill and color
    geom_fill = None if fill is not None else geom_fill
    geom_color = None if color is not None else geom_color

    # BUILD: the DataFrame
    variable_keys = _select_variable_keys(data=data, keys=keys)
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
    )

    frame = frame.unpivot(
        on=keys, index=index, value_name=value_column, variable_name=variable_column
    )
    if separator is None or len(keys) > 1:
        separator = variable_column

    # HANDLE: tooltips
    if tooltips is None:
        tooltips = [variable_column, value_column]
        tooltips_spec = layer_tooltips(tooltips)
    elif tooltips == "none" or isinstance(tooltips, str):
        tooltips_spec = tooltips
    elif isinstance(tooltips, Sequence):
        tooltips_spec = layer_tooltips(tooltips)
    elif isinstance(tooltips, FeatureSpec):
        tooltips_spec = tooltips

    # BUILD: the plot
    dst = ggplot(data=frame) + _THEME_DIST

    # add the geom layer
    if geom == "violin":
        dst += geom_violin(
            mapping=aes(x=separator, y=value_column, color=color, fill=fill),
            fill=geom_fill,
            color=geom_color,
            tooltips=layer_tooltips(frame.columns),
            **geom_kwargs,
        )
    elif geom == "boxplot":
        dst += geom_boxplot(
            mapping=aes(x=separator, y=value_column, color=color, fill=fill),
            fill=geom_fill,
            color=geom_color,
            tooltips=layer_tooltips(frame.columns),
            **geom_kwargs,
        )

    # handle the points (jitter,point,sina)
    if show_points:
        if point_geom in ["jitter", "point", "sina"]:
            geom_functions = {
                "jitter": geom_jitter,
                "point": geom_point,
                "sina": geom_sina,
            }
            geom_function = geom_functions.get(point_geom, geom_jitter)

            dst += geom_function(
                data=frame,
                mapping=aes(x=separator, y=value_column),
                color=point_color,
                alpha=point_alpha,
                size=point_size,
                tooltips=tooltips_spec,
                **point_kwargs,
            )
        else:
            msg = "point_geom must be one of ['jitter','point','sina']."
            raise KeyError(msg)

    # handle interactive
    if interactive:
        dst += ggtb(size_zoomin=-1)

    return dst


def violin(
    data: AnnData,
    key: str | Sequence[str],
    *,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str | None = "#FF00FF",
    geom_color: str | None = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    show_points: bool = True,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs,
) -> PlotSpec:
    """
    Violin Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str | Sequence[str]
        The key(s) to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the violin plot (categorical).
        e,g., 'cell_type' or 'leiden'.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: Literal['none'] | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    geom_fill : str | None, default="#FF00FF"
        Fill color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str | None, default="#2f2f2f"
        Border color for all violins in the violin plot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
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
    point_geom : Literal["jitter","point","sina"], default is "jitter",
        Geom type of the points, default is geom_jitter.
    observations_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default="Variable"
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
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
    **geom_kwargs
        Additional parameters for the `geom_violin` layer.
        For more information on geom_violin parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_violin.html

    Returns
    -------
    PlotSpec
        Violin plot.
    """
    return _distribution(
        data=data,
        key=key,
        geom="violin",
        axis=axis,
        color=color,
        fill=fill,
        add_keys=add_keys,
        tooltips=tooltips,
        geom_fill=geom_fill,
        geom_color=geom_color,
        point_color=point_color,
        point_alpha=point_alpha,
        point_size=point_size,
        point_geom=point_geom,
        observations_name=observations_name,
        variables_name=variables_name,
        show_points=show_points,
        interactive=interactive,
        value_column=value_column,
        variable_column=variable_column,
        point_kwargs=point_kwargs,
        **geom_kwargs,
    )


def boxplot(
    data: AnnData,
    key: str | Sequence[str],
    *,
    axis: Literal[0, 1] | None = None,
    color: str | None = None,
    fill: str | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    geom_fill: str | None = "#FF00FF",
    geom_color: str | None = "#2f2f2f",
    point_color: str = "#1f1f1f",
    point_alpha: float = 0.7,
    point_size: float = 0.5,
    point_geom: Literal["jitter", "point", "sina"] = "jitter",
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    show_points: bool = True,
    interactive: bool = False,
    value_column: str = "value",
    variable_column: str = "variable",
    point_kwargs: dict[str, Any] | None = None,
    **geom_kwargs,
) -> PlotSpec:
    """
    Boxplot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    key : str | Sequence[str]
        The key(s) to get the values (numerical).
        e.g., 'total_counts' or a gene name.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    color : str | None, default=None
        Color aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    fill : str | None, default=None
        Fill aesthetic to split the boxplot (categorical).
        e,g., 'cell_type' or 'leiden'.
    add_keys : Sequence[str] | str | None, default=None
        Additional keys to include in the dataframe.
    tooltips: Literal['none'] | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    geom_fill : str | None, default="#FF00FF"
        Fill color for all boxplots in the boxplot.
        - Accepts:
            - hex code e.g. '#f1f1f1'
            - color name (of a limited set of colors).
            - RGB/RGBA e.g. 'rgb(0, 0, 255)', 'rgba(0, 0, 255, 0.5)'.
    geom_color : str | None, default="#2f2f2f"
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
    point_geom : Literal["jitter","point","sina"], default is "jitter",
        Geom type of the points, default is geom_jitter.
    observations_name : str, default="Barcode"
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default="Variable"
        The name to give to variable index column in the dataframe.
    show_points : bool, default=True
        Whether to show points.
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
    **geom_kwargs
        Additional parameters for the `geom_boxplot` layer.
        For more information on geom_boxplot parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_boxplot.html

    Returns
    -------
    PlotSpec
        Boxplot.
    """
    return _distribution(
        data=data,
        key=key,
        geom="boxplot",
        axis=axis,
        color=color,
        fill=fill,
        add_keys=add_keys,
        tooltips=tooltips,
        geom_fill=geom_fill,
        geom_color=geom_color,
        point_color=point_color,
        point_alpha=point_alpha,
        point_size=point_size,
        point_geom=point_geom,
        observations_name=observations_name,
        variables_name=variables_name,
        show_points=show_points,
        interactive=interactive,
        value_column=value_column,
        variable_column=variable_column,
        point_kwargs=point_kwargs,
        **geom_kwargs,
    )
