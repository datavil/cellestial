from __future__ import annotations

from collections.abc import Sequence
from math import ceil, log10
from typing import Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    arrow,
    element_blank,
    element_text,
    geom_blank,
    geom_segment,
    gggrid,
    guide_legend,
    guides,
    layer_tooltips,
    scale_color_continuous,
    scale_color_gradient2,
    theme,
)
from lets_plot.plot.core import FeatureSpec, PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec


def _add_arrow_axis(
    frame: pl.DataFrame,
    *,
    x: str,
    y: str,
    axis_type: Literal["axis", "arrow"] | None,
    arrow_size: float,
    arrow_color: str,
    arrow_angle: float,
    arrow_length: float,
):
    """
    Adds arrows as the X and Y axis to the plot.

    Parameters
    ----------
    frame : `polars.DataFrame`
        DataFrame copied from the single cell data.
    x : str
        Name of the x axis column.
    y : str
        Name of the y axis column.
    axis_type : Literal["axis", "arrow"] | None
        Whether to use regular axis or arrows as the axis.
    arrow_size : float
        Size of the arrow.
    arrow_color : str
        Color of the arrow.
    arrow_angle : float
        Angle of the arrow head in degrees.
    arrow_length : float
        Length of the arrow head (px).

    Returns
    -------
    `FeatureSpec` or `FeatureSpecArray`
        Theme feature specification.

    for more information on the arrow parameters, see:
    https://lets-plot.org/python/pages/api/lets_plot.arrow.html
    """
    if axis_type is None:
        return theme(
            # remove axis elements
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line=element_blank(),
        )

    elif axis_type == "axis":
        return geom_blank()

    elif axis_type == "arrow":
        new_layer = theme(
            # remove axis elements
            axis_text_x=element_blank(),
            axis_text_y=element_blank(),
            axis_ticks_y=element_blank(),
            axis_ticks_x=element_blank(),
            axis_line=element_blank(),
            # position axis titles according to arrow size
            axis_title_x=element_text(hjust=arrow_length / 2.5),  # better than 2
            axis_title_y=element_text(hjust=arrow_length / 2.5),
        )
        x_max = frame[x].max()
        x_min = frame[x].min()
        y_max = frame[y].max()
        y_min = frame[y].min()

        # find total difference between the max and min for both axis
        x_diff = x_max - x_min  # ty:ignore[unsupported-operator]
        y_diff = y_max - y_min  # ty:ignore[unsupported-operator]

        # find the ends of the arrows
        xend = x_min + arrow_length * x_diff
        yend = y_min + arrow_length * y_diff

        # adjust bottom ends of arrows
        adjust_rate = 0.025
        x0 = x_min - x_diff * adjust_rate
        y0 = y_min - y_diff * adjust_rate

        # X axis
        new_layer += geom_segment(
            x=x0,
            y=y0,
            xend=xend,
            yend=y0,
            color=arrow_color,
            size=arrow_size,
            arrow=arrow(arrow_angle),
        )
        # Y axis
        new_layer += geom_segment(
            x=x0,
            y=y0,
            xend=x0,
            yend=yend,
            color=arrow_color,
            size=arrow_size,
            arrow=arrow(arrow_angle),
        )
    else:
        msg = f"expected 'axis' or 'arrow' for 'axis_type' argument, but received {axis_type}"
        raise ValueError(msg)

    return new_layer


def _decide_tooltips(
    base_tooltips: Sequence[str] | str | None,
    add_tooltips: Sequence[str] | str | None,
    custom_tooltips: Sequence[str] | str | None,
    *,
    show_tooltips: bool,
) -> list[str] | Literal["none"]:
    """
    Decide on the tooltips.

    Parameters
    ----------
    base_tooltips : list[str] | str
        Base tooltips, default ones by the function.
    add_tooltips : list[str] | str
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str] | str
        Custom tooltips, will overwrite the base_tooltips.
    show_tooltips : bool
        Whether to show tooltips at all.
        Set tooltip to the Literal 'none' if False.

    Returns
    -------
    list[str]
        Tooltips.
    """
    # PART 1: CONVERT str TO list
    if isinstance(base_tooltips, str):
        base_tooltips = [base_tooltips]
    if isinstance(add_tooltips, str):
        add_tooltips = [add_tooltips]
    if isinstance(custom_tooltips, str):
        custom_tooltips = [custom_tooltips]

    # PART 2: HANDLE TOOLTIP LOGIC
    if not show_tooltips:
        tooltips = "none"  # for letsplot, this removes the tooltips
    else:
        if isinstance(custom_tooltips, Sequence):
            tooltips = list(custom_tooltips)
        elif isinstance(add_tooltips, Sequence):
            tooltips = list(base_tooltips) + list(add_tooltips)
        else:
            tooltips = list(base_tooltips)

    return tooltips


def _build_tooltips(
    *,
    tooltips: list[str] | str,
    cluster_name: str,
    key: str | None = None,
    title: str | None = None,
    clustering: bool = False,
) -> FeatureSpec | Literal["none"]:
    """Crete the tooltips for the plot."""
    if tooltips == "none":
        return "none"

    tooltips_object = layer_tooltips()
    for tooltip in tooltips:
        if clustering:
            if tooltip != key:
                tooltips_object.line(f"{tooltip}|@{tooltip}")
            else:
                tooltips_object.line(f"{cluster_name}|@{key}")
        else:
            tooltips_object.line(f"{tooltip}|@{tooltip}")
    if title is not None:
        tooltips_object.title(title)

    return tooltips_object


def _range_inclusive(start: float, stop: float, step: int) -> list[float]:
    """Return a list of rounded numbers between start and stop, inclusive."""
    decimals = 0
    if stop - start < 1:
        if stop - start == 0:
            return [start]
        decimals = -round(log10(stop - start)) + 1

    diff = round(stop - start, decimals)
    increment = round(diff / (step - 1), decimals + 1)
    inc_list = []

    for i in range(step):
        inc_list.append(round(start + increment * i, decimals + 2))
    # make unique
    inc_list = list(set(inc_list))
    return sorted(inc_list)


def _color_gradient(
    series,
    color_low=None,
    color_mid=None,
    color_high=None,
    mid_point: Literal["mean", "median", "mid"] | float = "median",
) -> FeatureSpec:
    """
    Create a gradient color feature.

    Parameters
    ----------
    series : polars.Series
        Series to find the mid point of.
    color_low : str
        The color to use for the low end of the color gradient.
    color_mid : str
       The color to use for the mid part of the color gradient.
    color_high : str
        The color to use for the high end of the color gradient.
    mid_point : float, default='median'
        The midpoint (in data value) of the color gradient.
        Can be 'mean', 'median' and 'mid' or a number (float or int).
        If 'mean', the midpoint is the mean of the data.
        If 'median', the midpoint is the median of the data.
        If 'mid', the midpoint is the mean of 'min' and 'max' of the data.

    Returns
    -------
    FeatureSpec
        FeatureSpec object with the gradient color feature.
    """
    if color_mid is None:
        return scale_color_continuous(low=color_low, high=color_high)
    else:
        if isinstance(mid_point, (float, int)):
            mid_value = mid_point
        elif mid_point == "mean":
            mid_value = series.mean()
        elif mid_point == "median":
            mid_value = series.median()
        elif mid_point == "mid":
            mid_value = (series.max() + series.min()) / 2

        return scale_color_gradient2(
            low=color_low,
            mid=color_mid,
            high=color_high,
            midpoint=mid_value,
        )


def retrieve(plot: PlotSpec | SupPlotsSpec, index: int = 0) -> pl.DataFrame:
    """
    Retrieves the dataframe from a PlotSpec or SupPlotsSpec using the index.

    plot : PlotSpec | SupPlotsSpec
        The plot to retrieve the dataframe from.
    index : int, optional
        The index of the figure to retrieve the dataframe from, by default 0

    Returns
    -------
    pl.DataFrame
        The dataframe utilized in the plot.

    Raises
    ------
    TypeError
        If the plot is not a PlotSpec or SupPlotsSpec object.
    """
    SUP_PLOT_KEY = "_SupPlotsSpec__figures"
    PLOT_KEY = "_FeatureSpec__props"

    if isinstance(plot, PlotSpec):
        properties = vars(plot).get(PLOT_KEY)
        frame = properties.get("data") if properties else None
    elif isinstance(plot, SupPlotsSpec):
        figures = vars(plot).get(SUP_PLOT_KEY)
        if figures:
            frame = vars(figures[index]).get(PLOT_KEY).get("data")
        else:
            frame = None
    else:
        print(type(plot))
        msg = "plot must be a (lets_plot) PlotSpec or SupPlotsSpec object"
        raise TypeError(msg)

    if frame is None:
        msg = "Could not retrieve the dataframe from the plot."
        raise ValueError(msg)

    return frame


def slice(
    grid: SupPlotsSpec, index: int | Sequence[int], **kwargs
) -> PlotSpec | SupPlotsSpec | None:
    """
    Slice a ggrid (SupPlotsSpec) objects with given index.

    Parameters
    ----------
    grid : SupPlotsSpec
        The grid to slice.
    index : int | Sequence[int]
        The index or indices of the plots to slice.
    **kwargs : dict[str, Any]
        Additional arguments for the `gggrid` function.
        see: https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    Returns
    -------
    PlotSpec | SupPlotsSpec
        The sliced grid.

    Raises
    ------
    TypeError
        If the grid is not a SupPlotsSpec object.
        If the index is not an int or Sequence[int].
    """
    SUP_PLOT_KEY = "_SupPlotsSpec__figures"
    if isinstance(grid, SupPlotsSpec):
        figures = vars(grid).get(SUP_PLOT_KEY)

        if figures is not None:
            if isinstance(index, int):
                plot = figures[index]
                return plot
            elif isinstance(index, Sequence):
                list_plots = []
                for i in index:
                    list_plots.append(figures[i])
                return gggrid(list_plots, **kwargs)
            else:
                msg = f"Expected int or Sequence for index, but received {type(index)}"
                raise TypeError(msg)
    else:
        msg = f"Expected SupPlotsSpec for grid, but received {type(grid)}"
        raise TypeError(msg)


def _share_labels(plot, i: int, keys: Sequence[str], ncol: int | None) -> SupPlotsSpec:
    if ncol is None:
        ncol = len(keys)
    total = len(keys)
    nrow = ceil(total / ncol)
    left_places = [i for i in range(total) if i % ncol == 0]
    bottom_places = [i for i in range(total) if i >= ncol * (nrow - 1)]
    if len(bottom_places) < ncol:
        penultimate_row = list(range((nrow - 2) * ncol, (nrow - 1) * ncol))
        bottom_places.extend(penultimate_row)
    if i not in bottom_places:  # remove x axis title except for bottom row
        plot = plot + theme(axis_title_x=element_blank())
    if i not in left_places:  # remove y axis title except for left column
        plot = plot + theme(axis_title_y=element_blank())

    return plot


def _share_axis(
    plot, i: int, keys: Sequence[str], ncol: int | None, axis_type: Literal["axis", "arrow"]
) -> SupPlotsSpec:
    total = len(keys)
    if ncol is None:
        ncol = len(keys)
    nrow = ceil(total / ncol)
    left_places = [i for i in range(total) if i % ncol == 0]
    bottom_places = [i for i in range(total) if i >= ncol * (nrow - 1)]
    if len(bottom_places) < ncol:
        penultimate_row = list(range((nrow - 2) * ncol, (nrow - 1) * ncol))
        bottom_places.extend(penultimate_row)

    if axis_type == "axis":
        if i not in bottom_places:  # remove x axis title except for bottom row
            plot = plot + theme(
                # remove x axis elements
                axis_text_x=element_blank(),
                axis_ticks_x=element_blank(),
                axis_line_x=element_blank(),
            )
        if i not in left_places:  # remove y axis title except for left column
            plot = plot + theme(
                # remove y axis elements
                axis_text_y=element_blank(),
                axis_ticks_y=element_blank(),
                axis_line_y=element_blank(),
            )
    elif axis_type == "arrow":
        pass
    else:
        msg = f"expected 'axis' or 'arrow' for 'axis_type' argument, but received {axis_type}"
        raise ValueError(msg)

    return plot


'''
def _key_style(data: AnnData, key: str) -> str:
    """Find the layers with the given key."""
    if key in data.obs.columns:
        origin = "obs"
    elif key in data.var_names:
        origin = "obs"
    elif key in data.var.columns:
        origin = "var"
'''


def _wrap_legend(
    frame: pl.DataFrame, fill: str | None, color: str | None, nrow: int = 5
) -> FeatureSpec:
    legend = guides()
    # CASE1: LEGEND IS SEPARATED BY FILL
    if fill is not None:
        n_distinct = frame.select(fill).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            legend = guides(fill=guide_legend(ncol=ncol))
    # CASE2: LEGEND IS SEPARATED BY COLOR
    if color is not None:
        n_distinct = frame.select(color).unique().height
        if n_distinct > 10:
            ncol = ceil(n_distinct / 10)
            legend = guides(color=guide_legend(ncol=ncol))

    return legend


def _is_variable_key(data: AnnData, key: str) -> bool:
    if isinstance(data, AnnData):
        if key in data.var_names:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_variables(data: AnnData, keys: Sequence[str]) -> bool:
    if isinstance(data, AnnData):
        result = all(key in data.var_names for key in keys)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _is_observation_key(data: AnnData, key: str) -> bool:
    if isinstance(data, AnnData):
        if key in data.obs.columns:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_observations(data: AnnData, keys: Sequence[str]) -> bool:
    if isinstance(data, AnnData):
        result = all(key in data.obs.columns for key in keys)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _select_variable_keys(
    data: AnnData,
    keys: Sequence[str],
) -> list[str]:
    """From given keys, select only those that are variable keys."""
    if isinstance(data, AnnData):
        variable_keys = [key for key in keys if key in data.var_names]
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)
    return variable_keys


def _is_observation_feature(data: AnnData, key: str) -> bool:
    """Check whether the key is in observations axis (axis=0)."""
    if isinstance(data, AnnData):
        if key in data.obs.columns or key in data.var_names:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_observation_features(data: AnnData, keys: Sequence[str]) -> bool:
    """Check whether all the keys are in observations axis (axis=0)."""
    if isinstance(data, AnnData):
        result = all((key in data.obs.columns or key in data.var_names) for key in keys)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _is_variable_feature(data: AnnData, key: str) -> bool:
    """Check whether the key is in variable axis (axis=1)."""
    if isinstance(data, AnnData):
        if key in data.var.columns:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_variable_features(data: AnnData, keys: Sequence[str]) -> bool:
    """Check whether all the keys are in variable axis (axis=1)."""
    if isinstance(data, AnnData):
        result = all(key in data.var.columns for key in keys)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _determine_axis(
    data: AnnData,
    keys: str | Sequence[str],
) -> Literal[0, 1]:
    """Determine the axis based on the given key or keys."""
    if isinstance(keys, str):
        keys = [keys]
    if isinstance(data, AnnData):
        if _are_variable_features(data, keys):
            axis = 1
        elif _are_observation_features(data, keys):
            axis = 0
        else:
            msg = f"Could not determine the axis with given keys ({keys})."
            raise ValueError(msg)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return axis
