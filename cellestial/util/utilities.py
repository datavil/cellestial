from __future__ import annotations

from math import log10
from typing import Iterable, Literal

import polars as pl
from lets_plot import (
    arrow,
    element_blank,
    element_text,
    geom_blank,
    geom_segment,
    scale_color_continuous,
    scale_color_gradient2,
    theme,
)
from lets_plot.plot.core import FeatureSpec


def _add_arrow_axis(
    frame: pl.DataFrame,
    *,
    axis_type: Literal["axis", "arrow"] | None,
    arrow_size: float,
    arrow_color: str,
    arrow_angle: float,
    arrow_length: float,
    dimensions: str,
):
    """
    Adds arrows as the X and Y axis to the plot.

    Parameters
    ----------
    frame : `polars.DataFrame`
        DataFrame copied from the single cell data.
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
    dimensions : str
        Dimensions of the plot also the prefix of the arrow axis names.
        Accepted values are 'umap', 'pca', 'tsne'.

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
            axis_title_x=element_text(hjust=arrow_length / 2),
            axis_title_y=element_text(hjust=arrow_length / 2),
        )
        x_max = frame.select(f"{dimensions}1").max().item()
        x_min = frame.select(f"{dimensions}1").min().item()
        y_max = frame.select(f"{dimensions}2").max().item()
        y_min = frame.select(f"{dimensions}2").min().item()

        # find total difference between the max and min for both axis
        x_diff = x_max - x_min
        y_diff = y_max - y_min

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
    base_tooltips: Iterable[str],
    add_tooltips: Iterable[str],
    custom_tooltips: Iterable[str],
    *,
    show_tooltips: bool,
) -> list[str] | str:
    """
    Decide on the tooltips.

    Parameters
    ----------
    base_tooltips : list[str]
        Base tooltips, default ones by the function.
    add_tooltips : list[str]
        Additional tooltips, will be appended to the base_tooltips.
    custom_tooltips : list[str]
        Custom tooltips, will overwrite the base_tooltips.
    show_tooltips : bool
        Whether to show tooltips at all.
        Set tooltip to the Literal 'none' if False.

    Returns
    -------
    list[str]
        Tooltips.
    """
    if not show_tooltips:
        tooltips = "none"  # for letsplot, this removes the tooltips
    else:
        if isinstance(custom_tooltips, Iterable):
            tooltips = list(custom_tooltips)
        elif isinstance(add_tooltips, Iterable):
            tooltips = base_tooltips + list(add_tooltips)
        else:
            tooltips = base_tooltips

    return tooltips


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
