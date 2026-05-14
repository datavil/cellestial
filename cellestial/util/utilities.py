from __future__ import annotations

import re
from collections.abc import Sequence
from math import ceil, log10
from typing import Literal

import polars as pl
from anndata import AnnData
from lets_plot import (
    element_blank,
    guide_legend,
    guides,
    layer_tooltips,
    scale_color_continuous,
    scale_color_gradient2,
    scale_fill_continuous,
    scale_fill_gradient2,
    theme,
)
from lets_plot.plot.core import FeatureSpec
from lets_plot.plot.subplots import SupPlotsSpec


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


_TOOLTIP_LINE_RE = re.compile(r"@(?:\{([^}]+)\}|([\w.\-^]+))")


def _tooltip_fields(spec: FeatureSpec) -> list[str]:
    """
    Extract data variable names referenced by a `layer_tooltips` FeatureSpec.

    Aesthetic references (`^aes`) are skipped because they are not data columns.
    """
    spec_dict = spec.as_dict()
    fields = list(spec_dict.get("variables") or [])
    for line in spec_dict.get("lines") or []:
        for braced, bare in _TOOLTIP_LINE_RE.findall(line):
            fields.append(braced or bare)
    for fmt in spec_dict.get("formats") or []:
        field = fmt.get("field", "")
        if not field or field.startswith("^"):
            continue
        fields.append(field.removeprefix("@"))
    return fields


def _validate_tooltips(
    tooltips: Sequence[str] | FeatureSpec | Literal["none"] | None,
    frame: pl.DataFrame,
) -> None:
    """Raise ValueError if any tooltip field is not a column in `frame`."""
    if isinstance(tooltips, FeatureSpec):
        fields = _tooltip_fields(tooltips)
    elif isinstance(tooltips, Sequence) and not isinstance(tooltips, str):
        fields = list(tooltips)
    else:
        return
    missing = [f for f in fields if f not in frame.columns]
    if missing:
        msg = f"Tooltip fields not in data: {missing}"
        raise ValueError(msg)


def _resolve_tooltips(
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None,
    *,
    data: AnnData,
    variable_keys: list[str],
    defaults: Sequence[str],
) -> Sequence[str] | FeatureSpec | Literal["none"]:
    """
    Resolve tooltips. `defaults` apply only when tooltips is None.

    Extends `variable_keys` in place with any tooltip fields that are variable
    names, so they get pulled into the frame by `build_frame`.
    """
    if tooltips is None:
        fields = list(defaults)
        resolved: Sequence[str] | FeatureSpec | Literal["none"] = fields
    elif isinstance(tooltips, str):
        fields = []
        resolved = tooltips
    elif isinstance(tooltips, FeatureSpec):
        fields = _tooltip_fields(tooltips)
        resolved = tooltips
    elif isinstance(tooltips, Sequence):
        fields = list(tooltips)
        resolved = fields
    else:
        msg = f"Invalid tooltips type: {type(tooltips)}"
        raise TypeError(msg)

    for variable in _select_variable_keys(data, fields):
        if variable not in variable_keys:
            variable_keys.append(variable)
    return resolved


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

def _fill_gradient(
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
        return scale_fill_continuous(low=color_low, high=color_high)
    else:
        if isinstance(mid_point, (float, int)):
            mid_value = mid_point
        elif mid_point == "mean":
            mid_value = series.mean()
        elif mid_point == "median":
            mid_value = series.median()
        elif mid_point == "mid":
            mid_value = (series.max() + series.min()) / 2

        return scale_fill_gradient2(
            low=color_low,
            mid=color_mid,
            high=color_high,
            midpoint=mid_value,
        )



def _share_labels(plot, i: int, keys: Sequence[str], ncol: int | None) -> SupPlotsSpec:
    if ncol is None:
        ncol = len(keys)
    total = len(keys)
    nrow = ceil(total / ncol)
    left_places = [i for i in range(total) if i % ncol == 0]
    bottom_places = [i for i in range(total) if i >= ncol * (nrow - 1)]
    # the last grid row may be incomplete; for the columns it does not cover,
    # the bottom-most plot lives in the penultimate row.
    last_row_count = total - ncol * (nrow - 1)
    if nrow >= 2 and last_row_count < ncol:
        bottom_places.extend(
            ncol * (nrow - 2) + col for col in range(last_row_count, ncol)
        )
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
    # the last grid row may be incomplete; for the columns it does not cover,
    # the bottom-most plot lives in the penultimate row.
    last_row_count = total - ncol * (nrow - 1)
    if nrow >= 2 and last_row_count < ncol:
        bottom_places.extend(
            ncol * (nrow - 2) + col for col in range(last_row_count, ncol)
        )

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


def _share_ticks(plot, i: int, keys: Sequence[str], ncol: int | None) -> SupPlotsSpec:
    if ncol is None:
        ncol = len(keys)
    total = len(keys)
    nrow = ceil(total / ncol)
    left_places = [i for i in range(total) if i % ncol == 0]
    bottom_places = [i for i in range(total) if i >= ncol * (nrow - 1)]
    # the last grid row may be incomplete; for the columns it does not cover,
    # the bottom-most plot lives in the penultimate row.
    last_row_count = total - ncol * (nrow - 1)
    if nrow >= 2 and last_row_count < ncol:
        bottom_places.extend(
            ncol * (nrow - 2) + col for col in range(last_row_count, ncol)
        )
    if i not in bottom_places:  # remove x axis title except for bottom row
        plot = plot + theme(axis_text_x=element_blank())
    if i not in left_places:  # remove y axis title except for left column
        plot = plot + theme(axis_text_y=element_blank())

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


def _is_variable_key(data: AnnData, key: str | None) -> bool:
    if key is None:
        return False

    if isinstance(data, AnnData):
        if key in data.var_names:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_variables(data: AnnData, keys: Sequence[str] | None) -> bool:
    if keys is None:
        return False

    if isinstance(data, AnnData):
        result = all(key in data.var_names for key in keys)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _is_observation_key(data: AnnData, key: str | None) -> bool:
    if key is None:
        return False

    if isinstance(data, AnnData):
        if key in data.obs.columns:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_observations(data: AnnData, keys: Sequence[str] | None) -> bool:
    if keys is None:
        return False

    if isinstance(data, AnnData):
        result = all(key in data.obs.columns for key in keys)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _select_variable_keys(
    data: AnnData,
    keys: Sequence[str] | None,
) -> list[str]:
    """From given keys, select only those that are variable keys."""
    if keys is None:
        return []

    if isinstance(data, AnnData):
        variable_keys = [key for key in keys if key in data.var_names]
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)
    return variable_keys


def _is_observation_feature(data: AnnData, key: str | None) -> bool:
    """Check whether the key is in observations axis (axis=0)."""
    if key is None:
        return False

    if isinstance(data, AnnData):
        if key in data.obs.columns or key in data.var_names:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_observation_features(data: AnnData, keys: Sequence[str] | None) -> bool:
    """Check whether all the keys are in observations axis (axis=0)."""
    if keys is None:
        return False

    if isinstance(data, AnnData):
        result = all((key in data.obs.columns or key in data.var_names) for key in keys)
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _is_variable_feature(data: AnnData, key: str | None) -> bool:
    """Check whether the key is in variable axis (axis=1)."""
    if key is None:
        return False

    if isinstance(data, AnnData):
        if key in data.var.columns:
            result = True
        else:
            result = False
    else:
        msg = f"Unknown data type: {type(data)}."
        raise TypeError(msg)

    return result


def _are_variable_features(data: AnnData, keys: Sequence[str] | None) -> bool:
    """Check whether all the keys are in variable axis (axis=1)."""
    if keys is None:
        return False

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
