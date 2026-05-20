from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Sequence
from functools import lru_cache
from math import ceil, log10
from pathlib import Path
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

from cellestial.util.errors import CellestialWarning, KeyNotFoundError, UnsupportedDataTypeError

_PACKAGE_ROOT = str(Path(__file__).parents[1])


def _format_warning(message: Warning | str, category: type[Warning], *_args: object) -> str:
    """Render a cellestial warning as one line, dropping the source location."""
    return f"{category.__name__}: {message}\n"


def _warn(message: str) -> None:
    """
    Emit a `CellestialWarning` rendered without the source location.

    Parameters
    ----------
    message : str
        The warning message.

    Notes
    -----
    The warning is still attributed to the first caller outside cellestial so
    `warnings` filters and per-location de-duplication behave sensibly, but
    it is displayed as a single clean line.
    """
    caller = inspect.currentframe()
    caller = caller.f_back if caller is not None else None
    stacklevel = 2
    while caller is not None and caller.f_code.co_filename.startswith(_PACKAGE_ROOT):
        stacklevel += 1
        caller = caller.f_back
    original_format = warnings.formatwarning
    warnings.formatwarning = _format_warning
    try:
        warnings.warn(message, CellestialWarning, stacklevel=stacklevel)
    finally:
        warnings.formatwarning = original_format


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


# cache to avoid recomputing for every plot in a grid...
@lru_cache(maxsize=32)
def _grid_places(total: int, ncol: int | None) -> tuple[frozenset[int], frozenset[int]]:
    if ncol is None:
        ncol = total
    nrow = ceil(total / ncol)
    left_places = {idx for idx in range(total) if idx % ncol == 0}
    bottom_places = {idx for idx in range(total) if idx >= ncol * (nrow - 1)}
    # the last grid row may be incomplete; for the columns it does not cover,
    # the bottom-most plot lives in the penultimate row.
    last_row_count = total - ncol * (nrow - 1)
    if nrow >= 2 and last_row_count < ncol:
        bottom_places.update(ncol * (nrow - 2) + col for col in range(last_row_count, ncol))
    return frozenset(left_places), frozenset(bottom_places)


def _share_labels(plot, i: int, keys: Sequence[str], ncol: int | None) -> SupPlotsSpec:
    left_places, bottom_places = _grid_places(len(keys), ncol)
    if i not in bottom_places:  # remove x axis title except for bottom row
        plot = plot + theme(axis_title_x=element_blank())
    if i not in left_places:  # remove y axis title except for left column
        plot = plot + theme(axis_title_y=element_blank())

    return plot


def _share_axis(
    plot, i: int, keys: Sequence[str], ncol: int | None, axis_type: Literal["axis", "arrow"]
) -> SupPlotsSpec:
    left_places, bottom_places = _grid_places(len(keys), ncol)

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
    left_places, bottom_places = _grid_places(len(keys), ncol)
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


def _resolve_embedding_key(
    data: AnnData,
    dimensions: str,
    use_key: str | None,
    xy: Sequence[int],
) -> str:
    """
    Resolve the column-name prefix used to index an embedding in the built frame.

    Parameters
    ----------
    data : AnnData
        Single-cell data object providing the embedding store.
    dimensions : str
        Embedding family, e.g. `"umap"`, `"pca"`, `"tsne"`.
    use_key : str | None
        Explicit embedding key. When provided, no search is performed.
    xy : Sequence[int]
        Axis indices requested (1-based); used to filter shape-eligible
        candidates during fallback.

    Returns
    -------
    str
        Uppercase column-name prefix.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If no embedding matches the requested family.

    Notes
    -----
    `build_frame` emits embedding columns as `f"{KEY.upper()}{n}"`, so callers
    compose axis names like `X_UMAP1` / `X_UMAP2` by appending the axis
    index to the prefix returned here.

    Resolution rules:

    1. If `use_key` is given, return it uppercased.
    2. Otherwise, against `f"X_{dimensions.upper()}"`:
       - exact (case-insensitive) match -> return it silently.
       - no exact, but prefix matches exist -> pick the candidate whose
         dimensionality is sufficient for `max(xy)` (shortest name on ties),
         warn the user, and suggest `use_key` to silence.
       - no candidate -> raise `KeyNotFoundError`.
    """
    if use_key is not None:
        return use_key.upper()

    if isinstance(data, AnnData):
        candidates = list(data.obsm.keys())
        shapes = {key: data.obsm[key].shape[1] for key in candidates}
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

    target = f"X_{dimensions.upper()}"
    needed = max(xy)
    search_targets = (target, dimensions.upper())  # accept bare `UMAP` too, not only `X_UMAP`

    for key in candidates:
        if key.upper() in search_targets:
            return key.upper()

    prefix_matches = [
        key for key in candidates if any(key.upper().startswith(t) for t in search_targets)
    ]
    if not prefix_matches:
        msg = (
            f"No embedding found for `dimensions='{dimensions}'`\n"
            f"(looked for `{target}` or any key starting with it).\n"
            f"Available embeddings: {candidates}."
        )
        raise KeyNotFoundError(msg)

    eligible = [key for key in prefix_matches if shapes[key] >= needed]
    pool = eligible or prefix_matches
    chosen = sorted(pool, key=lambda key: (len(key), key))[0]

    alternatives = sorted(prefix_matches)
    _warn(
        f"No embedding named `{target}` found; "
        f"using `{chosen}` with (shape={shapes[chosen]})\n"
        f"Alternatives: {alternatives}, Rest: {list(set(candidates) - set(alternatives))}.\n"
        "Pass `use_key='...'` to use a specific key."
    )
    return chosen.upper()


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
