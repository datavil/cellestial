from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Sequence
from functools import lru_cache
from math import ceil, floor, isfinite, log10
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast, overload

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
from mudata import MuData

from cellestial.util.errors import CellestialWarning, KeyNotFoundError

if TYPE_CHECKING:
    from cellestial.frames._container import _Container

_PACKAGE_ROOT = str(Path(__file__).parents[1])


def _container(data: AnnData | MuData) -> _Container:
    """
    Return the backend-agnostic container view for `data`.

    Notes
    -----
    The underlying module is imported on call rather than at module level.
    `cellestial.frames` imports `cellestial.util.errors`, which executes this
    package's `__init__` and so imports this module while
    `cellestial.frames._container` is still initialising.
    """
    from cellestial.frames._container import _container as _build_container

    return _build_container(data)


@overload
def _modality_source(
    data: AnnData | MuData, modality: str | None, group_by: str
) -> tuple[AnnData, str]: ...


@overload
def _modality_source(
    data: AnnData | MuData, modality: str | None, group_by: None = None
) -> tuple[AnnData, None]: ...


def _modality_source(
    data: AnnData | MuData, modality: str | None, group_by: str | None = None
) -> tuple[AnnData, str | None]:
    """
    Return the data object holding stored results, and `group_by` as it names it.

    Stored results (rankings, dendrograms) live in a single modality, while
    `group_by` names a container-level column, so the column is translated to
    the name that modality uses.
    """
    container = _container(data)
    holder = container.select_modality(modality)
    if group_by is None:
        return holder, None
    return holder, container.modality_column(modality, group_by)


def _container_column(data: AnnData | MuData, modality: str | None, group_by: str) -> str:
    """
    Return the container's name for a group column that came from a modality.

    The inverse of `_modality_source`. A ranking stores the group column under
    the modality's own name, but the plotted frame is built from the container,
    where that column is qualified as `modality:column`.
    """
    return _container(data).container_column(modality, group_by)


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
    warnings.formatwarning = _format_warning  # ty:ignore[invalid-assignment]
    try:
        warnings.warn(message, CellestialWarning, stacklevel=stacklevel)
    finally:
        warnings.formatwarning = original_format


def _drop_nonfinite_rows(frame: pl.DataFrame, columns: Sequence[str]) -> pl.DataFrame:
    """Keep rows whose selected numeric columns are present and finite."""
    numeric_columns = [
        column
        for column in dict.fromkeys(columns)
        if column in frame.columns and frame.schema[column].is_numeric()
    ]
    if not numeric_columns:
        return frame
    return frame.filter(
        pl.all_horizontal(
            pl.col(column).is_not_null() & pl.col(column).is_finite() for column in numeric_columns
        )
    )


def _build_tooltips(
    *,
    tooltips: list[str] | str,
    cluster_name: str,
    key: str | None = None,
    title: str | None = None,
    clustering: bool = False,
) -> FeatureSpec | Literal["none"]:
    """Builds the tooltips for the plot."""
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


def _qualified_alternatives(data: AnnData | MuData | None, key: str) -> list[str]:
    """
    Return metadata columns that differ from `key` only by a modality prefix.

    A modality's columns are carried on the container as `modality:column`, so a
    bare name is a natural thing to reach for and a confusing thing to be told
    does not exist.
    """
    if data is None:
        return []
    return [
        name
        for name in _container(data).observation_columns()
        if name != key and name.partition(":")[2] == key
    ]


def _validate_aesthetic_columns(
    frame: pl.DataFrame,
    *,
    data: AnnData | MuData | None = None,
    **aesthetics: str | None,
) -> None:
    """Raise KeyNotFoundError if an aesthetic argument does not name a column in `frame`."""
    for aesthetic, column in aesthetics.items():
        if column is None or column in frame.columns:
            continue
        lines = [f"`{aesthetic}={column!r}` is not a column in the data."]
        if aesthetic in {"color", "fill"}:
            lines.append(f"For a constant {aesthetic}, use `geom_{aesthetic}={column!r}`.")
        alternatives = _qualified_alternatives(data, column)
        if alternatives:
            lines.append(
                f"Did you mean one of {alternatives}? "
                "Columns belonging to a modality carry its prefix."
            )
        lines.append(f"Available columns: {frame.columns}")
        raise KeyNotFoundError("\n".join(lines))


def _resolve_tooltips(
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None,
    *,
    data: AnnData | MuData,
    variable_keys: list[str],
    defaults: Sequence[str],
    metadata_columns: list[str] | None = None,
    axis: Literal[0, 1] = 0,
) -> Sequence[str] | FeatureSpec | Literal["none"]:
    """
    Resolve tooltips. `defaults` apply only when tooltips is None.

    Extends `variable_keys` in place with any tooltip fields that are variable
    names. When `metadata_columns` is provided, the tooltip fields are routed
    onto the requested `axis` (observation metadata for axis 0, variable
    metadata for axis 1) so callers can request a narrow frame from
    `build_frame`.
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

    if metadata_columns is None:
        # No narrowing requested; only pull gene tooltip fields into variable_keys.
        for variable in _select_variable_keys(data, fields):
            if variable not in variable_keys:
                variable_keys.append(variable)
    else:
        _collect_aes_columns(
            data,
            keys=fields,
            mapping=None,
            metadata_columns=metadata_columns,
            variable_keys=variable_keys,
            axis=axis,
        )
    return resolved


def _range_inclusive(start: float, stop: float, step: int) -> list[float]:
    """Return a list of rounded numbers between start and stop, inclusive."""
    if step < 1:
        msg = f"`step` must be >= 1, but received {step}."
        raise ValueError(msg)
    span = abs(stop - start)
    if span == 0:
        return [start]
    if step == 1:
        return [stop]

    # Interpolate from the endpoints so both of them are always hit, and round
    # to enough decimals to keep neighbouring values distinct.
    decimals = max(0, -floor(log10(span / (step - 1))) + 1)
    values = {
        round(start + (stop - start) * index / (step - 1), decimals) for index in range(step)
    }
    return sorted(values)


def _resolve_midpoint(
    series: pl.Series,
    midpoint: Literal["mean", "median", "mid"] | float,
) -> float:
    """Resolve and validate the midpoint of a diverging color scale."""
    finite_series = series.filter(series.is_not_null() & series.is_finite())
    if finite_series.is_empty():
        msg = "Cannot resolve `midpoint` without finite data values."
        raise ValueError(msg)
    data_min = cast("float", finite_series.min())
    data_max = cast("float", finite_series.max())

    if isinstance(midpoint, bool):
        msg = "`midpoint` must be a number or one of 'mean', 'median', or 'mid'."
        raise TypeError(msg)
    if isinstance(midpoint, Real):
        mid_value = float(midpoint)
    elif midpoint == "mean":
        mid_value = cast("float", finite_series.mean())
    elif midpoint == "median":
        mid_value = cast("float", finite_series.median())
    elif midpoint == "mid":
        mid_value = (data_min + data_max) / 2
    elif isinstance(midpoint, str):
        msg = "`midpoint` must be one of 'mean', 'median', or 'mid', or a number."
        raise ValueError(msg)
    else:
        msg = "`midpoint` must be a number or one of 'mean', 'median', or 'mid'."
        raise TypeError(msg)

    if not isfinite(mid_value):
        msg = "`midpoint` must be finite."
        raise ValueError(msg)

    if not data_min <= mid_value <= data_max:
        msg = (
            f"`midpoint` ({mid_value}) must be within the finite data range "
            f"[{data_min}, {data_max}]."
        )
        raise ValueError(msg)

    return mid_value


def _color_gradient(
    series,
    color_low=None,
    color_mid=None,
    color_high=None,
    midpoint: Literal["mean", "median", "mid"] | float = "median",
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
    midpoint : {'mean', 'median', 'mid'} | float, default='median'
        The midpoint (in data value) of the color gradient.
        Can be 'mean', 'median' and 'mid' or a number (float or int).
        If 'mean', the midpoint is the mean of the data.
        If 'median', the midpoint is the median of the data.
        If 'mid', the midpoint is the mean of 'min' and 'max' of the data.
        Non-finite data values are ignored. Numeric midpoints must be finite and
        within the finite data range.

    Returns
    -------
    FeatureSpec
        FeatureSpec object with the gradient color feature.
    """
    if color_mid is None:
        return scale_color_continuous(low=color_low, high=color_high)
    else:
        mid_value = _resolve_midpoint(series, midpoint)

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
    midpoint: Literal["mean", "median", "mid"] | float = "median",
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
    midpoint : {'mean', 'median', 'mid'} | float, default='median'
        The midpoint (in data value) of the color gradient.
        Can be 'mean', 'median' and 'mid' or a number (float or int).
        If 'mean', the midpoint is the mean of the data.
        If 'median', the midpoint is the median of the data.
        If 'mid', the midpoint is the mean of 'min' and 'max' of the data.
        Non-finite data values are ignored. Numeric midpoints must be finite and
        within the finite data range.

    Returns
    -------
    FeatureSpec
        FeatureSpec object with the gradient color feature.
    """
    if color_mid is None:
        return scale_fill_continuous(low=color_low, high=color_high)
    else:
        mid_value = _resolve_midpoint(series, midpoint)

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


def _reject_sequence_key(key: object, *, singular: str, plural: str) -> None:
    """Reject a non-string sequence where a single key is expected, pointing to the plural API."""
    if key is None or isinstance(key, str):
        return
    if isinstance(key, Sequence):
        msg = (
            f"`{singular}` plots a single key but received {len(key)} keys. "
            f"Use `{plural}` for multiple keys."
        )
        raise TypeError(msg)


def _require_feature_key(data: AnnData | MuData, key: str | None) -> None:
    """Validate that a provided color key resolves to a known column or variable."""
    if key is None:
        return
    container = _container(data)
    if key in container.observation_columns() or container.owns_variable(key):
        return
    msg = f"Key `{key}` not found.\nLooked in observation metadata and variable (gene) names."
    raise KeyNotFoundError(msg)


def _is_variable_key(data: AnnData | MuData, key: str | None) -> bool:
    if key is None:
        return False

    return _container(data).owns_variable(key)


def _are_variables(data: AnnData | MuData, keys: Sequence[str] | None) -> bool:
    if keys is None:
        return False

    container = _container(data)
    return all(container.owns_variable(key) for key in keys)


def _is_observation_key(data: AnnData | MuData, key: str | None) -> bool:
    if key is None:
        return False

    return key in _container(data).observation_columns()


def _are_observations(data: AnnData | MuData, keys: Sequence[str] | None) -> bool:
    if keys is None:
        return False

    columns = _container(data).observation_columns()
    return all(key in columns for key in keys)


def _select_variable_keys(
    data: AnnData | MuData,
    keys: Sequence[str] | None,
) -> list[str]:
    """From given keys, select only those that are variable keys."""
    if keys is None:
        return []

    container = _container(data)
    return [key for key in keys if container.owns_variable(key)]


def _collect_aes_columns(
    data: AnnData | MuData,
    *,
    keys: Sequence[str | None],
    mapping: FeatureSpec | None,
    metadata_columns: list[str],
    variable_keys: list[str],
    axis: Literal[0, 1] = 0,
) -> None:
    """
    Route `keys` and `mapping` references into metadata vs variable buckets.

    Mutates both `metadata_columns` and `variable_keys` in place so callers
    can pass them directly to `build_frame` for a minimal materialised frame.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    keys : Sequence[str | None]
        Explicit string references (e.g., the color key, x/y keys).
        `None` entries are skipped.
    mapping : FeatureSpec | None
        A `lets_plot.aes(...)` result. Any string value in its dict is
        treated as a column reference; non-string values are skipped.
    metadata_columns : list[str]
        Mutated in place: references that match observation metadata columns
        (axis=0) or variable metadata columns (axis=1) are appended.
    variable_keys : list[str]
        Mutated in place: references that match variable names (gene names)
        are appended. Only relevant for axis=0; ignored for axis=1.
    axis : {0, 1}, default=0
        The axis the frame is being built for.
    """
    container = _container(data)
    metadata_pool = (
        set(container.observation_columns()) if axis == 0 else set(container.variable_columns())
    )
    candidates: list[str] = []
    for key in keys:
        if isinstance(key, str):
            candidates.append(key)
    if mapping is not None:
        for value in mapping.as_dict().values():
            if isinstance(value, str):
                candidates.append(value)

    # Metadata is tested first, so a modality-qualified metadata column such as
    # `rna:leiden` resolves as a column and never reaches variable resolution.
    for candidate in candidates:
        if candidate in metadata_pool:
            if candidate not in metadata_columns:
                metadata_columns.append(candidate)
        elif axis == 0 and container.owns_variable(candidate) and candidate not in variable_keys:
            variable_keys.append(candidate)


def _is_observation_feature(data: AnnData | MuData, key: str | None) -> bool:
    """Check whether the key is in observations axis (axis=0)."""
    if key is None:
        return False

    container = _container(data)
    return key in container.observation_columns() or container.owns_variable(key)


def _are_observation_features(data: AnnData | MuData, keys: Sequence[str] | None) -> bool:
    """Check whether all the keys are in observations axis (axis=0)."""
    if keys is None:
        return False

    container = _container(data)
    columns = container.observation_columns()
    return all((key in columns or container.owns_variable(key)) for key in keys)


def _is_variable_feature(data: AnnData | MuData, key: str | None) -> bool:
    """Check whether the key is in variable axis (axis=1)."""
    if key is None:
        return False

    return key in _container(data).variable_columns()


def _are_variable_features(data: AnnData | MuData, keys: Sequence[str] | None) -> bool:
    """Check whether all the keys are in variable axis (axis=1)."""
    if keys is None:
        return False

    columns = _container(data).variable_columns()
    return all(key in columns for key in keys)


def _resolve_embedding_key(
    data: AnnData | MuData,
    dimensions: str,
    use_key: str | None,
    xy: Sequence[int],
) -> str:
    """
    Resolve the column-name prefix used to index an embedding in the built frame.

    Parameters
    ----------
    data : AnnData | MuData
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

    embeddings = _container(data).observation_embeddings()
    candidates = list(embeddings.keys())
    shapes = {key: embeddings[key].shape[1] for key in candidates}

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
    data: AnnData | MuData,
    keys: str | Sequence[str],
    companions: Sequence[str | None] = (),
) -> Literal[0, 1]:
    """
    Determine the axis based on the given key or keys.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    keys : str | Sequence[str]
        The keys the axis is being determined for.
    companions : Sequence[str | None], default=()
        Other column references from the same call, such as the grouping and
        aesthetic keys. Used only to break a tie. `None` entries are ignored.

    Notes
    -----
    A name can sit on both axes: quality-control metrics like `total_counts`
    are commonly stored per observation *and* per variable, so a key naming one
    is genuinely ambiguous. When the same call also references a column that
    belongs to only one axis, that settles it. With nothing to break the tie the
    variables axis wins, as it always has.
    """
    if isinstance(keys, str):
        keys = [keys]
    _container(data)  # reject unsupported data objects up front
    on_variables = _are_variable_features(data, keys)
    on_observations = _are_observation_features(data, keys)

    if on_variables and on_observations:
        hints = [companion for companion in companions if companion is not None]
        if hints:
            hints_on_variables = _are_variable_features(data, hints)
            hints_on_observations = _are_observation_features(data, hints)
            if hints_on_observations and not hints_on_variables:
                return 0
            if hints_on_variables and not hints_on_observations:
                return 1
        return 1

    if on_variables:
        return 1
    if on_observations:
        return 0

    msg = f"Could not determine the axis with given keys ({keys})."
    raise ValueError(msg)
