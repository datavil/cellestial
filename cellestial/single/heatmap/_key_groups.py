from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import polars as pl
from lets_plot import aes, geom_path, geom_text

from cellestial.util.errors import DuplicateKeysError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec


def _resolve_key_groups(
    keys: Sequence[str] | Mapping[str, Sequence[str]],
    *,
    key_labels: bool = True,
) -> tuple[list[str], dict[str, list[str]] | None]:
    """
    Flatten ``keys`` if it is a mapping of group label to keys.

    Returns
    -------
    flat_keys : list[str]
        Concatenated list of keys preserving group order.
    groups : dict[str, list[str]] | None
        Group label to keys, or ``None`` when ``keys`` was already a flat sequence.

    Raises
    ------
    DuplicateKeysError
        If the same key appears in more than one group.
    """
    if not isinstance(keys, Mapping):
        return list(keys), None

    grouped_keys = cast("Mapping[str, Sequence[str]]", keys)
    groups: dict[str, list[str]] = {}
    for label, values in grouped_keys.items():
        if isinstance(values, str):
            msg = (
                f"Keys for group {label!r} must be a sequence of strings, "
                "not a single string."
            )
            raise TypeError(msg)
        groups[label] = list(values)
    flat: list[str] = []
    location: dict[str, str] = {}
    for label, values in groups.items():
        for key in values:
            if key in location:
                msg = (
                    f"Key {key!r} appears in multiple groups: "
                    f"{location[key]!r} and {label!r}. "
                    "Each key must belong to a single group."
                )
                raise DuplicateKeysError(msg)
            location[key] = label
            flat.append(key)
    return flat, groups


# Bracket geometry. Most callers express the bar offset, tip length, and label
# gap as a fraction of the y-axis total span so they scale with the plot.
_BAR_OFFSET_FRACTION = 0.030
_TIP_LENGTH_FRACTION = 0.018
_LABEL_GAP_FRACTION = 0.012

# ---------------------------------------------------------------------------
# Label sizing has two modes:
#
# - "absolute" (default, used by heatmap): label_size is in lets-plot's default
#   text-size unit (~pt-like). Works well when no other scale_size interferes
#   with the geom_text. Padding is computed as a fraction of the total y-axis
#   span using the original heuristic.
#
# - "y" (used by dotplot and stacked_violin): label_size is in y-axis units
#   via ``geom_text(size_unit="y")``. This bypasses the silent shrinkage that
#   ``scale_size(range=[...])`` applies to ``geom_text`` size constants in
#   plots that map a size aesthetic. Padding is computed directly from the
#   rotated text extent in y-axis units.
#
# Each mode has its own size formula and padding formula below.
# ---------------------------------------------------------------------------

# Absolute mode (heatmap).
_ABS_LABEL_PADDING_BASE_FRACTION = 0.060
_ABS_LABEL_PADDING_PER_CHAR_SIZE_FRACTION = 0.0051
_ABS_LABEL_PADDING_MIN_DENSITY_SCALE = 0.45
_ABS_LABEL_PADDING_FULL_DENSITY_KEYS = 30
_ABS_LABEL_PADDING_MAX_TARGET_FRACTION = 0.650
_ABS_LABEL_TEXT_SIZE_MAX = 6.0
_ABS_LABEL_TEXT_SIZE_MIN = 3.75
_ABS_LABEL_TEXT_SIZE_PER_KEY = 0.06
_ABS_LABEL_TEXT_SIZE_PER_EXTRA_CHAR = 0.08
_ABS_LABEL_TEXT_SIZE_CHAR_THRESHOLD = 16

# y-unit mode (dotplot, stacked_violin).
# Empirically calibrated: cap height ~ 22 px on a typical 6-inch panel when
# size = 0.037 * total_span y units. Each rotated character occupies roughly
# ``_Y_CHAR_WIDTH_PER_SIZE * label_size`` y units along the y axis.
_Y_LABEL_VISUAL_CAP_FRACTION = 0.037
_Y_LABEL_LONG_THRESHOLD = 12
_Y_LABEL_LONG_MIN_FRACTION = 0.6
_Y_LABEL_SIZE_FLOOR = 0.10
_Y_CHAR_WIDTH_PER_SIZE = 0.40
_Y_PADDING_SAFETY_FRACTION = 0.030


def _label_size_absolute(
    groups: dict[str, list[str]], *, scale: float = 1.0
) -> float:
    """Label size in absolute lets-plot text units (used without ``size_unit``)."""
    n_keys = sum(len(values) for values in groups.values())
    max_chars = max((len(label) for label in groups), default=0)
    extra_chars = max(0, max_chars - _ABS_LABEL_TEXT_SIZE_CHAR_THRESHOLD)
    size = max(
        _ABS_LABEL_TEXT_SIZE_MIN,
        min(
            _ABS_LABEL_TEXT_SIZE_MAX,
            _ABS_LABEL_TEXT_SIZE_MAX
            - _ABS_LABEL_TEXT_SIZE_PER_KEY * n_keys
            - _ABS_LABEL_TEXT_SIZE_PER_EXTRA_CHAR * extra_chars,
        ),
    )
    return max(_ABS_LABEL_TEXT_SIZE_MIN, size * scale)


def _label_size_y_units(
    groups: dict[str, list[str]], *, span: float, scale: float = 1.0
) -> float:
    """
    Label size in y-axis units (used with ``geom_text(size_unit='y')``).

    Scales with ``span`` so the rendered visual size stays roughly constant
    across plots with different y-axis spans. Long labels are shrunk slightly
    to keep the bracket area from dominating the plot.
    """
    max_chars = max((len(label) for label in groups), default=0)
    visual_fraction = _Y_LABEL_VISUAL_CAP_FRACTION
    if max_chars > _Y_LABEL_LONG_THRESHOLD:
        visual_fraction *= max(
            _Y_LABEL_LONG_MIN_FRACTION, _Y_LABEL_LONG_THRESHOLD / max_chars
        )
    return max(_Y_LABEL_SIZE_FLOOR, visual_fraction * span * scale)


def _resolve_padding(
    groups: dict[str, list[str]],
    *,
    padding: float | None,
    data_range: float | None = None,
    scale: float = 1.0,
    label_size_scale: float = 1.0,
    size_unit: str | None = None,
) -> float:
    """
    Compute y-axis padding (in data units) for the bracket area.

    ``padding`` overrides the auto value when given. ``scale`` is a per-plot
    multiplier applied to the auto padding (use ``<1`` for a more compact
    bracket area). ``size_unit`` selects the sizing mode (``None`` for the
    absolute heatmap mode, ``"y"`` for the y-unit dotplot/stacked_violin mode).
    """
    if padding is not None:
        return padding
    data = data_range if data_range is not None and data_range > 0 else 5.0
    max_chars = max((len(label) for label in groups), default=0)

    if size_unit is None:
        n_keys = sum(len(values) for values in groups.values())
        label_size = _label_size_absolute(groups, scale=label_size_scale)
        density_scale = max(
            _ABS_LABEL_PADDING_MIN_DENSITY_SCALE,
            min(1.0, n_keys / _ABS_LABEL_PADDING_FULL_DENSITY_KEYS),
        )
        target_fraction = min(
            _BAR_OFFSET_FRACTION
            + _LABEL_GAP_FRACTION
            + _ABS_LABEL_PADDING_BASE_FRACTION
            + _ABS_LABEL_PADDING_PER_CHAR_SIZE_FRACTION
            * max_chars
            * label_size
            * density_scale,
            _ABS_LABEL_PADDING_MAX_TARGET_FRACTION,
        )
        resolved = target_fraction * data / (1.0 - target_fraction) * scale
        if data_range is None or data_range <= 0:
            return max(1.5, resolved)
        return resolved

    # y-unit mode: padding fits the rotated text extent directly in y units.
    approx_total_span = data * 1.3
    label_size = _label_size_y_units(
        groups, span=approx_total_span, scale=label_size_scale
    )
    text_extent = max_chars * _Y_CHAR_WIDTH_PER_SIZE * label_size
    bar_offset = _BAR_OFFSET_FRACTION * data
    label_gap = _LABEL_GAP_FRACTION * data
    safety = _Y_PADDING_SAFETY_FRACTION * data
    return (bar_offset + label_gap + text_extent + safety) * scale


def _build_key_groups_frame(
    groups: dict[str, list[str]],
    *,
    y: float,
    width: float = 0.6,
) -> pl.DataFrame:
    """
    Build the ``geom_bracket`` data frame spanning column groups on the variable axis.

    ``width`` is the bracket width for a singleton group in column units. For a
    group of ``N >= 2`` keys at column indices ``first..last``, the bracket
    extends ``width / 2`` past the first and last key center on each side.
    """
    extension = width / 2
    xmins: list[float] = []
    xmaxs: list[float] = []
    labels: list[str] = []
    cursor = 0
    for label, values in groups.items():
        first = cursor
        last = cursor + len(values) - 1
        xmins.append(float(first) - extension)
        xmaxs.append(float(last) + extension)
        labels.append(label)
        cursor += len(values)
    return pl.DataFrame(
        {
            "xmin": xmins,
            "xmax": xmaxs,
            "y": [float(y)] * len(labels),
            "label": labels,
        }
    )


def _key_groups_bar_y(top_edge: float, *, total_span: float = 5.0) -> float:
    """Y position of the bracket bar above the plot top edge."""
    return top_edge + _BAR_OFFSET_FRACTION * total_span


def _key_groups_layers(
    groups: dict[str, list[str]],
    *,
    y: float,
    total_span: float,
    text_color: str = "black",
    bracket_color: str = "black",
    bracket_size: float = 0.6,
    width: float = 0.6,
    label_size_scale: float = 1.0,
    size_unit: str | None = None,
    extra_kwargs: dict | None = None,
) -> list[FeatureSpec]:
    """
    Build the layers that annotate column groups above the plot.

    Each bracket is one connected ``geom_path`` (left tip down, top bar across,
    right tip down) so the corners join cleanly with no sub-pixel gap. A
    vertical label is centered on each bracket as a separate ``geom_text``.

    ``size_unit`` selects the label sizing mode (``None`` for absolute units,
    ``"y"`` to express the size in y-axis units). The y-unit mode is used by
    dotplot and stacked_violin to bypass ``scale_size`` interference; the
    absolute mode is used by heatmap.
    """
    frame = _build_key_groups_frame(groups, y=y, width=width)
    tip_length = _TIP_LENGTH_FRACTION * total_span
    label_gap = _LABEL_GAP_FRACTION * total_span

    path_records: list[dict] = []
    label_records: list[dict] = []
    for group_index, row in enumerate(frame.iter_rows(named=True)):
        path_records.extend(
            [
                {"x": row["xmin"], "y": y - tip_length, "g": group_index},
                {"x": row["xmin"], "y": y, "g": group_index},
                {"x": row["xmax"], "y": y, "g": group_index},
                {"x": row["xmax"], "y": y - tip_length, "g": group_index},
            ]
        )
        label_records.append(
            {
                "x": (row["xmin"] + row["xmax"]) / 2,
                "y": y + label_gap,
                "label": row["label"],
            }
        )

    path_frame = pl.DataFrame(path_records)
    label_frame = pl.DataFrame(label_records)

    text_kwargs: dict = {
        "angle": 90,
        "hjust": 0,
        "vjust": 0.5,
    }
    if size_unit is not None:
        text_kwargs["size_unit"] = size_unit
    text_kwargs.update(extra_kwargs or {})

    if size_unit is None:
        label_size = _label_size_absolute(groups, scale=label_size_scale)
    else:
        label_size = _label_size_y_units(
            groups, span=total_span, scale=label_size_scale
        )

    return [
        geom_path(
            data=path_frame,
            mapping=aes(x="x", y="y", group="g"),
            color=bracket_color,
            size=bracket_size,
        ),
        geom_text(
            data=label_frame,
            mapping=aes(x="x", y="y", label="label"),
            color=text_color,
            size=label_size,
            **text_kwargs,
        ),
    ]
