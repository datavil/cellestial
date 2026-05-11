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


# Bracket geometry is scaled in data coordinates, but label text stays in
# fixed visual units so it remains readable across tall heatmaps and short
# matrix/dot/violin variants.
_BAR_OFFSET_FRACTION = 0.030
_TIP_LENGTH_FRACTION = 0.018
_LABEL_GAP_FRACTION = 0.012
_LABEL_PADDING_BASE_FRACTION = 0.060
_LABEL_PADDING_PER_CHAR_SIZE_FRACTION = 0.0051
_LABEL_PADDING_MIN_DENSITY_SCALE = 0.45
_LABEL_PADDING_FULL_DENSITY_KEYS = 30
_LABEL_PADDING_MAX_TARGET_FRACTION = 0.650
_LABEL_TEXT_SIZE_MAX = 6.0
_LABEL_TEXT_SIZE_MIN = 3.75
_LABEL_TEXT_SIZE_PER_KEY = 0.06
_LABEL_TEXT_SIZE_PER_EXTRA_CHAR = 0.08
_LABEL_TEXT_SIZE_CHAR_THRESHOLD = 16


def _resolve_padding(
    groups: dict[str, list[str]],
    *,
    padding: float | None,
    data_range: float | None = None,
    scale: float = 1.0,
) -> float:
    """
    Compute y-axis padding for key labels.

    The auto value sizes the bracket+label area as a fraction of the total y
    axis span. The longest group label drives the rotated-text room.
    ``padding`` overrides the auto value when given; ``data_range`` is the data
    axis span before padding is added.
    """
    if padding is not None:
        return padding
    max_chars = max((len(label) for label in groups), default=0)
    n_keys = sum(len(values) for values in groups.values())
    label_size = _key_groups_label_size(groups)
    density_scale = max(
        _LABEL_PADDING_MIN_DENSITY_SCALE,
        min(1.0, n_keys / _LABEL_PADDING_FULL_DENSITY_KEYS),
    )
    has_data_range = data_range is not None and data_range > 0
    span = data_range if has_data_range else 5.0
    target_fraction = min(
        _BAR_OFFSET_FRACTION
        + _LABEL_GAP_FRACTION
        + _LABEL_PADDING_BASE_FRACTION
        + _LABEL_PADDING_PER_CHAR_SIZE_FRACTION
        * max_chars
        * label_size
        * density_scale,
        _LABEL_PADDING_MAX_TARGET_FRACTION,
    )
    resolved = target_fraction * span / (1.0 - target_fraction) * scale
    if not has_data_range:
        return max(1.5, resolved)
    return resolved


def _build_key_groups_frame(
    groups: dict[str, list[str]],
    *,
    y: float,
) -> pl.DataFrame:
    """
    Build the ``geom_bracket`` data frame spanning column groups on the variable axis.

    For a group of ``N >= 2`` keys at column indices ``first..last``, the bracket
    spans ``first..last`` (width ``N - 1``, ends at key centers). For a singleton
    group at index ``p``, the bracket spans ``p - 0.4..p + 0.4`` (width ``0.8``)
    so the bar remains visible.
    """
    xmins: list[float] = []
    xmaxs: list[float] = []
    labels: list[str] = []
    cursor = 0
    for label, values in groups.items():
        first = cursor
        last = cursor + len(values) - 1
        # expand the bracket
        xmins.append(float(first) - 0.4)
        xmaxs.append(float(last) + 0.4)
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


def _key_groups_label_size(groups: dict[str, list[str]]) -> float:
    """Text size for group labels, reduced slightly for dense key layouts."""
    n_keys = sum(len(values) for values in groups.values())
    max_chars = max((len(label) for label in groups), default=0)
    extra_chars = max(0, max_chars - _LABEL_TEXT_SIZE_CHAR_THRESHOLD)
    return max(
        _LABEL_TEXT_SIZE_MIN,
        min(
            _LABEL_TEXT_SIZE_MAX,
            _LABEL_TEXT_SIZE_MAX
            - _LABEL_TEXT_SIZE_PER_KEY * n_keys
            - _LABEL_TEXT_SIZE_PER_EXTRA_CHAR * extra_chars,
        ),
    )


def _key_groups_layers(
    groups: dict[str, list[str]],
    *,
    y: float,
    total_span: float,
    color: str = "black",
    size: float = 1.0,
    extra_kwargs: dict | None = None,
) -> list[FeatureSpec]:
    """
    Build the layers that annotate column groups above the plot.

    Each bracket is one connected ``geom_path`` (left tip down, top bar across,
    right tip down) so the corners join cleanly with no sub-pixel gap. A
    vertical label is centered on each bracket as a separate ``geom_text``.
    ``total_span`` is the full y axis range (data + padding) used to scale tip
    length and label offset proportionally.
    """
    frame = _build_key_groups_frame(groups, y=y)
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
    text_kwargs.update(extra_kwargs or {})

    return [
        geom_path(
            data=path_frame,
            mapping=aes(x="x", y="y", group="g"),
            color=color,
            size=size,
        ),
        geom_text(
            data=label_frame,
            mapping=aes(x="x", y="y", label="label"),
            color=color,
            size=_key_groups_label_size(groups),
            **text_kwargs,
        ),
    ]
