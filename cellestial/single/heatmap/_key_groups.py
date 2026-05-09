from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast
from warnings import warn

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

    if key_labels:
        warn(
            "key labels on top of the plot are not stable yet "
            "and may behave abruptly. "
            "Consider setting `key_labels=False`.",
            stacklevel=2,
        )
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


_BAR_OFFSET = 0.4
_TIP_LENGTH = 0.25
_LABEL_GAP = 0.2
_LABEL_SIZE = 10.0
# ``size_unit='x'`` is used so text size is in x-axis (column-width) units.
# At ``size = 1.0`` (col-width) one rotated character takes ~``_CHAR_HEIGHT_FACTOR``
# y-units of vertical height when y-axis range matches x-axis range.
_CHAR_HEIGHT_FACTOR = 0.6
_LABEL_SIZE_SCALE = 10.0


def _resolve_padding(
    groups: dict[str, list[str]],
    *,
    padding: float | None,
) -> float:
    """
    Compute y-axis padding for key labels.

    The auto value fits the bracket bar plus the rotated label of the longest
    group. ``padding`` overrides the auto value when given.
    """
    if padding is not None:
        return padding
    internal_size = _LABEL_SIZE / _LABEL_SIZE_SCALE
    max_chars = max((len(label) for label in groups), default=0)
    char_height = internal_size * _CHAR_HEIGHT_FACTOR
    return max(1.5, _BAR_OFFSET + _LABEL_GAP + max_chars * char_height + 0.3)


def _build_key_groups_frame(
    groups: dict[str, list[str]],
    *,
    y: float,
) -> pl.DataFrame:
    """
    Build the ``geom_bracket`` data frame spanning column groups on the variable axis.

    For a group of ``N >= 2`` keys at column indices ``first..last``, the bracket
    spans ``first..last`` (width ``N - 1``, ends at key centers). For a singleton
    group at index ``p``, the bracket spans ``p - 0.25..p + 0.25`` (width ``0.5``)
    so the bar remains visible.
    """
    xmins: list[float] = []
    xmaxs: list[float] = []
    labels: list[str] = []
    cursor = 0
    for label, values in groups.items():
        first = cursor
        last = cursor + len(values) - 1
        if len(values) == 1:
            xmins.append(float(first) - 0.4)
            xmaxs.append(float(last) + 0.4)
        else:
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


def _key_groups_bar_y(top_edge: float) -> float:
    """Y position of the bracket bar above the plot top edge."""
    return top_edge + _BAR_OFFSET


def _key_groups_layers(
    groups: dict[str, list[str]],
    *,
    y: float,
    color: str = "black",
    size: float = 1.0,
    extra_kwargs: dict | None = None,
) -> list[FeatureSpec]:
    """
    Build the layers that annotate column groups above the plot.

    Each bracket is one connected ``geom_path`` (left tip down, top bar across,
    right tip down) so the corners join cleanly with no sub-pixel gap. A
    vertical label is centered on each bracket as a separate ``geom_text``.
    """
    frame = _build_key_groups_frame(groups, y=y)

    path_records: list[dict] = []
    label_records: list[dict] = []
    for group_index, row in enumerate(frame.iter_rows(named=True)):
        path_records.extend(
            [
                {"x": row["xmin"], "y": y - _TIP_LENGTH, "g": group_index},
                {"x": row["xmin"], "y": y, "g": group_index},
                {"x": row["xmax"], "y": y, "g": group_index},
                {"x": row["xmax"], "y": y - _TIP_LENGTH, "g": group_index},
            ]
        )
        label_records.append(
            {
                "x": (row["xmin"] + row["xmax"]) / 2,
                "y": y + _LABEL_GAP,
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
            size=1,
            size_unit="max",
            family="mono",
            **text_kwargs,
        ),
    ]
