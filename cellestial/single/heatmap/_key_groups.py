from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import polars as pl
from lets_plot import aes, geom_segment, geom_text

from cellestial.util.errors import DuplicateKeysError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec


def _resolve_key_groups(
    keys: Sequence[str] | Mapping[str, Sequence[str]],
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

    groups = {label: list(values) for label, values in keys.items()}
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


_BAR_OFFSET = 0.25
_TIP_LENGTH = 0.15
_LABEL_GAP = 0.1
_DEFAULT_LABEL_SIZE = 0.7
# Empirical: with `size_unit='x'`, one rotated character takes roughly
# `_CHAR_HEIGHT_FACTOR * label_size` y-units of vertical height.
_CHAR_HEIGHT_FACTOR = 0.18


def _resolve_padding(
    groups: dict[str, list[str]],
    *,
    padding: float | None,
    label_size: float = _DEFAULT_LABEL_SIZE,
) -> float:
    """
    Compute the y-axis padding (in row units) above the plot top edge to fit
    the bracket bar plus the rotated label of the longest group. ``padding``
    overrides the auto value when given. ``label_size`` is the bracket label's
    size aesthetic (in column-width units, since ``size_unit='x'`` is used).
    """
    if padding is not None:
        return padding
    max_chars = max((len(label) for label in groups), default=0)
    char_height = max(0.2, label_size * _CHAR_HEIGHT_FACTOR)
    return max(
        1.5, _BAR_OFFSET + _LABEL_GAP + max_chars * char_height + 0.3
    )


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
            xmins.append(float(first) - 0.25)
            xmaxs.append(float(last) + 0.25)
        else:
            xmins.append(float(first))
            xmaxs.append(float(last))
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
    label_size: float = _DEFAULT_LABEL_SIZE,
    extra_kwargs: dict | None = None,
) -> list[FeatureSpec]:
    """
    Build the layers that annotate column groups above the plot.

    Returns a horizontal bar with downward tips at each end (drawn with
    ``geom_segment``) and a vertical label centered on each bar (drawn with
    ``geom_text``). The label is rendered as a separate layer so it can be
    centered on the bracket midpoint regardless of bar width.
    """
    frame = _build_key_groups_frame(groups, y=y)

    bar_records: list[dict] = []
    tip_records: list[dict] = []
    label_records: list[dict] = []
    for row in frame.iter_rows(named=True):
        bar_records.append(
            {"x": row["xmin"], "xend": row["xmax"], "y": y, "yend": y}
        )
        tip_records.append(
            {"x": row["xmin"], "xend": row["xmin"], "y": y, "yend": y - _TIP_LENGTH}
        )
        tip_records.append(
            {"x": row["xmax"], "xend": row["xmax"], "y": y, "yend": y - _TIP_LENGTH}
        )
        label_records.append(
            {
                "x": (row["xmin"] + row["xmax"]) / 2,
                "y": y + _LABEL_GAP,
                "label": row["label"],
            }
        )

    segment_frame = pl.DataFrame(bar_records + tip_records)
    label_frame = pl.DataFrame(label_records)

    text_kwargs: dict = {
        "angle": 90,
        "hjust": 0,
        "vjust": 0.5,
        "size_unit": "x",
    }
    text_kwargs.update(extra_kwargs or {})

    return [
        geom_segment(
            data=segment_frame,
            mapping=aes(x="x", xend="xend", y="y", yend="yend"),
            color=color,
            size=size,
        ),
        geom_text(
            data=label_frame,
            mapping=aes(x="x", y="y", label="label"),
            color=color,
            size=label_size,
            **text_kwargs,
        ),
    ]
