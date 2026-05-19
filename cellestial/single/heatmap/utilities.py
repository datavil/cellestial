from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import polars as pl
from anndata import AnnData
from lets_plot import aes, geom_path, geom_text
from scipy.stats import gaussian_kde

from cellestial.util import _warn
from cellestial.util.errors import DuplicateKeysError, KeyNotFoundError, UnsupportedDataTypeError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec


# ---------------------------------------------------------------------------
# heatmap.py helpers
# ---------------------------------------------------------------------------

_GROUP_BAR_RATIO = 0.02
_GROUP_BAR_GAP_RATIO = 0.025
_GROUP_BAR_GAP_MIN = 0.15


def _bin_within_groups(
    frame: pl.DataFrame,
    *,
    observations_name: str,
    group_by: str,
    variable_column: str,
    value_column: str,
    y_order_groups: list[str],
    max_rows: int,
) -> pl.DataFrame:
    """
    Collapse observations into at most ``max_rows`` virtual rows, within each group.

    Cells within a group are sorted by observation id and chunked into contiguous
    bins; each bin's expression value is the per-variable mean of its members.
    Group boundaries are preserved, so group bars/lines remain meaningful.
    Returns ``frame`` unchanged when the unique observation count fits in
    ``max_rows``.
    """
    group_key = "_group_key"
    frame_with_key = frame.with_columns(pl.col(group_by).cast(pl.String).alias(group_key))
    obs_per_group = (
        frame_with_key.select(observations_name, group_key)
        .unique()
        .group_by(group_key, maintain_order=False)
        .agg(pl.len().alias("_n_cells"))
    )
    total = int(obs_per_group["_n_cells"].sum())
    if total <= max_rows:
        return frame

    obs_per_group = obs_per_group.with_columns(
        pl.max_horizontal(
            pl.lit(1, dtype=pl.Int64),
            (pl.col("_n_cells") * max_rows / total).round().cast(pl.Int64),
        ).alias("_bin_count")
    )

    group_index_frame = pl.DataFrame(
        {
            group_key: y_order_groups,
            "_group_index": list(range(len(y_order_groups))),
        }
    )
    offsets_frame = (
        obs_per_group.join(group_index_frame, on=group_key)
        .sort("_group_index")
        .with_columns((pl.col("_bin_count").cum_sum() - pl.col("_bin_count")).alias("_bin_offset"))
        .select(group_key, "_bin_count", "_n_cells", "_bin_offset", "_group_index")
    )

    obs_frame = (
        frame_with_key.select(observations_name, group_key)
        .unique(maintain_order=True)
        .join(offsets_frame, on=group_key)
        .sort(["_group_index", observations_name])
        .with_columns(pl.int_range(pl.len()).over(group_key).alias("_rank_in_group"))
        .with_columns(
            (
                (pl.col("_rank_in_group") * pl.col("_bin_count") // pl.col("_n_cells"))
                + pl.col("_bin_offset")
            )
            .cast(pl.String)
            .alias("_binned_obs")
        )
        .select(observations_name, "_binned_obs")
    )

    return (
        frame.join(obs_frame, on=observations_name)
        .drop(observations_name)
        .rename({"_binned_obs": observations_name})
        .group_by(observations_name, group_by, variable_column, maintain_order=False)
        .agg(pl.col(value_column).mean())
    )


def _scale_values(frame: pl.DataFrame, *, value_column: str, partition_key: str) -> pl.DataFrame:
    """Min-max scale `value_column` within partitions defined by `partition_key`."""
    value = pl.col(value_column)
    value_min = value.min().over(partition_key)
    value_max = value.max().over(partition_key)
    value_range = value_max - value_min
    return frame.with_columns(
        pl.when(value_range == 0)
        .then(0.0)
        .otherwise((value - value_min) / value_range)
        .alias(value_column)
    )


def _group_bar_gap(n_x: int) -> float:
    """Return a stable visual gap between the heatmap and group bars."""
    return max(_GROUP_BAR_GAP_MIN, n_x * _GROUP_BAR_GAP_RATIO)


def _assign_positions(
    frame: pl.DataFrame,
    *,
    aggregate: bool,
    group_by: str,
    observations_name: str,
    variable_column: str,
    x_keys: list[str],
    y_order_groups: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame | None, int, int, list[float]]:
    """
    Attach `_x`/`_y` to `frame` and compute layout metadata.

    Returns
    -------
    frame : pl.DataFrame
        Frame with `_x` and `_y` columns.
    cell_frame : pl.DataFrame | None
        Per-cell layout frame (None when aggregating).
    n_x : int
    n_y : int
    group_centers : list[float]
        Y center of each group, in `y_order_groups` order.
    """
    n_x = len(x_keys)
    x_pos = {k: i for i, k in enumerate(x_keys)}
    frame = frame.with_columns(
        pl.col(variable_column).replace_strict(x_pos, return_dtype=pl.Float64).alias("position_x")
    )

    if aggregate:
        n_y = len(y_order_groups)
        y_pos = {g: i for i, g in enumerate(y_order_groups)}
        frame = frame.with_columns(
            pl.col(group_by)
            .cast(pl.String)
            .replace_strict(y_pos, return_dtype=pl.Float64)
            .alias("position_y"),
        )
        return frame, None, n_x, n_y, [float(i) for i in range(n_y)]

    # non-aggregate: rescale per-cell _y to span [0, n_x] for square-ish aspect
    group_index = {g: i for i, g in enumerate(y_order_groups)}
    cell_frame = (
        frame.select(observations_name, group_by)
        .unique()
        .with_columns(
            pl.col(group_by)
            .cast(pl.String)
            .replace_strict(group_index, return_dtype=pl.Int64)
            .alias("_group_index")
        )
        .sort(["_group_index", observations_name])
    )
    n_y = cell_frame.height
    y_step = (n_x - 1) / max(n_y - 1, 1)
    cell_frame = cell_frame.with_columns(
        (pl.int_range(pl.len()).cast(pl.Float64) * y_step).alias("position_y")
    )
    frame = frame.join(cell_frame.select(observations_name, "position_y"), on=observations_name)
    centers_frame = (
        cell_frame.group_by(group_by, maintain_order=False)
        .agg(pl.col("position_y").mean().alias("_center"), pl.col("_group_index").first())
        .sort("_group_index")
    )
    return frame, cell_frame, n_x, n_y, centers_frame["_center"].to_list()


def _get_group_bar_frame(cell_frame: pl.DataFrame, *, group_by: str, n_x: int) -> pl.DataFrame:
    """Build the per-group colored bar frame drawn to the left of the heatmap."""
    bar_width = max(1.0, n_x * _GROUP_BAR_RATIO)
    bar_xend = -_group_bar_gap(n_x)
    bar_x = bar_xend - bar_width
    bar_x_mid = (bar_x + bar_xend) / 2
    bar_frame = (
        cell_frame.group_by(group_by, maintain_order=False)
        .agg(
            pl.col("position_y").min().alias("y_min"),
            pl.col("position_y").max().alias("y_max"),
            pl.col("_group_index").first(),
        )
        .sort("_group_index")
        .with_columns(pl.lit(bar_x_mid).alias("x"))
    )
    return bar_frame


def _get_group_lines_frame(
    cell_frame: pl.DataFrame | None,
    *,
    aggregate: bool,
    group_by: str,
    n_x: int,
    n_y: int,
    n_groups: int,
) -> pl.DataFrame:
    """Build the horizontal separator-line frame between groups."""
    if aggregate:
        line_ys = [i + 0.5 for i in range(n_groups - 1)]
    else:
        assert cell_frame is not None
        boundaries = (
            cell_frame.group_by(group_by, maintain_order=False)
            .agg(pl.col("position_y").max().alias("y_max"), pl.col("_group_index").first())
            .sort("_group_index")
            .head(n_groups - 1)["y_max"]
            .to_list()
        )
        half_step = (n_x - 1) / max(n_y - 1, 1) / 2
        line_ys = [y + half_step for y in boundaries]

    x_start = -0.5
    x_end = n_x - 0.5
    return pl.DataFrame(
        {
            "x": [x_start] * len(line_ys),
            "xend": [x_end] * len(line_ys),
            "y": line_ys,
            "yend": line_ys,
        }
    )


# ---------------------------------------------------------------------------
# key_groups helpers (formerly _key_groups.py)
# ---------------------------------------------------------------------------

# Bracket geometry. Most callers express the bar offset, tip length, and label
# gap as a fraction of the y-axis total span so they scale with the plot.
_BAR_OFFSET_FRACTION = 0.030
_TIP_LENGTH_FRACTION = 0.018
_LABEL_GAP_FRACTION = 0.012

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


def _resolve_key_groups(
    keys: Sequence[str] | Mapping[str, Sequence[str]],
    *,
    key_labels: bool = True,
) -> tuple[list[str], dict[str, list[str]] | None]:
    """
    Flatten `keys` if it is a mapping of group label to keys.

    Returns
    -------
    flat_keys : list[str]
        Concatenated list of keys preserving group order.
    groups : dict[str, list[str]] | None
        Group label to keys, or `None` when `keys` was already a flat sequence.

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
            msg = f"Keys for group {label!r} must be a sequence of strings, not a single string."
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


def _label_size_absolute(groups: dict[str, list[str]], *, scale: float = 1.0) -> float:
    """Label size in absolute lets-plot text units (used without `size_unit`)."""
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


def _label_size_y_units(groups: dict[str, list[str]], *, span: float, scale: float = 1.0) -> float:
    """
    Label size in y-axis units (used with `geom_text(size_unit='y')`).

    Scales with `span` so the rendered visual size stays roughly constant
    across plots with different y-axis spans. Long labels are shrunk slightly
    to keep the bracket area from dominating the plot.
    """
    max_chars = max((len(label) for label in groups), default=0)
    visual_fraction = _Y_LABEL_VISUAL_CAP_FRACTION
    if max_chars > _Y_LABEL_LONG_THRESHOLD:
        visual_fraction *= max(_Y_LABEL_LONG_MIN_FRACTION, _Y_LABEL_LONG_THRESHOLD / max_chars)
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

    `padding` overrides the auto value when given. `scale` is a per-plot
    multiplier applied to the auto padding (use `<1` for a more compact
    bracket area). `size_unit` selects the sizing mode (`None` for the
    absolute heatmap mode, `"y"` for the y-unit dotplot/stacked_violin mode).
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
            + _ABS_LABEL_PADDING_PER_CHAR_SIZE_FRACTION * max_chars * label_size * density_scale,
            _ABS_LABEL_PADDING_MAX_TARGET_FRACTION,
        )
        resolved = target_fraction * data / (1.0 - target_fraction) * scale
        if data_range is None or data_range <= 0:
            return max(1.5, resolved)
        return resolved

    # y-unit mode: padding fits the rotated text extent directly in y units.
    approx_total_span = data * 1.3
    label_size = _label_size_y_units(groups, span=approx_total_span, scale=label_size_scale)
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
    Build the `geom_bracket` data frame spanning column groups on the variable axis.

    `width` is the bracket width for a singleton group in column units. For a
    group of `N >= 2` keys at column indices `first..last`, the bracket
    extends `width / 2` past the first and last key center on each side.
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
    label_size_span: float | None = None,
    size_unit: str | None = None,
    extra_kwargs: dict | None = None,
) -> list[FeatureSpec]:
    """
    Build the layers that annotate column groups above the plot.

    Each bracket is one connected `geom_path` (left tip down, top bar across,
    right tip down) so the corners join cleanly with no sub-pixel gap. A
    vertical label is centered on each bracket as a separate `geom_text`.

    `size_unit` selects the label sizing mode (`None` for absolute units,
    otherwise express the size in that axis' data units). `label_size_span`
    lets callers size text from a different axis span than the y span used for
    bracket geometry.
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
            groups,
            span=total_span if label_size_span is None else label_size_span,
            scale=label_size_scale,
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


# ---------------------------------------------------------------------------
# rank_genes_groups helpers (formerly _rank_genes_groups.py)
# ---------------------------------------------------------------------------

_DEFAULT_RANK_GENES_KEY = "rank_genes_groups"


def _resolve_rank_genes_groups_key(rank_genes_groups: bool | str) -> str:  # noqa: FBT001
    """Translate the user-facing flag into the `adata.uns` key to read."""
    if rank_genes_groups is True:
        return _DEFAULT_RANK_GENES_KEY
    if isinstance(rank_genes_groups, str):
        return rank_genes_groups
    msg = (
        "`rank_genes_groups` must be True or a string naming the "
        "`adata.uns` key (e.g. 'rank_genes_groups_wilcoxon')."
    )
    raise TypeError(msg)


def _extract_rank_genes_groups(
    data: AnnData,
    *,
    rank_genes_groups: bool | str,
    n_genes: int,
    groups: Sequence[str] | None = None,
) -> tuple[dict[str, list[str]], str]:
    """
    Read top-N ranked genes per group from a precomputed rank result.

    Parameters
    ----------
    data : AnnData
        Source object that already contains a `rank_genes_groups` result.
    rank_genes_groups : bool | str
        `True` reads from the default `rank_genes_groups` key; a string
        reads from the matching custom key (e.g. `rank_genes_groups_wilcoxon`).
    n_genes : int
        Number of top genes to pull per group.
    groups : Sequence[str] | None, default=None
        Subset of groups to include in mapping order. `None` keeps all
        groups in their stored order.

    Returns
    -------
    keys : dict[str, list[str]]
        Mapping from group label to top-N gene names, ready to feed into
        `_resolve_key_groups`.
    group_by : str
        The categorical key used when `rank_genes_groups` was computed.
    """
    if n_genes < 1:
        msg = f"`n_genes` must be >= 1, got {n_genes}."
        raise ValueError(msg)

    uns_key = _resolve_rank_genes_groups_key(rank_genes_groups)

    if isinstance(data, AnnData):
        if uns_key not in data.uns:
            msg = (
                f"`adata.uns[{uns_key!r}]` not found. "
                "Run `scanpy.tl.rank_genes_groups` first "
                "(or pass the correct `key_added` value as `rank_genes_groups`)."
            )
            raise KeyNotFoundError(msg)

        record = data.uns[uns_key]
        if "names" not in record or "params" not in record:
            msg = (
                f"`adata.uns[{uns_key!r}]` does not look like a "
                "`rank_genes_groups` result (missing 'names' or 'params')."
            )
            raise KeyNotFoundError(msg)

        names = record["names"]
        available_groups = list(names.dtype.names)

        if groups is None:
            selected = available_groups
        else:
            selected = list(groups)
            unknown = [group for group in selected if group not in available_groups]
            if unknown:
                msg = (
                    f"Groups {unknown!r} not found in "
                    f"`adata.uns[{uns_key!r}]`. Available: {available_groups!r}."
                )
                raise KeyNotFoundError(msg)

        n_available = len(names)
        if n_genes > n_available:
            msg = (
                f"`n_genes={n_genes}` exceeds the {n_available} ranked genes "
                f"stored in `adata.uns[{uns_key!r}]`."
            )
            raise ValueError(msg)

        keys: dict[str, list[str]] = {}
        seen: dict[str, str] = {}
        dropped: list[tuple[str, str, str]] = []
        for group in selected:
            group_keys: list[str] = []
            for gene in names[group][:n_genes]:
                gene_name = str(gene)
                if gene_name in seen:
                    dropped.append((gene_name, seen[gene_name], group))
                    continue
                seen[gene_name] = group
                group_keys.append(gene_name)
            keys[group] = group_keys

        if dropped:
            examples = ", ".join(
                f"{gene!r} (kept in {first!r}, dropped from {later!r})"
                for gene, first, later in dropped[:3]
            )
            suffix = f" and {len(dropped) - 3} more" if len(dropped) > 3 else ""
            _warn(
                "Some genes ranked highly in multiple groups; "
                "keeping the first occurrence and dropping the rest "
                f"({examples}{suffix}). "
                "Increase `n_genes`, restrict `groups`, "
                "or pass an explicit `keys` mapping to control this."
            )

        params = record["params"]
        stored_group_by = params.get("groupby")
        if stored_group_by is None:
            msg = (
                f"`adata.uns[{uns_key!r}]['params']` is missing 'groupby'; "
                "cannot infer `group_by`."
            )
            raise KeyNotFoundError(msg)

        return keys, str(stored_group_by)

    msg = f"Unsupported data type: `{type(data)}`"
    raise UnsupportedDataTypeError(msg)


def _resolve_rank_genes_groups_args(
    data: AnnData,
    *,
    rank_genes_groups: bool | str,
    n_genes: int,
    groups: Sequence[str] | None,
    keys: object,
    group_by: str | None,
) -> tuple[dict[str, list[str]], str]:
    """
    Validate caller inputs and return the resolved `keys` and `group_by`.

    Raises if `keys` is also supplied, or if a caller-provided `group_by`
    disagrees with the value stored in `adata.uns[...]['params']['groupby']`.
    """
    if keys is not None:
        msg = (
            "`keys` must be omitted when `rank_genes_groups` is set; "
            "genes are derived from the stored ranking."
        )
        raise ValueError(msg)

    resolved_keys, stored_group_by = _extract_rank_genes_groups(
        data,
        rank_genes_groups=rank_genes_groups,
        n_genes=n_genes,
        groups=groups,
    )

    if group_by is not None and group_by != stored_group_by:
        msg = (
            f"`group_by={group_by!r}` does not match the value used when "
            f"`rank_genes_groups` was computed ({stored_group_by!r}). "
            "Omit `group_by` to inherit it, or recompute with the desired key."
        )
        raise ValueError(msg)

    return resolved_keys, stored_group_by


# ---------------------------------------------------------------------------
# stacked_violin.py helper
# ---------------------------------------------------------------------------


def _compute_violin_polygons(
    frame: pl.DataFrame,
    *,
    variable_column: str,
    value_column: str,
    group_by: str,
    x_keys: list[str],
    y_order_groups: list[str],
    n_points: int,
    scale: Literal["area", "count", "width"],
    width_scale: float,
    height_scale: float,
    aggregate: Literal["mean", "median"],
    aggregate_key: str,
    kde_max_samples: int | None = None,
) -> pl.DataFrame:
    """
    Build a polygon-vertex frame for one violin per (variable, group) cell.

    Each violin is centered at `(x=variable_index, y=group_index)` with value
    mapped to local y inside the row band and density mapped to local x around
    the column position.
    """
    var_stats = frame.group_by(variable_column).agg(
        pl.col(value_column).min().alias("_vmin"),
        pl.col(value_column).max().alias("_vmax"),
    )
    var_range = {
        row[variable_column]: (row["_vmin"], row["_vmax"])
        for row in var_stats.iter_rows(named=True)
    }

    rows: list[dict] = []
    polygon_id = 0
    half_height = height_scale / 2
    half_width_max = width_scale / 2
    # Deterministic subsample per (variable, group) before KDE.
    rng = np.random.default_rng(0) if kde_max_samples is not None else None

    for x_index, var_key in enumerate(x_keys):
        if var_key not in var_range:
            continue
        var_min, var_max = var_range[var_key]
        if var_min is None or var_max is None or var_max <= var_min:
            continue
        value_span = var_max - var_min

        var_frame = frame.filter(pl.col(variable_column) == var_key)
        kde_results: dict[str, tuple[int, np.ndarray, np.ndarray, int, float]] = {}

        for y_index, group_key in enumerate(y_order_groups):
            group_values = (
                var_frame.filter(pl.col(group_by).cast(pl.String) == group_key)[value_column]
                .drop_nulls()
                .to_numpy()
            )
            if len(group_values) < 2 or group_values.std() == 0:
                continue
            if rng is not None and len(group_values) > kde_max_samples:
                kde_input = group_values[
                    rng.choice(len(group_values), size=kde_max_samples, replace=False)
                ]
            else:
                kde_input = group_values
            try:
                kde = gaussian_kde(kde_input)
            except Exception:
                continue
            grid = np.linspace(var_min, var_max, n_points)
            density = kde(grid)
            agg_value = (
                float(np.median(group_values))
                if aggregate == "median"
                else float(group_values.mean())
            )
            kde_results[group_key] = (
                y_index,
                grid,
                density,
                len(group_values),
                agg_value,
            )

        if not kde_results:
            continue

        if scale == "width":
            normalizers = {gk: float(d.max()) for gk, (_, _, d, _, _) in kde_results.items()}
        elif scale == "count":
            global_max = max(float((d * n).max()) for _, _, d, n, _ in kde_results.values())
            normalizers = dict.fromkeys(kde_results, global_max)
        elif scale == "area":
            global_max = max(float(d.max()) for _, _, d, _, _ in kde_results.values())
            normalizers = dict.fromkeys(kde_results, global_max)
        else:
            msg = f"scale must be one of 'area', 'count', 'width' (got {scale!r})"
            raise ValueError(msg)

        for group_key, (y_index, grid, density, n_obs, agg_value) in kde_results.items():
            if scale == "count":
                density = density * n_obs
            normalizer = normalizers[group_key]
            if normalizer <= 0:
                continue

            half_width = (density / normalizer) * half_width_max
            y_local = (y_index - half_height) + (grid - var_min) / value_span * height_scale
            x_left = x_index - half_width
            x_right = x_index + half_width
            poly_x = np.concatenate([x_right, x_left[::-1]])
            poly_y = np.concatenate([y_local, y_local[::-1]])

            for px, py in zip(poly_x.tolist(), poly_y.tolist(), strict=True):
                rows.append(
                    {
                        "polygon_id": polygon_id,
                        "x": px,
                        "y": py,
                        variable_column: var_key,
                        group_by: group_key,
                        aggregate_key: agg_value,
                    }
                )
            polygon_id += 1

    if not rows:
        schema = {
            "polygon_id": pl.Int64,
            "x": pl.Float64,
            "y": pl.Float64,
            variable_column: pl.String,
            group_by: pl.String,
            aggregate_key: pl.Float64,
        }
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows)
