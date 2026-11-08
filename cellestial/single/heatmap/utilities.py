from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "numpy", "scipy.stats"]

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import polars as pl
from anndata import AnnData
from ggrust import element_line, guide_axis_nested, key_range_manual, theme
from scipy.stats import gaussian_kde

from cellestial.util import _warn
from cellestial.util.dendrogram import _GEOM_LINE_UNIT
from cellestial.util.errors import DuplicateKeysError, KeyNotFoundError, _unsupported_data_type

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ggrust.plot.core import FeatureSpec


# ---------------------------------------------------------------------------
# heatmap.py helpers
# ---------------------------------------------------------------------------

_GROUP_BAR_RATIO = 0.02
_GROUP_BAR_GAP_RATIO = 0.025
_GROUP_BAR_GAP_MIN = 0.15


def _bin_within_groups(
    frame: pl.DataFrame,
    *,
    observation_id_column: str,
    group_by: str,
    keys: list[str],
    y_order_groups: list[str],
    max_rows: int,
) -> pl.DataFrame:
    """
    Collapse observations into at most `max_rows` virtual rows, within each group.

    Notes
    -----
    `frame` is wide, one row per observation and one column per key.
    Cells within a group are sorted by observation id and chunked into contiguous
    bins; each bin's expression value is the per-variable mean of its members.
    Group boundaries are preserved, so group bars/lines remain meaningful.
    Returns `frame` unchanged when the unique observation count fits in `max_rows`.
    """
    group_key = "_group_key"
    frame_with_key = frame.with_columns(pl.col(group_by).cast(pl.String).alias(group_key))
    obs_per_group = (
        frame_with_key.select(observation_id_column, group_key)
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
        frame_with_key.select(observation_id_column, group_key)
        .unique(maintain_order=True)
        .join(offsets_frame, on=group_key)
        .sort(["_group_index", observation_id_column])
        .with_columns(pl.int_range(pl.len()).over(group_key).alias("_rank_in_group"))
        .with_columns(
            (
                (pl.col("_rank_in_group") * pl.col("_bin_count") // pl.col("_n_cells"))
                + pl.col("_bin_offset")
            )
            .cast(pl.String)
            .alias("_binned_obs")
        )
        .select(observation_id_column, "_binned_obs")
    )

    return (
        frame.join(obs_frame, on=observation_id_column)
        .drop(observation_id_column)
        .rename({"_binned_obs": observation_id_column})
        .group_by(observation_id_column, group_by, maintain_order=False)
        .agg(pl.col(keys).mean())
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


def _scale_wide_values(frame: pl.DataFrame, *, keys: list[str], scale_axis: int) -> pl.DataFrame:
    """Min-max scale `keys` per column (`scale_axis=0`) or per row (`scale_axis=1`)."""
    row_bounds = ["__cellestial_row_min", "__cellestial_row_max"]
    if scale_axis == 1:
        frame = frame.with_columns(
            pl.min_horizontal(keys).alias(row_bounds[0]),
            pl.max_horizontal(keys).alias(row_bounds[1]),
        )
    scaled = []
    for key in keys:
        value = pl.col(key)
        value_min = value.min() if scale_axis == 0 else pl.col(row_bounds[0])
        value_max = value.max() if scale_axis == 0 else pl.col(row_bounds[1])
        value_range = value_max - value_min
        # Nulls stay null so they are dropped after unpivoting.
        scaled.append(
            pl.when(value.is_not_null() & (value_range == 0))
            .then(0.0)
            .otherwise((value - value_min) / value_range)
            .alias(key)
        )
    return frame.with_columns(scaled).drop(row_bounds, strict=False)


def _group_bar_gap(n_x: int) -> float:
    """Return a stable visual gap between the heatmap and group bars."""
    return max(_GROUP_BAR_GAP_MIN, n_x * _GROUP_BAR_GAP_RATIO)


def _nonaggregate_y_step(n_x: int, n_y: int) -> float:
    """Return the distance between adjacent nonaggregated row centers."""
    y_span = max(n_x - 1, 1)
    return y_span / max(n_y - 1, 1)


def _assign_positions(
    frame: pl.DataFrame,
    *,
    aggregate: bool,
    group_by: str,
    observation_id_column: str,
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
    x_positions = {k: i for i, k in enumerate(x_keys)}
    frame = frame.with_columns(
        pl.col(variable_column)
        .replace_strict(x_positions, return_dtype=pl.Float64)
        .alias("position_x")
    )

    if aggregate:
        n_y = len(y_order_groups)
        y_positions = {g: i for i, g in enumerate(y_order_groups)}
        frame = frame.with_columns(
            pl.col(group_by)
            .cast(pl.String)
            .replace_strict(y_positions, return_dtype=pl.Float64)
            .alias("position_y"),
        )
        return frame, None, n_x, n_y, [float(i) for i in range(n_y)]

    # Nonaggregate: use a width-comparable y span for a square-ish aspect.
    group_index = {g: i for i, g in enumerate(y_order_groups)}
    cell_frame = (
        frame.select(observation_id_column, group_by)
        .unique()
        .with_columns(
            pl.col(group_by)
            .cast(pl.String)
            .replace_strict(group_index, return_dtype=pl.Int64)
            .alias("_group_index")
        )
        .sort(["_group_index", observation_id_column])
    )
    n_y = cell_frame.height
    y_step = _nonaggregate_y_step(n_x, n_y)
    cell_frame = cell_frame.with_columns(
        (pl.int_range(pl.len()).cast(pl.Float64) * y_step).alias("position_y")
    )
    frame = frame.join(
        cell_frame.select(observation_id_column, "position_y"), on=observation_id_column
    )
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
        half_step = _nonaggregate_y_step(n_x, n_y) / 2
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


def _key_groups_guide(groups: dict[str, list[str]], *, width: float) -> FeatureSpec:
    """
    Axis guide bracketing each key group, across the panel from the key labels.

    Notes
    -----
    Keys sit at continuous positions `0..n-1` in group order. A group spans its
    first to last key, extended by `width / 2` on each side.
    """
    starts: list[float] = []
    ends: list[float] = []
    position = 0
    for values in groups.values():
        starts.append(position - width / 2)
        ends.append(position + len(values) - 1 + width / 2)
        position += len(values)
    return guide_axis_nested(
        key_range_manual(starts, ends, list(groups)), bracket="square", side="opposite", angle=90
    )


def _key_groups_theme(*, color: str, size: float) -> FeatureSpec:
    """Style the key-group brackets; `size` stays in the geom line units of the former path layer."""
    return theme(legendry_bracket=element_line(color=color, size=size * _GEOM_LINE_UNIT))


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

    raise _unsupported_data_type(data, AnnData)


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

    Notes
    -----
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
    aggregate_column: str,
    kde_max_samples: int | None = None,
) -> pl.DataFrame:
    """
    Build a polygon-vertex frame for one violin per (variable, group) cell.

    Notes
    -----
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
            aggregated_value = (
                float(np.median(group_values))
                if aggregate == "median"
                else float(group_values.mean())
            )
            kde_results[group_key] = (
                y_index,
                grid,
                density,
                len(group_values),
                aggregated_value,
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

        for group_key, (
            y_index,
            grid,
            density,
            n_observations,
            aggregated_value,
        ) in kde_results.items():
            if scale == "count":
                density = density * n_observations
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
                        aggregate_column: aggregated_value,
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
            aggregate_column: pl.Float64,
        }
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows)
