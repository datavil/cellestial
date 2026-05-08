# Polars Patterns for Plotting

Three recurring Polars idioms used across the library to prepare data for lets-plot. Worth knowing when adding a new plot.

## 1. Categorical → numeric position via `replace_strict`

To plot categorical labels on a continuous axis (see [continuous_axes_as_categorical.md](continuous_axes_as_categorical.md)), each label needs a float position. Use `replace_strict` with a `{label: index}` dict:

```python
x_position = {key: i for i, key in enumerate(x_keys)}
frame = frame.with_columns(
    pl.col(variable_column)
    .replace_strict(x_position, return_dtype=pl.Float64)
    .alias("position_x")
)
```

Why `replace_strict` over `replace`:

- Raises if a value in the column isn't in the mapping (catches typos and missing groups early).
- `return_dtype=pl.Float64` avoids a later cast.

For y when `group_by` is `Categorical` or `Enum`, cast first:

```python
pl.col(group_by).cast(pl.String).replace_strict(y_position, return_dtype=pl.Float64)
```

Reference: `_assign_positions` in [heatmap.py](../cellestial/single/heatmap/heatmap.py).

## 2. Per-group min-max scaling with `.over()`

Use a window expression to scale within partitions without `group_by` + `join`:

```python
v = pl.col(value_column)
vmin = v.min().over(partition_key)
vmax = v.max().over(partition_key)
frame = frame.with_columns(((v - vmin) / (vmax - vmin)).alias(value_column))
```

`.over(key)` is the Polars equivalent of SQL `OVER (PARTITION BY key)`: the aggregate is broadcast back across each row of its partition, leaving row count unchanged. Cleaner and faster than a `group_by(...).agg(...)` followed by `join`.

Generalizes to any per-group transformation: z-score, rank, fraction-of-max, etc.

Reference: `_scale_values` in [heatmap.py](../cellestial/single/heatmap/heatmap.py).

## 3. Unpivot to long form for grammar-of-graphics

Lets-plot expects long-form data: one row per (observation, variable) pair. Standard recipe:

```python
index_columns = [c for c in frame.columns if c not in keys]
frame = frame.unpivot(
    on=keys,
    index=index_columns,
    variable_name=variable_column,
    value_name=value_column,
)
frame = frame.drop_nulls()
```

Notes:

- Compute `index_columns` from the difference, not by hand: it adapts when `build_frame` adds metadata columns and prevents accidental drops.
- Always `drop_nulls` after unpivot. Sparse expression matrices yield nulls that break aggregations downstream.
- For aggregated views, follow with `group_by([group_by, variable_column]).agg(pl.col(value_column).mean())`.

Reference implementations: [heatmap.py](../cellestial/single/heatmap/heatmap.py), [dotplot.py](../cellestial/single/heatmap/dotplot.py), [stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py), [distribution.py](../cellestial/single/core/distribution.py), [ridge.py](../cellestial/single/quick/ridge.py).

## When to wrap in `.lazy()` ... `.collect()`

Optional: wrap a chain of (unpivot + group_by + agg + filter) in `.lazy()` ... `.collect()` if the chain is long enough that the optimizer can fuse projections and pushdowns. Don't sprinkle it on every transform; the overhead isn't free and most chains in this library are short enough that eager mode is fine. Use it when:

- You unpivot, then aggregate, then filter, then sort: 3+ steps the optimizer can rewrite.
- The frame is large (full-cell heatmap, not aggregated).

Example: [dotplot.py](../cellestial/single/heatmap/dotplot.py) wraps unpivot + group_by + agg in lazy mode before downstream sorting.
