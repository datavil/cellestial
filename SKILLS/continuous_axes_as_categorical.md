# Continuous Axes Faking Categorical Labels

The pattern that underpins the heatmap variants: positions are encoded as floats on a continuous axis, but tick labels are placed at integer breaks to read like categorical labels. This unlocks dendrograms, rectangles, group bars, and bracket labels, all of which need to live at fractional coordinates outside the data grid.

## When to use it

Use this any time a "categorical-looking" plot needs side decorations at fractional or out-of-range coordinates:

- Dendrogram beyond the right edge (see [dendrogram.md](dendrogram.md))
- Rectangle border around the data area (see [rectangle.md](rectangle.md))
- Group color bars in negative-x space (see [group_bars_and_lines.md](group_bars_and_lines.md))
- Key-group brackets above the data top

If you don't need any of those, a normal discrete scale is simpler. Use this pattern only when at least one decoration requires it.

## The shape

**1. Encode positions as floats.** Map each variable/group to a `Float64` index:

```python
x_pos = {k: i for i, k in enumerate(x_keys)}
frame = frame.with_columns(
    pl.col(variable_column).replace_strict(x_pos, return_dtype=pl.Float64).alias("position_x")
)
```

Use `position_x` and `position_y` as the aesthetic keys (`aes(x="position_x", y="position_y", ...)`). Cells live at integer coordinates `0, 1, ..., n_x - 1` and `0, 1, ..., n_y - 1`.

**2. Use `scale_*_continuous` with explicit breaks and labels.** This places tick labels at the integer positions while keeping the axis numeric:

```python
plot += scale_x_continuous(
    breaks=list(range(n_x)),
    labels=x_keys,
    limits=[-0.5, x_max_limit],
    expand=[0, 0],
)
plot += scale_y_continuous(
    breaks=list(range(n_y)),         # or `group_centers` for per-cell heatmaps
    labels=y_order_groups,
    limits=[-0.5, y_max_limit],
    expand=[0, 0],
)
```

**3. `expand=[0, 0]` is mandatory.** Without it, lets-plot adds default padding and decorations stop aligning with the panel edge: rectangles drift inward, dendrograms get clipped, group bars float in space.

**4. Compute limits explicitly.** Three quantities to keep separate:

| Quantity | Meaning | Formula |
|---|---|---|
| `data_top` | Top of data area | aggregated: `n_y - 0.5`; per-cell: `(n_x - 1) + half_step` |
| `x_max_limit` | Right axis limit | `n_x - 0.5`, extended to `dendro_frame["x"].max()` if dendrogram present |
| `y_max_limit` | Top axis limit | `data_top`, extended by `_resolve_padding(...)` if key-group brackets present |

`limits[0]` is `-0.5` so that cells at integer position `0` sit half a step from the edge.

## Why -0.5 to n - 0.5

Cell `i` occupies the interval `[i - 0.5, i + 0.5]`. The data area spans `n` cells, hence `[-0.5, n - 0.5]`. All decorations are positioned relative to this convention:

- Group lines span `x in [-0.5, n_x - 0.5]`.
- Rectangle: `xmin=-0.5`, `xmax=n_x - 0.5`.
- Dendrogram starts at `x = n_x - 0.5` and extends to `n_x - 0.5 + n_x * dendrogram_ratio`.
- Group bars sit at `x < -0.5` (negative space, off-axis on the left).

If you change one constant, change all of them.

## Per-cell heatmap caveat

For the non-aggregated heatmap each row is a cell, not a group. The y range is rescaled to `[0, n_x - 1]` (square-ish aspect) instead of `[0, n_y - 1]`:

```python
y_step = (n_x - 1) / max(n_y - 1, 1)
position_y = row_index * y_step
```

Group centers, group line y-positions, and `data_top` all use `half_step = y_step / 2` instead of `0.5`. See `_assign_positions` in [heatmap.py](../cellestial/single/heatmap/heatmap.py).

## Reference implementations

- [cellestial/single/heatmap/heatmap.py](../cellestial/single/heatmap/heatmap.py)
- [cellestial/single/heatmap/dotplot.py](../cellestial/single/heatmap/dotplot.py)
- [cellestial/single/heatmap/stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py)
