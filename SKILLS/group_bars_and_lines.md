# Group Bars and Group Separator Lines

How the colored vertical bar (left) and horizontal separator lines (within the data area) are wired into the heatmap. Use this as the template when adding either to a new plot.

## What they are

- **Group bars**: a thin colored vertical bar drawn just to the **left** of the data area, one segment per group, colored by `group_by`. Acts as a category legend that doesn't compete with the y-axis tick labels.
- **Group lines**: thin horizontal lines drawn **inside** the data area at the boundaries between groups. Visual separator only.

Both currently live inline in [heatmap.py](../cellestial/single/heatmap/heatmap.py): `_get_group_bar_frame` and `_get_group_lines_frame`. They are not exported. If you need them in another plot, lift them out of `heatmap.py` to a shared module.

## Required parameters to expose

```python
group_bars: bool = True,
group_bars_size: float = 6,
group_bars_labels: bool = True,        # show group names on the y-axis (replaces color legend)
group_lines: bool = True,
group_lines_color: str = "black",
group_lines_size: float = 1.0,
group_lines_kwargs: dict | None = None,
```

Group bars only make sense when you have **per-cell rows** (i.e. heatmap with `aggregate=False`); when each row already corresponds to a group, use `group_bars_labels` directly on the y-scale.

## Wiring: group bars

After computing `cell_frame` and `n_x`, build the bar frame and add a `geom_segment`:

```python
if not aggregate and group_bars:
    bar_frame = _get_group_bar_frame(cell_frame, group_by=group_by, n_x=n_x)
    plot += geom_segment(
        data=bar_frame,
        mapping=aes(x="x", xend="x", y="y_min", yend="y_max", color=group_by),
        size=group_bars_size,
    )
```

Geometry constants in [heatmap.py](../cellestial/single/heatmap/heatmap.py):

```python
_GROUP_BAR_RATIO = 0.02   # bar width = max(1.0, n_x * _GROUP_BAR_RATIO)
_GROUP_BAR_GAP   = 0.5    # gap between bar and data area; bar's right edge sits at -_GROUP_BAR_GAP
```

The bar lives at negative x. **Don't** extend the x-axis limits to include it; let the bar bleed off the panel on the left, since x-axis ticks are at `0..n_x-1`. The visible result is a bar flush with the left edge of the panel.

## Wiring: group bar labels (replacing the legend)

When `group_bars=True` and `group_bars_labels=True`, the bar replaces the color legend with y-axis tick labels at each group's center:

```python
plot += scale_y_continuous(
    breaks=group_centers,
    labels=y_order_groups,
    limits=[-0.5, y_max_limit],
)
plot += guides(color="none")    # hide the color legend produced by the bar
```

`group_centers` is the y-coordinate of each group's center, in `y_order_groups` order. See [dendrogram.md](dendrogram.md) for how to compute it.

When `group_bars_labels=False`, hide y-axis text instead:

```python
plot += theme(axis_text_y=element_blank(), axis_ticks_y=element_blank())
```

## Wiring: group lines

After determining group ordering and layout, build the line frame and add a `geom_segment`. `_get_group_lines_frame` handles both aggregated and per-cell cases:

```python
if group_lines and len(y_order_groups) > 1:
    lines_frame = _get_group_lines_frame(
        cell_frame,                # None when aggregate=True
        aggregate=aggregate,
        group_by=group_by,
        n_x=n_x,
        n_y=n_y,
        n_groups=len(y_order_groups),
    )
    plot += geom_segment(
        data=lines_frame,
        mapping=aes(x="x", xend="xend", y="y", yend="yend"),
        color=group_lines_color,
        size=group_lines_size,
        **(group_lines_kwargs or {}),
    )
```

Line y-positions:

- **Aggregated**: `i + 0.5` for `i in 0..n_groups-2` (halfway between adjacent rows).
- **Per-cell**: max `position_y` of each group plus `half_step = (n_x - 1) / max(n_y - 1, 1) / 2`.

Lines span `x in [-0.5, n_x - 0.5]`, matching the data area exactly.

## Order of operations

Add these layers **after** the main data layer (`geom_raster`/`geom_tile`/etc.) so they draw on top, and **before** the dendrogram and rectangle layers so those still sit on the outermost edges visually.

## Reference implementation

- [cellestial/single/heatmap/heatmap.py](../cellestial/single/heatmap/heatmap.py) - both helpers and full wiring.
