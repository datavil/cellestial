# Adding a Dendrogram to a Plot

How dendrograms are wired into the heatmap variants (`heatmap`, `matrixplot`, `dotplot`, `stacked_violin`). Use this as the template when adding a dendrogram to a new plot.

## Helpers

Two private helpers live in [cellestial/util/dendrogram.py](../cellestial/util/dendrogram.py) and are re-exported from [cellestial.util](../cellestial/util/__init__.py):

- `_get_dendrogram(data, group_by) -> (categories_ordered, paths)`
  - Reads `data.uns[f"dendrogram_{group_by}"]`, computes via `scanpy.tl.dendrogram` if missing.
  - `categories_ordered`: list of group labels in dendrogram leaf order. **Use this as the y order.**
  - `paths`: Polars DataFrame with normalized `x` (leaf positions 0..n_groups-1) and `y` (dcoord scaled to 0..1), plus `group` id per linkage segment.

- `_get_dendrogram_path_frame(paths, *, n_x, n_groups, group_centers, dendrogram_ratio=0.15) -> pl.DataFrame`
  - Maps the normalized paths into actual plot coordinates on the **right side**, along the y-axis.
  - Width of dendrogram = `n_x * dendrogram_ratio`, anchored at `x = n_x - 0.5`.
  - `group_centers` is the y-pixel center of each group, ordered to match `categories_ordered`.

## Required parameters to expose

```python
dendrogram: bool = False,
dendrogram_color: str = "black",
dendrogram_size: float = 0.5,
dendrogram_kwargs: dict | None = None,
```

## Wiring (three steps)

**1. Determine y order before building positions.** The dendrogram dictates group order; everything else (positions, separator lines, group bars) must follow.

```python
if dendrogram:
    y_order_groups, paths = _get_dendrogram(data, group_by)
else:
    y_order_groups = frame.select(group_by).unique(maintain_order=True)[group_by].cast(pl.String).to_list()
    paths = None
```

**2. After computing layout, build the path frame and add `geom_path`.** You need `n_x`, `n_groups`, and `group_centers` (the y-coordinate of each group's center, in `y_order_groups` order).

```python
if dendrogram:
    dendro_frame = _get_dendrogram_path_frame(
        paths, n_x=n_x, n_groups=len(y_order_groups), group_centers=group_centers,
    )
    plot += geom_path(
        data=dendro_frame,
        mapping=aes(x="x", y="y", group="group"),
        color=dendrogram_color,
        size=dendrogram_size,
        **(dendrogram_kwargs or {}),
    )
```

**3. Extend the x-axis limit to include the dendrogram width.** The dendrogram lives at `x in [n_x - 0.5, n_x - 0.5 + n_x*dendrogram_ratio]`. If you set explicit `scale_x_continuous(limits=...)` or have a bordering `geom_rect`, account for this:

```python
x_max_limit = n_x - 0.5
if dendrogram:
    x_max_limit = dendro_frame["x"].max()
plot += scale_x_continuous(breaks=list(range(n_x)), labels=x_keys, limits=[-0.5, x_max_limit], expand=[0, 0])
```

If a border rectangle frames the data area, draw it from `-0.5` to `n_x - 0.5` so the dendrogram sits **outside** the frame (see dotplot pattern).

## Axis type

The plot must use **continuous numeric** axes (not discrete), because the dendrogram lives at fractional x-coordinates beyond the data area. Pattern:

- Encode positions as floats (`position_x`, `position_y`).
- Use `scale_x_continuous(breaks=list(range(n_x)), labels=x_keys)` and analogous y scale to fake categorical labels on a continuous axis.
- Always pass `limits=[...]` and `expand=[0, 0]` when a dendrogram is present, so it isn't clipped.

## Computing `group_centers`

- **Aggregated row-per-group plots** (matrixplot, dotplot, stacked_violin, heatmap with `aggregate=True`): `group_centers = [float(i) for i in range(n_groups)]`.
- **Per-cell rows** (heatmap with `aggregate=False`): compute the mean `position_y` of cells in each group, sorted by group order. See `_assign_positions` in [heatmap.py](../cellestial/single/heatmap/heatmap.py) for the canonical implementation.

## Reference implementations

- [cellestial/single/heatmap/heatmap.py](../cellestial/single/heatmap/heatmap.py) (per-cell and aggregated)
- [cellestial/single/heatmap/dotplot.py](../cellestial/single/heatmap/dotplot.py) (with bordering rectangle)
- [cellestial/single/heatmap/stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py)
