# Adding a Dendrogram to a Plot

How dendrograms are wired into the heatmap variants (`heatmap`, `matrixplot`, `dotplot`, `stacked_violin`, `annotated_heatmap`). Use this as the template when adding a dendrogram to a new plot.

## Two cases

- **One row per group** (dotplot, stacked_violin, matrixplot, heatmap with `aggregate=True`): use ggrust's native axis dendrogram. The tree orders the y-axis and draws itself right of the panel.
- **Per-cell rows** (heatmap with `aggregate=False`; `complex/annotated_heatmap.py`): groups span many rows. Still the axis dendrogram, with `leaf_levels={group: [row levels]}` so each leaf spans its group's rows. Rows must sit one unit apart (level `k` at `k`): the heatmap passes `unit_rows=True` to `_assign_positions` / `_nonaggregate_y_step` / `_get_group_lines_frame`. The annotated heatmap builds `scale_x_dendro`/`scale_y_dendro` itself on the observation axis (`position=observation_side`, `reverse=not transpose` so the first group sits at the top), drawn outside the annotation tracks.

## Helpers

All live in [cellestial/util/dendrogram.py](../cellestial/util/dendrogram.py), re-exported from [cellestial.util](../cellestial/util/__init__.py). Each reads `data.uns[use_key or f"dendrogram_{group_by}"]` and computes it with `scanpy.tl.dendrogram` if missing.

- `_get_dendrogram_tree(data, group_by, *, use_key=None) -> ClusterTree`: scanpy's linkage as a ggrust tree; `tree.ordered_labels` is scanpy's `categories_ordered`.
- `_dendrogram_y_axis(tree, *, limits, fixed=False, leaf_levels=None, breaks=None, labels=None)`: `scale_y_dendro(..., guide=guide_axis_dendro(side="opposite"))` plus a coord holding the y `limits`. Pass `fixed=True` for `geom_raster` plots, which default to `coord_fixed()`; `leaf_levels`, `breaks` and `labels` for per-cell rows.
- `_dendrogram_line_theme(*, color, size, axis="y")`: the tree is drawn with the axis tick line, so this styles `axis_ticks_{axis}` (size in geom units) and sets the tick length to 0. Add it **after** any full theme (`theme_classic()` in `_THEME_DOTPLOT` resets it).

## Wiring a one-row-per-group plot

Keep numeric `position_y` values (group `i` of the tree order at `i`): numeric layers (dots, violins, group lines) still work on the discrete dendrogram scale.

```python
if dendrogram:
    tree = _get_dendrogram_tree(data, group_by, use_key=dendrogram_key)
    y_order_groups = list(tree.ordered_labels)
...
if dendrogram:
    plot += _dendrogram_y_axis(tree, limits=[-0.5, data_top])
else:
    plot += scale_y_continuous(breaks=list(range(n_y)), labels=y_order_groups,
                               limits=[-0.5, data_top], expand=[0, 0])
...
plot += _THEME_DOTPLOT
if dendrogram:
    plot += _dendrogram_line_theme(color=dendrogram_color, size=dendrogram_size)
```

No x limit needs to make room for the tree; it lives in its own band outside the panel.

## Parameters to expose

```python
dendrogram: bool = False,
dendrogram_color: str = "black",
dendrogram_size: float = 0.5,
dendrogram_key: str | None = None,
```

## Reference implementations

- [cellestial/single/heatmap/dotplot.py](../cellestial/single/heatmap/dotplot.py) (with bordering rectangle and key-group brackets)
- [cellestial/single/heatmap/stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py)
- [cellestial/single/heatmap/heatmap.py](../cellestial/single/heatmap/heatmap.py) (per-cell rows through `leaf_levels`)
