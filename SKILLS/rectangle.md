# Adding a Rectangle Border to a Plot

How a border rectangle is wired around the data area in `dotplot` and `stacked_violin`. Use this as the template when adding a rectangle border to a new plot.

## Why

When the plot uses continuous numeric axes to fake categorical labels (see [dendrogram.md](dendrogram.md)), the default panel frame doesn't sit tightly against the data. A `geom_rect` lets you draw a tight border around the data area only, while keeping side elements (dendrogram, group bars) **outside** the frame.

## Required parameters to expose

```python
rectangle: bool = True,
rectangle_size: float = 0.8,
rectangle_color: str = "#3f3f3f",
rectangle_kwargs: dict | None = None,
```

## Wiring

After data layers and after computing `n_x` and `data_top` (the top y-coordinate of the data area, which differs per plot type, see below), add:

```python
if rectangle:
    plot += geom_rect(
        data={
            "xmin": [-0.5],
            "xmax": [n_x - 0.5],
            "ymin": [-0.5],
            "ymax": [data_top],
        },
        mapping=aes(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax"),
        color=rectangle_color,
        size=rectangle_size,
        fill="rgba(0,0,0,0)",
        inherit_aes=False,
        **(rectangle_kwargs or {}),
    )
else:
    plot += theme(axis_line=element_line(color="#1f1f1f"))
```

Key points:

- `fill="rgba(0,0,0,0)"` keeps the rectangle a stroke-only border.
- `inherit_aes=False` prevents the main aesthetic mapping (e.g. `fill=value_column`) from polluting the rect.
- The `else` branch restores a normal axis line so the plot still has a visible frame when the rectangle is disabled.
- The rectangle ends at `n_x - 0.5`, so any dendrogram drawn beyond that x stays outside the border (this is intentional, see [dendrogram.md](dendrogram.md)).

## Computing `data_top`

`data_top` is the y-coordinate of the top edge of the data area, **excluding** any padding for key-group brackets above:

- **Aggregated row-per-group plots** (matrixplot, dotplot, stacked_violin, heatmap with `aggregate=True`): `data_top = n_y - 0.5`.
- **Per-cell heatmap** (`aggregate=False`): cells extend half a row above `position_y`'s max, so `data_top = (n_x - 1) + half_step` where `half_step = (n_x - 1) / max(n_y - 1, 1) / 2`.

Do not use `y_max_limit` here. The y-axis limit may be extended above `data_top` to make room for key-group bracket labels, but the rectangle should hug the data, not the bracket area.

## Axis interaction

The plot must already use continuous numeric scales with explicit `limits` and `expand=[0, 0]` (same requirement as dendrograms). Without `expand=[0, 0]`, lets-plot adds default padding and the rectangle no longer aligns with the panel edge.

## Reference implementations

- [cellestial/single/heatmap/dotplot.py](../cellestial/single/heatmap/dotplot.py)
- [cellestial/single/heatmap/stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py)
