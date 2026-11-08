# Adding a Rectangle Border to a Plot

How a border rectangle is wired around the data area in `dotplot` and `stacked_violin`. Use this as the template when adding a rectangle border to a new plot.

## Why

The border boxes the data area only, while side elements (dendrogram, key-group brackets, group bars) stay **outside** the frame. Since ggrust 0.7 the tree and the brackets are axis guides outside the panel, so the panel border is exactly the data area.

## Required parameters to expose

```python
rectangle: bool = True,
rectangle_size: float = 0.8,
rectangle_color: str = "#3f3f3f",
rectangle_kwargs: dict | None = None,
```

## Wiring

The plot theme (`_THEME_DOTPLOT`) blanks `panel_border`, so add the border **after** it:

```python
if not rectangle:
    plot += theme(axis_line=element_line(color="#1f1f1f"))
...
plot += _THEME_DOTPLOT
if rectangle:
    plot += theme(
        panel_border=element_rect(
            color=rectangle_color,
            size=rectangle_size * _GEOM_LINE_UNIT,
            **(rectangle_kwargs or {}),
        )
    )
```

Key points:

- `_GEOM_LINE_UNIT` (`cellestial/util/dendrogram.py`) keeps `rectangle_size` in the geom line units of the former `geom_rect` border.
- `rectangle_kwargs` goes to `element_rect`, so only theme rect options (`linetype`, `fill`) are accepted.
- Plots with ggside track bands set `ggside_panel_border=element_blank()` so a panel border never frames the bands (see `annotated_heatmap`).

## Axis interaction

The panel must end at the data: continuous numeric scales with `limits=[-0.5, n - 0.5]` and `expand=[0, 0]`, as for dendrograms.

## Reference implementations

- [cellestial/single/heatmap/dotplot.py](../cellestial/single/heatmap/dotplot.py)
- [cellestial/single/heatmap/stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py)
