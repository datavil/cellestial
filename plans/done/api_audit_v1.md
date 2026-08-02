# Cellestial v1.0 API Audit

Inconsistencies and "makes-no-sense" items found across the public API ahead of v1.0.

## Open items

None.

## Closed items

### Kept as-is

1. **`borders()` has debug-colour defaults** — accepted as intentional. The `panel_color="green"`, `legend_color="blue"`, and `plot_color="red"` defaults help distinguish the panel, legend, and plot regions while debugging.

7. **Count params: `n` vs `n_genes` vs `top_n`** — the names are deliberate. `volcano(top_n=10)` is a rank-by-significance cutoff; `n_genes` is the marker-selection count shared by `markers`/`marker_genes`/`marker_genes_dict`/`dotplot`/`heatmap`/`matrixplot`/`stacked_violin`; and `highest_expressed_genes(n=20)` is a plain count. `n_genes` is clearer than bare `n` in the `markers=True` matrix plots.

9. **The `share_*` grid family differs across plurals** — accepted as intentional. The private helpers remove different layers of redundant decoration from inner grid panels:

   - `_share_labels` hides axis titles, such as "UMAP1".
   - `_share_ticks` hides axis tick text, such as numbers or categories.
   - `_share_axis` hides the whole axis: text, ticks, and line.

   The public parameters remain family-specific:

   - `dimensionals`/`umaps`/`tsnes`/`pcas`/`expressions`/`markers`/`volcanos`: `share_labels` and `share_axis`
   - `histograms`/`violins`/`boxplots`: `share_ticks` and `share_axis`
   - `spatials`: `share_labels` only
   - `ridges`: none

   `markers` also intentionally defaults `share_labels=True`, while the scatter siblings and `volcanos` default it to `False`.

10. **`groups` accepted types differ** — accepted as intentional. Dimensional/umap/violin/etc. accept `Sequence[str] | str | None`; dotplot/heatmap/matrixplot/stacked_violin accept `Sequence[str] | None` and reject a bare string.

11. **`mid_point` defaults differ by family** — accepted as intentional. The scatter family (`expression`/`umap`/etc.) defaults to `"median"`; the matrix family (`dotplot`/`heatmap`/`matrixplot`/`stacked_violin`/`spatial`) defaults to `"mid"`.

12. **`group_by` is positional-required in `ridge`/`ridges`** — accepted as intentional because ridge plots require grouping. It remains keyword-optional in `violin`, `boxplot`, and `histogram`.

15. **`volcano(group=...)` / `volcanos(groups=...)` overload `group(s)`** — names kept. The `group`/`group_by` pairing matches the standard differential-expression convention. The docstrings clarify that `group(s)` identifies the level(s) of `group_by` whose ranking is plotted, for example "B Cells" versus the rest.

16. **`heatmap`/`matrixplot` carry both `axis` and `scale_axis`** — accepted as intentional because the parameters control unrelated things: data orientation and standardization direction. `spatial` exposes only `scale_axis`.

17. **Singular `histogram`/`violin`/`boxplot` accept `key: str | Sequence[str]`** — accepted as intentional. Multiple keys through a singular function use overlay semantics; the plural variants use facet layout semantics.

### Done

2. **`stream(cutoff_percentile)` was mis-typed as `None`** — fixed to `float | None`.

3. **`volcanos` was missing its return annotation** — fixed to `-> SupPlotsSpec`.

4. **`highest_expressed_genes(size=0.5)` was untyped** — fixed to `size: float = 0.5`.

5. **`add_keys` vs `add_columns`** — standardized on `add_keys` across dimensional/subdimensional(s)/xyplot(s)/spatial(s). This matches the library's "key" vocabulary, accepts a gene/variable name as well as an observations column, and avoids collision with the `value_column`/`variable_column` output-name parameters.

6. **`line_type` vs `linetype`** — renamed `elbow(line_type=...)` to `linetype`, matching `cluster_outlines(linetype=...)`, `volcano(threshold_linetype=...)`, and lets-plot.

8. **`axis_type` default differed for `expression` alone** — changed the default to `None`, matching `expressions`/`dimensional`/`umap`/`tsne`/`pca`. This is a visual change: `None` blanks axis text, ticks, and lines; pass `axis_type="axis"` to restore the previous full axis.

9a. **Distribution `share_ticks` docstrings disagreed with the signatures** — updated `histograms`/`violins` to document `default=False`.

9b. **Distribution `share_ticks` docstrings referred to labels** — changed the copied text to "Whether to share the ticks...".

13. **`widths`/`heights` used `list` instead of `list[float]`** — distribution plurals (`histograms`/`violins`/`boxplots`/`ridges`) now use `list[float] | None`, matching the other plural APIs.

14. **`build_frame` defaults were lowercase while plot defaults were capitalized** — changed `build_frame`, `anndata_observations_frame`, and `anndata_variables_frame` to default to `observations_name="Barcode"` and `variables_name="Variable"`. Pass lowercase names explicitly to restore the previous output.
