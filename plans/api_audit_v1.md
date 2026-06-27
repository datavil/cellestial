# Cellestial v1.0 API Audit

Inconsistencies and "makes-no-sense" items found across the public API ahead of v1.0.

## A. Clear defects / "makes-no-sense" (fix before 1.0)

1. **`borders()` has debug-colour defaults** — defaults are `panel_color="green", legend_color="blue", plot_color="red"`. ACCEPTED AS-IS (intentional, kept for debugging panel/legend/plot regions).

2. ~~**`stream(cutoff_percentile)` is mis-typed as `None`**~~ — FIXED: now `float | None`.

3. ~~**`volcanos` is missing its return annotation**~~ — FIXED: now `-> SupPlotsSpec`.

4. ~~**`highest_expressed_genes(size=0.5)` is untyped**~~ — FIXED: now `size: float = 0.5`.

## B. Naming inconsistencies (same concept, two names)

5. ~~**`add_keys` vs `add_columns`**~~ — RESOLVED: standardized on `add_keys` across the board (renamed `add_columns` → `add_keys` in dimensional/subdimensional(s)/xyplot(s)/spatial(s)). Chosen because it matches the library's "key" vocabulary and accepts a gene/variable name as well as an obs column (so "column" was too narrow, and collided with the `value_column`/`variable_column` output-name params).

6. ~~**`line_type` vs `linetype`**~~ — FIXED: renamed `elbow(line_type=...)` → `linetype` to match `cluster_outlines(linetype=...)`, `volcano(threshold_linetype=...)`, and lets-plot.

7. **Count params: `n` vs `n_genes` vs `top_n`** — RESOLVED (kept as-is, names are deliberate): `volcano(top_n=10)` is a rank-by-significance cutoff; `n_genes` is the marker-selection count shared by `markers`/`marker_genes`/`marker_genes_dict`/`dotplot`/`heatmap`/`matrixplot`/`stacked_violin`; `highest_expressed_genes(n=20)` is a plain count. `n_genes` is clearer than bare `n` in the `markers=True` matrix plots, so it stays.

## C. Behavioral / default inconsistencies across sibling functions

8. ~~**`axis_type` default differs for `expression` alone**~~ — FIXED: `expression` now defaults `axis_type=None` like `expressions`/`dimensional`/`umap`/`tsne`/`pca`. Note this is a visual change: `None` blanks the axis text/ticks/lines, whereas the old `"axis"` showed a full axis. Pass `axis_type="axis"` to restore.

9. **The `share_*` grid family is inconsistent across plurals.** Three private helpers each trim a different layer of redundant decoration from the inner panels of a grid (keeping it only on the bottom row / left column):
   - `_share_labels` → hides axis **titles** (e.g. "UMAP1").
   - `_share_ticks` → hides axis **tick text** (the numbers/categories).
   - `_share_axis` → hides the **whole axis** (text + ticks + line).

   Which of these a plural exposes is inconsistent, so the same "tidy the grid" intent needs a different parameter depending on which family you are in:
   - `dimensionals`/`umaps`/`tsnes`/`pcas`/`expressions`/`markers`/`volcanos`: `share_labels` + `share_axis` (no `share_ticks`)
   - `histograms`/`violins`/`boxplots`: `share_ticks` + `share_axis` (no `share_labels`)
   - `spatials`: `share_labels` only (no `share_axis`)
   - `ridges`: none of them

   The differing param surface across families is INTENTIONAL — left as-is. Secondary defects found while tracing this:
   - ~~**Docstring/signature default mismatch:**~~ FIXED: `histograms`/`violins` `share_ticks` docstrings now say `default=False` to match the signature.
   - ~~**Copy-pasted docstring:**~~ FIXED: the distribution `share_ticks` doc text now reads "Whether to share the ticks..." (was "...the labels...").
   - **Default disagreement:** `markers` defaults `share_labels=True` while the scatter siblings and `volcanos` default `False`. INTENTIONAL — left as-is.

10. **`groups` accepted types differ** — dimensional/umap/violin/etc.: `Sequence[str] | str | None` (bare str ok); dotplot/heatmap/matrixplot/stacked_violin: `Sequence[str] | None` (bare str rejected). Same param name, different accepted input. INTENTIONAL — left as-is.

11. **`mid_point` default split** — scatter-family (`expression`/`umap`/...) default `'median'`; matrix-family (`dotplot`/`heatmap`/`matrixplot`/`stacked_violin`/`spatial`) default `'mid'`. INTENTIONAL — left as-is.

12. **`group_by` is positional-required in `ridge`/`ridges`** but keyword-optional everywhere else (`violin`, `boxplot`, `histogram`). INTENTIONAL (ridge needs grouping) — left as-is.

## D. Type-annotation inconsistencies (cosmetic but visible in 1.0 docs)

13. ~~**`widths`/`heights`: `list` vs `list[float]`**~~ — FIXED: distribution plurals (`histograms`/`violins`/`boxplots`/`ridges`) now annotate `list[float] | None`, matching every other plural.

14. ~~**`build_frame` defaults lowercase, plots capitalize**~~ — FIXED (Option A): `build_frame`, `anndata_observations_frame`, and `anndata_variables_frame` now default `observations_name='Barcode'`/`variables_name='Variable'`, matching the plots. A hand-built frame fed via `frame=` now lines up by default. Pass lowercase explicitly to restore the old column names.

## Worth a second look (semantic, possibly intentional)

15. ~~**`volcano(group=...)` / `volcanos(groups=...)` overload `group(s)`**~~ — RESOLVED (names kept, docstrings clarified): the `group`/`group_by` pairing matches the standard DE convention, so the names stay. The `group(s)` docstrings now state it is the level(s) of `group_by` whose differential-expression ranking is plotted (e.g. "B Cells" vs the rest).

16. **`heatmap`/`matrixplot` carry both `axis` and `scale_axis`** — two axis-valued params on one function; easy to confuse. `spatial` exposes only `scale_axis`. INTENTIONAL — left as-is (the two control unrelated things: data orientation vs standardization direction).

17. **Singular `histogram`/`violin`/`boxplot` accept `key: str | Sequence[str]`** even though plural variants exist — "give me multiple keys" has two entry points with different layout semantics (overlay vs facet). INTENTIONAL — left as-is.

---

Items **1-4** are straight defects to fix regardless. **5-10** are the inconsistencies most likely to confuse users and lock in bad names post-1.0.
