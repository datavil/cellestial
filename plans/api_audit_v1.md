# Cellestial v1.0 API Audit

Inconsistencies and "makes-no-sense" items found across the public API ahead of v1.0.

## A. Clear defects / "makes-no-sense" (fix before 1.0)

1. **`borders()` has debug-colour defaults** — defaults are `panel_color="green", legend_color="blue", plot_color="red"`. Those are primary-colour placeholders, not a sensible public default. Should default to something neutral/off. Docstring also has a typo: "rectangels".

2. **`stream(cutoff_percentile)` is mis-typed as `None`** — `cellestial/layers/stream.py:34` annotates `cutoff_percentile: None = None`, but the docstring says `float | None` and it is actually used at `stream.py:244`. The annotation says "only `None` is valid," contradicting its own behavior. Should be `float | None`.

3. **`volcanos` is missing its return annotation** — every other plural returns `-> SupPlotsSpec`; `volcanos` has none.

4. **`highest_expressed_genes(size=0.5)` is untyped** — every comparable param is annotated `float`; this one alone has no annotation.

## B. Naming inconsistencies (same concept, two names)

5. **`add_keys` vs `add_columns`** — same documented meaning ("additional keys/columns to include in the dataframe"). Distributions + ridge use `add_keys` (`distribution.py:84`); dimensional/xyplot/spatial use `add_columns` (`dimensional.py:105`). Pick one.

6. **`line_type` vs `linetype`** — `elbow(line_type=...)` is the only one with an underscore; `cluster_outlines(linetype=...)` and `volcano(threshold_linetype=...)` (and lets-plot itself) use `linetype`.

7. **Count params: `n` vs `n_genes` vs `top_n`** — `highest_expressed_genes(n=20)` vs `markers(n_genes=20)` / `marker_genes(n_genes=5)` vs `volcano(top_n=10)`. The first three all count genes; three different names.

## C. Behavioral / default inconsistencies across sibling functions

8. **`axis_type` default differs for `expression` alone** — `expression` defaults `axis_type='axis'`; `expressions`, `dimensional`, `umap`, `tsne`, `pca` all default `None`. The singular/plural pair `expression`/`expressions` disagreeing is the most jarring.

9. **The `share_*` grid family is inconsistent across plurals:**
   - `histograms`/`violins`/`boxplots`: `share_axis` + `share_ticks`
   - `dimensionals`/`expressions`/`umaps`/`tsnes`/`pcas`/`markers`: `share_labels` + `share_axis`
   - `spatials`: `share_labels` only (no `share_axis`)
   - `ridges`: none of them

   Four different combinations for the same "shared layout" concept.

10. **`groups` accepted types differ** — dimensional/umap/violin/etc.: `Sequence[str] | str | None` (bare str ok); dotplot/heatmap/matrixplot/stacked_violin: `Sequence[str] | None` (bare str rejected). Same param name, different accepted input.

11. **`mid_point` default split** — scatter-family (`expression`/`umap`/...) default `'median'`; matrix-family (`dotplot`/`heatmap`/`matrixplot`/`stacked_violin`/`spatial`) default `'mid'`. May be intentional, but worth a conscious decision.

12. **`group_by` is positional-required in `ridge`/`ridges`** but keyword-optional everywhere else (`violin`, `boxplot`, `histogram`). Defensible (ridge needs grouping) but breaks the uniform `(data, key, *, group_by=...)` shape.

## D. Type-annotation inconsistencies (cosmetic but visible in 1.0 docs)

13. **`widths`/`heights`: `list` vs `list[float]`** — distribution plurals (`histograms`/`violins`/`boxplots`/`ridges`) use bare `list | None`; all other plurals use `list[float] | None`.

14. **`build_frame` defaults lowercase, plots capitalize** — `build_frame(observations_name='barcode', variables_name='variable')` vs every plot's `'Barcode'`/`'Variable'`. Since `build_frame`'s output is fed back via `frame=`, the column-name casing mismatch can bite users who pass a hand-built frame.

## Worth a second look (semantic, possibly intentional)

15. **`volcano(group=...)` / `volcanos(groups=...)` overload `group(s)`** — here it means "the DE comparison group(s)", whereas `groups` in `umaps`/`violins` means "filter to these categories." Same word, opposite role. Plus `volcano` has both `group` and `group_by`.

16. **`heatmap`/`matrixplot` carry both `axis` and `scale_axis`** — two axis-valued params on one function; easy to confuse. `spatial` exposes only `scale_axis`.

17. **Singular `histogram`/`violin`/`boxplot` accept `key: str | Sequence[str]`** even though plural variants exist — "give me multiple keys" has two entry points with different layout semantics (overlay vs facet). Confirm deliberate and documented.

---

Items **1-4** are straight defects to fix regardless. **5-10** are the inconsistencies most likely to confuse users and lock in bad names post-1.0.
