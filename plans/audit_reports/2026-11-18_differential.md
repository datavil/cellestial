# Audit: differential plots, marker helpers, brackets (2026-11-18)

Branch `claude/audit-differential-078c2b`. Scope: the rows of `plans/audit_AI.md`
with blank check cells for `cellestial/single/differential/`, the marker helpers in
`cellestial/util/markers.py`, the bracket layer and the group filter helpers.

Method: line-by-line read; run on pbmc3k (`cl.datasets.pbmc3k(cache_directory="data")`,
`data.raw = None`, `rank_genes_groups(groupby="cell_type_lvl1", method="wilcoxon", pts=True)`);
independent cross-checks with numpy, scipy and statsmodels; key plots rendered to PNG and
inspected; a failing test written before each fix (`tests/test_audit_differential.py`);
a bug-hunting subagent over the final diff.

## Verdicts

| Function | File | Verdict | Notes |
| --- | --- | --- | --- |
| `volcano` | `single/differential/volcano.py` | fixed | Bugs 1, 2, 3 (input validation). Classification, cap and labels verified. |
| `volcanos` | `single/differential/volcano.py` | fixed | Bug 4 (ranking recomputed per panel). |
| `_build_volcano_frame` | `single/differential/utilities.py` | fixed | Bug 5 (missing fields). Numbers match an independent computation. |
| `_resolve_pvalue_column` | `single/differential/utilities.py` | OK | |
| `markers` | `single/differential/markers.py` | fixed | Bug 6 (empty `groups`), fix lives in `_build_markers_frame`. |
| `_build_markers_frame` | `single/differential/utilities.py` | fixed | Bug 6. Order and ranks verified. |
| `marker_genes` | `util/markers.py` | OK | Dedup keeps first occurrence, group order kept. |
| `marker_genes_dict` | `util/markers.py` | OK | |
| `_marker_names_per_group` | `util/markers.py` | OK | `groups=[]` returns empty; see suggestions. |
| `_filter_groups` | `util/utilities.py` | OK | Null rows: kept by `drop`, excluded by `groups`. Tests added. |
| `_require_categories` | `util/utilities.py` | OK | No close-match hint; see suggestions. |
| `bracket` | `layers/bracket.py` | fixed | Bugs 7, 8. p-values verified against scipy + BH. |
| `_compute_bracket_frame` | (removed) | OK | No longer exists: the layer now delegates to ggrust `stat_compare_means`; its successor `_bracket_layout` / `_expand_comparisons` was audited instead. |
| `_range_inclusive` | `util/utilities.py` | OK | Dead in the library (only used by old notebooks and tests); see suggestions. |

## Numerical cross-checks (pbmc3k)

- **Volcano frame** for all 4 groups vs a numpy recomputation from
  `sc.get.rank_genes_groups_df`: 0 significance mismatches, `-log10(padj)` within
  5e-5 (the 4-decimal rounding). Underflowed `padj == 0` per group: B Cells 299,
  Erythroid 44, Lymphocytes 164, Monocytes 228; all drawn at
  `1.05 * finite_max + 1`, none dropped.
- **Boundaries**: `p == pvalue_threshold` is `ns` (strict `<`); `|logFC| == threshold`
  is up/down (inclusive `>=`); `logFC == 0` with `p == 0` is `ns`. Locked in by
  `test_volcano_frame_classifies_threshold_boundaries`.
- **Brackets** (CD3D by `cell_type_lvl1`, BH): Mann-Whitney labels
  `<2e-16, <2e-16, <2e-16, 0.41, 8e-12, 2.8e-10` match scipy `mannwhitneyu` + statsmodels
  exactly; `test="ttest"` labels `0.14, 4.8e-16, <2e-16` match Welch (`equal_var=False`),
  not Student (`0.15, 5.2e-18, 5.9e-20`).
- **Markers / marker_genes**: names and scores equal `uns["rank_genes_groups"]` order;
  `marker_genes(n_genes=3)` dedups across groups keeping first occurrence.

Rendered and inspected: `volcano` (B Cells), `volcanos` (2 and 3 panels), `markers`
(4 panels, `rank_color`, `line`), `violin + bracket(label="pvalue")`, a bracket with
an explicit low `y_position`, and a bracket on constant data. Labels, legends and
geoms stay inside the plot area.

## Confirmed bugs and fixes

All tests are in `tests/test_audit_differential.py`; each failed before its fix.

1. **Negative `logfoldchange_threshold` silently misclassifies.**
   Repro: `cl.volcano(data, "B Cells", logfoldchange_threshold=-1)` renders; every
   significant feature with `logFC >= -1` is called `up`. Fix: `InvalidArgumentError`
   when `< 0`. Test: `test_volcano_rejects_negative_logfoldchange_threshold`.
   Recorded under `### Breaking` (a call that used to run now raises).
2. **`top_n < 0` labels almost everything.** polars `head(-1)` keeps all rows but the
   last. Fix: `InvalidArgumentError`. Test: `test_volcano_rejects_negative_top_n`.
   `nonsignificant_subsample=-1` crashed in polars with a dtype message; same fix.
   Test: `test_volcano_rejects_negative_nonsignificant_subsample`.
3. **`pvalue_threshold <= 0` failed late in the renderer**
   (`geom_hline requires aesthetics yintercept`, from `-log10(0)`). Fix:
   `InvalidArgumentError` unless `0 < pvalue_threshold <= 1`, before any ranking runs.
   Test: `test_volcano_rejects_pvalue_threshold_outside_unit_interval`.
4. **`volcanos(group_by=...)` recomputed the ranking for every panel** when
   `rank_genes_kwargs` holds keys scanpy does not store in `params` (`pts`, `n_genes`,
   `tie_correct`, ...): the staleness check sees them as changed every time.
   Repro on pbmc3k: 2 panels with `{"method": "wilcoxon", "pts": True}` gave 2
   `rank_genes_groups` calls. The docstring promises one. Fix: only the first panel
   passes `group_by`. Real data after the fix: 3 panels, 1 call.
   Test: `test_volcanos_computes_the_ranking_once`.
5. **Ranking without fold changes or p-values** (e.g. `method="logreg"`) raised a bare
   `KeyError: 'logfoldchanges'`. Fix: `KeyNotFoundError` naming the missing fields and
   the tests that provide them. Test: `test_volcano_names_missing_ranking_fields`.
6. **`markers(groups=[])`** raised polars `ValueError: cannot concat empty list`.
   Fix: `EmptySelectionError` (a `ValueError`) in `_build_markers_frame`.
   Test: `test_markers_rejects_empty_groups`.
7. **`bracket` with an explicit `y_position` below the data top shrank the canvas**,
   down to a negative height: `cl.bracket(y_position=-5, y_step=0.01)` gave
   `ggsize(600, -125.0)`. Fix: clamp the growth at 0.
   Test: `test_bracket_below_the_data_never_shrinks_the_canvas`.
8. **`bracket` on a `y` column with no finite values** crashed with
   `TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'` before
   the layout's own "no valid comparisons" check could run. Fix: same `ValueError`
   wording, raised early. Test: `test_bracket_rejects_a_column_without_finite_values`.

Edge-case coverage added (passed before and after): threshold boundaries, `inf` kept
in the frame and capped in the plot, marker ranking order, `_filter_groups` with
null categories (`drop` keeps nulls, `groups` excludes them, `"None"` is not a category).

## 2026-08-05 math audit: open items

| Item | Status |
| --- | --- |
| `bracket(test="ttest")` is Student's | **Resolved.** ggrust `stat_compare_means` runs Welch; verified numerically above and by `test_bracket_ttest_does_not_assume_equal_variances`. |
| Brackets test the plot's filtered frame | **Resolved (documented).** `bracket` Notes and examples state it. |
| `p = 0` renders as `"p = 0"` | **Resolved.** ggrust prints `<2e-16` (seen on CD3D). |
| Volcano inherits scanpy's logFC pseudocount artifact | **Still open.** pbmc3k B Cells (wilcoxon): 8972 of 23427 called `down`; 6023 of those are detected in under 1% of the group's cells; 10653 features have `logFC < -15`. Needs the `min_fraction_expressed` decision in `plans/math_audit_remaining.md` item 8. |
| Polygon holes merged into the exterior ring | **Still open** (outside this audit's scope): `cellestial/spatial/utilities.py:208` still calls `shapely.get_coordinates(geoms)`. |

## Open issues not fixed

- **Constant `y` in `bracket`**: with zero data range every bracket is drawn at the
  same height on the data line (the step is sized from the range). Labels are `NA`
  anyway, since the tests are undefined. Left open: degenerate input, and the right
  fallback height is a design choice (e.g. treat the range as 1, like the scale does).
- **`volcano` errors are bare `KeyError`** for a missing key or group, and do not
  suggest a close match. Switching to `KeyNotFoundError` would break `except KeyError`
  because `KeyNotFoundError` subclasses `Exception`, not `KeyError` (AGENTS.md says the
  error classes subclass the matching builtin). Needs a decision on `errors.py` first.
- **pbmc3k annotation looks swapped**: the bundled `cell_type_lvl1` label "B Cells"
  has HBB mean 8.2 and CD74 0.06, "Erythroid" has CD74 3.0; the top markers of
  "B Cells" are HBB/HBA2/HBA1. Not a plotting bug; the dataset labels need checking.

## Improvement suggestions (lower priority)

- Add the feature name as a final sort key for volcano labels so ties on both
  `-log10(p)` and the 4-decimal-rounded `|logFC|` resolve by name, not sort order.
- `volcanos(groups=[])` fails in `gggrid` with ggrust's "plots is empty" message;
  an `EmptySelectionError` up front would name the parameter.
- `marker_genes(groups=[])` returns `[]`, which then fails later in the plot that
  consumes it; consider `EmptySelectionError` there too.
- `markers(ncol=0)` raises `ZeroDivisionError`; plural functions in general do not
  validate `ncol >= 1`.
- `bracket(y="missing")` surfaces polars `ColumnNotFoundError`; a
  `MissingAestheticError` naming the column would match the other bracket errors.
- `_require_categories` could add a "did you mean" hint (AGENTS.md error rules).
- `_range_inclusive` has no callers in the library; consider removing it.
- The new early guard in `bracket` raises a plain `ValueError`, matching the
  layout's existing error for the same condition; both could move to a dedicated
  class in `errors.py`. (Raised by the bug-hunting subagent, which confirmed no
  bugs in the fixes.)
- `_build_markers_frame` drops non-finite scores after taking the top `n_genes`, so a panel
  may show fewer genes than asked, with a gap in ranks. Unlikely with real rankings.
- `volcano` / `markers` use the flattened `if not isinstance(...): raise` form at the
  top of the plot body; the per-modality dispatch below keeps it harmless.

## Follow-up fixes

Branch `claude/fix-diff-heatmap-3f96af`. Each fix has a test that failed first, in
`tests/test_followup_differential.py` unless noted; run on pbmc3k.

| Item | Outcome |
| --- | --- |
| Volcano logFC artifact (math audit item 8) | **Fixed.** `volcano`/`volcanos` take `min_fraction_expressed: float \| None = None`. Reads `pts` from the stored ranking; when absent raises `KeyNotFoundError` naming `group_by` / `pts=True`; a `group_by` recompute passes `pts=True` (a ranking without `pts` counts as stale when the filter is set). Values outside [0, 1] raise `InvalidArgumentError`, as do non-unique feature names (found by the bug-hunting subagent: pandas `reindex` raised a bare `ValueError`). pbmc3k B Cells at 0.1: 873 of 23427 features remain (169 up, 442 down); rendered and inspected. CHANGELOG `### Added`; item 8 marked resolved. |
| `volcano` bare `KeyError` | **Fixed.** Missing key or group raises `KeyNotFoundError` with a case-insensitive close match (`did you mean 'A'`). `KeyNotFoundError` still subclasses `Exception` on this base (fix-frames had not landed), so three `tests/test_differential.py` assertions moved from `KeyError` to `KeyNotFoundError`. CHANGELOG `### Breaking`. |
| Constant `y` in `bracket` | **Fixed.** Zero range is taken as 1 for both the padding and the pixel-sized step; brackets stack above the data (rendered). |
| `bracket(y="missing")` | **Fixed.** A missing `x` or `y` column raises `MissingAestheticError` with a close match. |
| `bracket` plain `ValueError` guards | **Fixed.** Too few finite observations / no pairs raise `InvalidComparisonError`; unknown `test`/`correction`/`label` raise `InvalidArgumentError`. Both subclass `ValueError`, so no new class in `errors.py` was needed. |
| Volcano label ties | **Fixed.** Feature name is the final ascending sort key. |
| `volcanos(groups=[])`, `marker_genes(groups=[])` | **Fixed.** `EmptySelectionError` (also `marker_genes_dict`). CHANGELOG `### Breaking` for the marker helpers. |
| Plural `ncol >= 1` | **Fixed** for `volcanos` and `markers` (`InvalidArgumentError`). |
| `_require_categories` hint | **Fixed.** Case-insensitive close match per unknown value. |
| `_build_markers_frame` non-finite scores | **Fixed.** Non-finite scores are dropped per group before taking the top `n_genes`; ranks stay gapless. |
| Bug-hunting subagent | One confirmed bug (duplicate feature names with `min_fraction_expressed`), fixed with `test_volcano_detection_rate_with_duplicate_feature_names_names_the_fix`. No regressions found elsewhere. |
| `_range_inclusive` | **Reported, not deleted.** Still no callers in `cellestial/`. |
| pbmc3k annotation | **Investigated**, see `plans/pbmc3k_labels.md`: "B Cells" and "Erythroid" are swapped, the null cluster is erythroblasts, and the tissue is bone marrow. Dataset unchanged. |
