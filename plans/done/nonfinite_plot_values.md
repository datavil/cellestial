# Non-finite values in plots

## Goal

Prevent Cellestial's built-in plot layers and statistics from receiving `NaN`,
`+inf`, or `-inf` in the numeric columns they consume, without dropping rows
because an unrelated metadata or tooltip column is non-finite.

This work must also distinguish non-finite source data from values created by a
scale transformation. In particular, a log scale turns zero or negative finite
values into non-finite values inside Lets-Plot. Filtering `NaN` and infinities
before rendering does not solve that case.

## Decisions

- Filter only the numeric columns required by the built-in geom or statistic.
- Keep finite zero and negative values unless the plot itself requires a
  positive domain. A later user-added log scale must not silently change the
  base plot's data policy.
- Filter long-form value columns after unpivoting so one invalid feature does
  not remove the observation's other valid features.
- Filter before KDE, aggregation, binning, scaling, or range calculations so an
  invalid value cannot contaminate a derived result.
- Do not change `build_frame()` globally. It is public and may legitimately
  carry non-finite columns that a plot does not use.
- Do not change the generic `plot()` constructor. It has no built-in geom and
  cannot know which columns later user layers will consume.
- Preserve `volcano()`'s intentional treatment of `pvalue == 0`: cap the
  resulting `+inf` rather than dropping the most significant feature. Drop
  genuinely invalid p-values separately.

## Pass 1: reproduce and lock down the behavior

1. Locate the exact tests or rendered examples producing the current
   `[violin]` and `[box_plot]` messages.
2. Record whether each offending value is:
   - already `NaN`/infinite in the source frame; or
   - finite but outside a later scale's domain, such as zero under
     `scale_y_log2()`.
3. Add small synthetic regression fixtures containing `None`, `NaN`, `+inf`,
   `-inf`, and ordinary finite values.
4. For every implementation pass below, verify both:
   - the plot's relevant data columns contain only finite values; and
   - SVG rendering completes without a Lets-Plot non-finite warning caused by
     source `NaN`/infinities.

Verification: targeted tests fail before the implementation and pass after it.

## Pass 2: statistical distribution plots

These are the highest priority because Lets-Plot performs a statistic over the
input and currently emits the reported warnings.

- In `_distribution`, replace the null-only value filter with a finite-value
  filter after unpivoting. This covers:
  - `violin`, `violins`
  - `boxplot`, `boxplots`
  - `histogram`, `histograms`
- In `ridge`, filter the plotted key before thresholding and density
  estimation. The singular fix also covers `ridges`, which delegates to it.
- In `highest_expressed_genes`, filter the long-form value column before the
  boxplot statistic.
- Keep grouping-column null filtering separate from numeric value filtering.

Files:

- [cellestial/single/core/utilities.py](../cellestial/single/core/utilities.py)
- [cellestial/single/quick/ridge.py](../cellestial/single/quick/ridge.py)
- [cellestial/single/quick/highest.py](../cellestial/single/quick/highest.py)
- [tests/test_distributions.py](../tests/test_distributions.py)
- [tests/test_coverage_boost.py](../tests/test_coverage_boost.py)

Verification: all distribution variants retain the expected finite rows and
render without source-data non-finite warnings.

## Pass 3: matrix and density computations

- `stacked_violin`: filter the long-form value column before computing variable
  ranges, group variance, and SciPy KDE polygons.
- `dotplot`: filter the long-form value column before computing group means and
  percentages.
- `heatmap` and `matrixplot`: filter after unpivoting but before aggregation,
  binning, or min-max scaling.
- `annotated_heatmap`: convert non-finite selected expression values to null
  before wide-frame binning so Polars means can ignore them per feature, then
  drop missing values from the long-form heatmap frame.
- Ensure groups or feature/group cells left with no finite observations are
  omitted predictably rather than producing invalid derived values.

Files:

- [cellestial/single/heatmap/stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py)
- [cellestial/single/heatmap/dotplot.py](../cellestial/single/heatmap/dotplot.py)
- [cellestial/single/heatmap/heatmap.py](../cellestial/single/heatmap/heatmap.py)
- [cellestial/single/heatmap/utilities.py](../cellestial/single/heatmap/utilities.py)
- [cellestial/complex/annotated_heatmap.py](../cellestial/complex/annotated_heatmap.py)
- [tests/test_heatmap.py](../tests/test_heatmap.py)
- [tests/test_annotated_heatmap.py](../tests/test_annotated_heatmap.py)

Verification: aggregation and KDE use only finite observations; one invalid
feature value does not discard other valid features from the same observation.

## Pass 4: point, line, and spatial coordinates

Filter required numeric coordinates before constructing built-in layers:

- `xyplot` and `xyplots`: plotted x and y columns.
- `scatter`: numeric columns referenced by its explicit mapping.
- `dimensional`, `dimensionals`, `umap(s)`, `tsne(s)`, `pca(s)`, and
  `expression(s)`: embedding x/y coordinates; also handle a non-finite numeric
  color key without letting it influence gradient calculations.
- `spatial` and `spatials`: point or polygon coordinates; separately handle a
  non-finite numeric color key before optional min-max scaling.
- `elbow`: PCA variance ratio.
- `markers`: ranking score used by text and optional path layers.
- `bar`: numeric mapped stat inputs, while leaving categorical count behavior
  unchanged.

Filtering must be applied in the singular shared implementations so plural
wrappers inherit it without duplicating work.

Files:

- [cellestial/single/common/xyplot.py](../cellestial/single/common/xyplot.py)
- [cellestial/single/basic/scatter.py](../cellestial/single/basic/scatter.py)
- [cellestial/single/basic/bar.py](../cellestial/single/basic/bar.py)
- [cellestial/single/core/dimensional.py](../cellestial/single/core/dimensional.py)
- [cellestial/spatial/spatial.py](../cellestial/spatial/spatial.py)
- [cellestial/single/quick/elbow.py](../cellestial/single/quick/elbow.py)
- [cellestial/single/differential/markers.py](../cellestial/single/differential/markers.py)

Verification: invalid coordinates are absent from the layer data, while a
non-finite tooltip-only column does not cause its row to be removed.

## Pass 5: differential plots and existing helpers

- Update `volcano` so:
  - finite positive p-values are transformed normally;
  - zero p-values retain the existing capped `+inf` behavior;
  - null, `NaN`, negative, or otherwise invalid p-values are dropped;
  - non-finite log-fold changes remain dropped.
- Keep `bracket` unchanged; it already filters its y observations with
  `is_not_null() & is_finite()`.
- Keep midpoint resolution unchanged; `_resolve_midpoint` already calculates
  gradients from finite values only.

Files:

- [cellestial/single/differential/volcano.py](../cellestial/single/differential/volcano.py)
- [cellestial/single/differential/utilities.py](../cellestial/single/differential/utilities.py)
- [tests/test_differential.py](../tests/test_differential.py)
- [tests/test_audit_regressions.py](../tests/test_audit_regressions.py)

Verification: zero p-values remain visible at the cap, whereas invalid
p-values and fold changes do not reach Lets-Plot.

## Pass 6: logarithmic-scale examples

Input finite filtering will not prevent warnings created by a log scale. Audit
the examples and notebook tests that add `scale_y_log2()` or
`scale_y_log10()` to distributions.

For each example, choose one explicit resolution:

- use data that is strictly positive;
- set a positive `threshold` when removing zero/non-positive observations is
  the intended visualization; or
- remove the logarithmic scale when it is not important to the example.

Do not silently remove all non-positive observations from the base plotting
functions merely to accommodate an optional user-added scale.

Files to start with:

- [cellestial/single/core/distribution.py](../cellestial/single/core/distribution.py)
- [cellestial/single/core/distributions.py](../cellestial/single/core/distributions.py)
- [test/single/violin.ipynb](../test/single/violin.ipynb)
- [test/single/boxplot.ipynb](../test/single/boxplot.ipynb)

Verification: rendered examples no longer emit the current violin/boxplot
messages, and their treatment of zero values is visible in the example code.

## Final verification

1. Run targeted non-finite regression tests after each pass.
2. Run `poetry run ruff check cellestial tests` and the project's formatting
   check.
3. Run the complete pytest suite.
4. Render the affected plots to SVG, since constructing a `PlotSpec` alone does
   not execute Lets-Plot's statistics and will not expose these warnings.
5. Confirm existing row counts and plot data are unchanged for fully finite
   inputs.

## Acceptance criteria

- Source `NaN` and infinities do not reach built-in numeric geoms or
  statistics.
- Derived KDE, aggregate, scale, and range calculations use finite data only.
- Finite inputs produce the same plot data as before.
- Non-finite tooltip-only or unused metadata does not remove a row.
- Zero p-values remain represented in volcano plots.
- Log-scale examples handle their positive-domain requirement explicitly.
- The full test suite passes without the reported Lets-Plot messages.
