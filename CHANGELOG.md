# Changelog

All notable changes to cellestial are documented here. Breaking changes are
called out explicitly so users can migrate between versions.

Entries for 0.58.2 and earlier were reconstructed from the commit history after
the fact, so they summarize each release in a line or two rather than list every
change. Versions absent from the list were never released.

## [Unreleased]

### Breaking
- `bracket(test="ttest")` is now Welch's t-test rather than the pooled-variance
  one, matching R's `t.test()` and scanpy. P-values change.
  - Migration: pass `test="student"`, a new option, for the previous behaviour.
- `volcano` and `volcanos` name their p-value column after the values it holds.
  Under the default `use_adjusted_pvalue=True` the column, tooltip entry and
  y-axis title read `pvalue_adj` and `-log10(Padj)`; raw p-values are unaffected.
  - Migration: pass `pvalue_column="pvalue"` to keep the old column name.

### Fixed
- `volcano` and `volcanos` break gene-label ties by absolute log fold change, so
  the labels picked among p-values that underflow to zero are the strongest
  features and stable between builds.
- `stream` no longer fails on a two-vertex streamline, and drops streamlines
  whose midpoint vertex is repeated.
- `_range_inclusive` reaches both endpoints without overshooting them, and
  accepts a stop below the start.

### Notes
- `bracket` tests the rows the plot draws, so a plot built with `threshold`
  restricts the comparison to the observations above it. Documented only;
  behaviour unchanged.

## [0.60.0] - 2026-08-06

### Added
- Added multimodal (`MuData`) support to frames and the dimensional,
  distribution, scatter and heatmap plot families, so joint embeddings (WNN,
  MOFA) can be plotted and coloured by a feature from any modality. A bare
  variable name works when it is unique across modalities, `modality:name`
  disambiguates otherwise, and observations absent from a modality surface as
  NaN.
- Added a `modality` parameter to `markers`, `marker_genes`, `marker_genes_dict`,
  `volcano`, `volcanos`, `elbow`, `highest_expressed_genes`, `heatmap`,
  `dotplot`, `stacked_violin` and `annotated_heatmap`, selecting which modality's
  stored analysis results to use.
- Added `cellestial.datasets.pbmc_cite()`, a small CITE-seq multimodal dataset,
  and an `extension` parameter to `from_url` so it can serve `.h5mu` files.
- Added `AmbiguousVariableError`.

### Fixed
- A variable name matching more than one column raises `AmbiguousVariableError`
  instead of returning the matching columns interleaved.
- A key stored on both axes no longer silently resolves to the variables axis;
  the grouping and aesthetic keys of the same call break the tie. Affects AnnData
  as well as multimodal inputs.
- `group_by` is validated up front, raising `KeyNotFoundError` and suggesting the
  qualified form (`rna:celltype`). Affects AnnData as well as multimodal inputs.
- Type annotations admit multimodal inputs, and `dimensions` accepts embedding
  names beyond `umap`, `pca` and `tsne`.
- Variable names containing colons resolve, so ATAC peaks (`chr1:1000-2000`) and
  already-prefixed names (`rna:SAMD11`) work.
- Embeddings stored as a `DataFrame` rather than an array materialise into a
  frame. Affects AnnData as well as multimodal inputs.
- Marker-derived plots (`heatmap`, `dotplot`, `stacked_violin`) read the group
  column of a stored ranking under the name the container uses, instead of
  grouping by the wrong clustering.

### Notes
- Spatial plots do not accept a multimodal object; pass the modality and a
  pre-built frame:
  `spatial(data['rna'], key='prot:CD3', frame=build_frame(data, variable_keys=['prot:CD3']))`.
- Only containers whose modalities share observations (`axis=0`) are supported.

## [0.59.0] - 2026-08-02

### Breaking
- Renamed `mid_point` to `midpoint` across all gradient-enabled plotting APIs,
  including dimensional, heatmap, and spatial plot families.
  - Migration: replace `mid_point=...` with `midpoint=...`.
- Removed underscore-prefixed internal helpers from the `__all__` of
  `cellestial.frames`, `cellestial.layers`, `cellestial.themes` and
  `cellestial.util`, so wildcard imports no longer expose them. Explicit imports
  still work.

### Added
- Added `threshold` to `violins`, `boxplots` and `histograms`, and
  `point_mapping` to `violins` and `boxplots`, matching their singular
  counterparts.
- Added browser-rendering checks for generated plot HTML and expanded regression
  coverage across plotting, frames, layers, spatial data and edge cases.
- Added `CITATION.cff` and citation information to the README.

### Fixed
- Null, NaN and infinite values are discarded before aggregation or rendering
  across every plot family.
- `ridge` drops observations with missing grouping values before plotting.
- Arrow axes reject empty coordinate data with a clear error instead of failing
  while calculating bounds.
- `layout` validates flat and nested `widths` consistently.
- Corrected stale parameter names, defaults and descriptions in the distribution,
  subdimensional, volcano, stacked-violin, ridge and spatial docstrings.

## [0.58.2] - 2026-08-01

- Added citation metadata (`CITATION.cff`, Zenodo badge) and multithreaded test
  runs. Started the non-finite filtering, arrow-bounds and nested `widths` work
  that 0.59.0 completed.

## [0.58.1] - 2026-07-23

- Fixed heatmap spacing and the `axis=1` cases.

## [0.58.0] - 2026-07-20

- Fixed `scatter` and `bar` ignoring an explicit `axis=0`, a crash in
  `xyplot(include_dimensions=True)`, brackets silently reporting invalid tests
  as "ns", and `mid_point` handling for gradients.
- Moved several hard dependencies to optional, and corrected return annotations
  and docstrings across the API.

## [0.57.1] - 2026-07-19

- Forced the lets-plot 4.11 upgrade so fresh installs pick it up.

## [0.57.0] - 2026-07-02

- Added `annotated_heatmap`, with transposition and per-track annotation bars.
- Applied the second batch of v1.0 API audit fixes, dropped `variable` from
  tooltips, and moved to lets-plot 4.11 with halo options on the on-data legend.

## [0.56.0] - 2026-06-27

### Breaking
- Renamed `add_columns` to `add_keys` in the dimensional, `xyplot` and `spatial`
  families, matching the distribution and ridge plots.
  - Migration: replace `add_columns=...` with `add_keys=...`.
- Renamed `line_type` to `linetype` in `elbow`, matching the other plots and
  lets-plot itself.
  - Migration: replace `line_type=...` with `linetype=...`.
- Changed the default `axis_type` of `expression` from `"axis"` to `None`, so it
  hides the axis text, ticks and lines like the other dimensional plots.
  - Migration: pass `axis_type="axis"` to restore the previous full-axis default.
- Changed the default `observations_name` and `variables_name` to `"Barcode"` and
  `"Variable"` in `build_frame`, `anndata_observations_frame` and
  `anndata_variables_frame`, so a hand-built frame passed via `frame=` lines up
  with what the plots expect.
  - Migration: pass `observations_name="barcode"` / `variables_name="variable"` to
    keep the old lowercase column names.

### Fixed
- Corrected annotations: `cutoff_percentile` on `stream`, the missing return type
  on `volcanos`, `size` on `highest_expressed_genes`, and `widths`/`heights` on
  the grid plurals.
- `histograms`/`violins`: `share_ticks` docstrings say `default=False` and no
  longer read "share the labels".

## [0.55.0] - 2026-06-24

- Rewrote the public error messages so they name the offending key and the
  accepted values, and diversified the documented examples.

## [0.54.0] - 2026-06-10

- Added the `layout` utility for composing plots into custom grids.

## [0.53.1] - 2026-06-10

- Packaging-only release.

## [0.53.0] - 2026-06-05

- Datasets are now served from Hugging Face, and `spatialdata` is imported
  lazily so it is only needed when spatial data is actually plotted.

## [0.51.0] - 2026-05-28

- Frame building now materializes only the columns needed for scaling, cutting
  both time and memory. Added `CONTRIBUTING.md`.

## [0.50.1] - 2026-05-23

- Added `groups` and `drop` parameters, and a fallback to the high-resolution
  tissue image when no other is available.
- Extended CI to Windows and Python 3.12, and added export coverage tests.

## [0.50.0] - 2026-05-21

- Multi-plots now build the frame once and share it across panels instead of
  rebuilding it per panel (#70).

## [0.49.3] - 2026-05-20

- README and asset polish: logo placement, absolute asset links, refreshed
  spatial figure.

## [0.49.2] - 2026-05-20

- Added marker extraction and removed the remaining deprecated code paths ahead
  of 1.0.

## [0.49.1] - 2026-05-20

- Added binning to `heatmap`, switched arrow layers to paths, and replaced the
  benchmark suite with a more comprehensive one.

## [0.49.0] - 2026-05-18

- Up to 200x faster frame building, and a 5-7x faster `highest_expressed_genes`
  that uses orders of magnitude less memory. Dataset downloads use a larger
  block size.

## [0.48.4] - 2026-05-17

- `outline` now raises when the grouping key is missing instead of failing
  later, and the docstring summaries were corrected.

## [0.48.3] - 2026-05-17

- Docstring cleanup: removed the "accepts" blocks and stray double backticks,
  and moved expositional text into Notes sections.

## [0.48.2] - 2026-05-17

- Added `dendrogram_key` and `pca_key`, accepted embedding keys written without
  the `X_` prefix, and warned when a spatial scale factor is missing.

## [0.48.1] - 2026-05-17

- Added a fallback for embedding keys, so a missing exact name resolves to a
  close match rather than raising.

## [0.48.0] - 2026-05-17

- Added CI: lint, a Linux and macOS test matrix over Python 3.11 and 3.13, and a
  build check.
- `spatial` now defaults to the low-resolution tissue image, and the slice
  parameter was replaced.

## [0.47.1] - 2026-05-16

- Follow-up patch to the dataset loading fix.

## [0.47.0] - 2026-05-16

- Fixed dataset loading.

## [0.46.0] - 2026-05-15

- Renamed the stored-ranking accessor to `markers` and rewrote its warnings.

## [0.45.0] - 2026-05-14

- Added the `markers` plot, and categorical coloring for `spatial`.

## [0.44.0] - 2026-05-13

- Added the `elbow` plot.
- Fixed heatmap group-bar placement, dotplot and stacked-violin key label sizes,
  and on-data legend positioning.

## [0.43.0] - 2026-05-12

- The on-data legend gained a background and better label positioning.

## [0.42.0] - 2026-05-11

- Added support for stored `rank_genes_groups` results (#68) and
  `datasets.from_url`, and loosened the dependency pins.

## [0.40.0] - 2026-05-10

- Added the `datasets` module and a quickstart guide.
- Brackets gained group-versus-rest and wildcard comparisons, validated with the
  new `InvalidComparisonError`, and the remaining generic exceptions became
  typed errors.

## [0.39.0] - 2026-05-09

- Heatmap variants accept a dict of groups, `SpatialData` inputs are supported,
  and integer columns that are really categories are detected rather than
  treated as continuous.
- `group_by` is now required where it was previously ambiguous.

## [0.33.0] - 2026-05-04

- `spatial` accepts non-Visium data, and tooltip resolution and validation moved
  into the frame builder, with better null filtering.

## [0.32.0] - 2026-04-28

- Corrected the `spatial` and `spatials` docstrings.

## [0.30.0] - 2026-04-28

- Added `spatial` and `spatials`, `volcano`, matrixplot and stacked violin, and
  turned the on-data legend into a layer.

## [0.21.0] - 2026-04-23

- Added `heatmap`, dendrograms with color and size scaling on `dotplot`, and the
  `bracket` layer. Layers can now be built without passing a plot.
- Added the pytest suite and the benchmarks. Renamed the `velocity` parameter to
  `velocity_key`.

## [0.14.0] - 2026-04-10

- Added `ridge` and `ridges` and `highest_expressed_genes`, and gave the
  distribution plots `group_by` and `point_mapping`.

## [0.13.1] - 2026-04-05

- Point size can be driven by a mapping rather than a constant.

## [0.13.0] - 2026-03-31

- Added the `stream` layer for velocity streamlines (#16).

## [0.12.0] - 2026-03-28

- Documentation build fixes, most visibly the missing function signatures, plus
  a wider set of examples.

## [0.11.0] - 2026-03-17

- Added `threshold` to the distribution plots, renamed the arrow-axis layer, and
  made `LetsPlot.setup_html()` fire automatically on import.

## [0.10.4] - 2026-03-10

- Added the Sphinx documentation site.
- Frame building moved to lazy evaluation, `scipy` and `skimage` imports moved
  inside the functions that need them, and several distribution plot bugs were
  fixed.

## [0.10.3] - 2026-02-28

- Added cluster highlighting (#20) and a `mapping` parameter to the dimensional
  and distribution plots, and sped up frame building.
- Removed the scatter theme and the `group` parameter of the distribution plots.

## [0.9.0] - 2026-02-22

- Added `bar`, a simple `heatmap` (#45) and a ggplot-style scatter with axis
  prediction, and enabled faceting in the dimensional and distribution plots.
- Re-implemented `dotplot`, standardized the observation and variable parameter
  names, and simplified the tooltip logic (#43).

## [0.8.0] - 2026-01-25

- Restructured the package into frames, layers, themes and util modules around a
  shared `build_frame`.
- Added `scatter` and `scatters`, sina and point geometries in the distribution
  plots (#34), and `guides` in the multi-plots (#36).

## [0.7.1] - 2025-05-04

- Scatter and distribution plots accept variables as keys, and the grid plots no
  longer accept a bare string where a sequence is meant.

## [0.7.0] - 2025-03-19

- Added `xyplot`, and docstrings for the distribution plots (#12, #13).

## [0.6.0] - 2025-03-10

- Dimensional plots accept `key=None`, and slicing and retrieval helpers were
  added alongside a modularized dimensional implementation.

## [0.5.1] - 2025-02-16

- On-data legend labels are placed at weighted means, which keeps them inside
  their cluster.

## [0.5.0] - 2025-02-16

- Added the on-data legend (#8) and docstrings for the grid plots (#13).

## [0.4.0] - 2025-02-05

- Added `scatter` and `dotplot`, color gradients, and the hand-picked palettes.
  Dimensional plots accept observation keys.

## [0.2.0] - 2024-12-19

- Added `violin` and `boxplot`, the multi-panel dimensional plots with shared
  axes, and the first documentation site.

## [0.1.4] - 2024-11-22

- Early release.
