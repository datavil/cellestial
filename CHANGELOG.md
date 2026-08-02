# Changelog

All notable changes to cellestial are documented here. Breaking changes are
called out explicitly so users can migrate between versions.

## [0.59.0] - 2026-08-02

### Breaking
- Renamed `mid_point` to `midpoint` across all gradient-enabled plotting APIs,
  including dimensional, heatmap, and spatial plot families.
  - Migration: replace `mid_point=...` with `midpoint=...`.
- Removed underscore-prefixed internal helpers from the `__all__` declarations of
  `cellestial.frames`, `cellestial.layers`, `cellestial.themes`, and
  `cellestial.util`. Explicit internal imports remain available, but wildcard
  imports no longer expose private names.

### Added
- Added `threshold` to `violins`, `boxplots`, and `histograms`, matching their
  singular counterparts.
- Added `point_mapping` to `violins` and `boxplots`. Mapping-only columns are now
  materialized correctly for both singular and plural distribution plots.
- Added browser-rendering checks for generated plot HTML and expanded regression
  coverage across plotting, frames, layers, spatial data, and edge cases.
- Added `CITATION.cff` and citation information to the README.

### Fixed
- Plotting functions now consistently discard null, NaN, and infinite values before
  aggregation or rendering, including dimensional, distribution, differential,
  heatmap, ridge, spatial, outline, stream, and on-data legend paths.
- `ridge` now drops observations with missing grouping values before plotting.
- Arrow axes now reject empty coordinate data with a clear error instead of failing
  while calculating bounds.
- `layout` now validates flat and nested `widths` consistently, including uneven
  plot rows and general sequence inputs.
- Corrected stale parameter names, defaults, and descriptions in distribution,
  subdimensional, volcano, stacked-violin, ridge, and spatial docstrings.

## [0.56.0] - 2026-06-27

### Breaking
- Renamed the `add_columns` parameter to `add_keys` in `dimensional`, `umap`,
  `tsne`, `pca` (and their plural forms), `xyplot`, `xyplots`, `spatial`, and
  `spatials`. This unifies the parameter with the `add_keys` already used by the
  distribution and ridge plots.
  - Migration: replace `add_columns=...` with `add_keys=...`.
- Renamed the `line_type` parameter to `linetype` in `elbow`, matching
  `cluster_outlines`, `volcano` (`threshold_linetype`), and lets-plot itself.
  - Migration: replace `line_type=...` with `linetype=...`.
- Changed the default `axis_type` of `expression` from `"axis"` to `None`, so it
  now matches `expressions`, `dimensional`, `umap`, `tsne`, and `pca`. By default
  the axis text, ticks, and lines are now hidden instead of shown.
  - Migration: pass `axis_type="axis"` to restore the previous full-axis default.
- Changed the default `observations_name` to `"Barcode"` and `variables_name` to
  `"Variable"` in `build_frame`, `anndata_observations_frame`, and
  `anndata_variables_frame` (were `"barcode"`/`"variable"`). The frame builders now
  emit the same column names the plots expect, so a hand-built frame passed via
  `frame=` lines up by default.
  - Migration: pass `observations_name="barcode"` / `variables_name="variable"` to
    keep the old lowercase column names.

### Fixed
- `stream`: `cutoff_percentile` is now annotated `float | None` (was `None`).
- `volcanos`: added the missing `-> SupPlotsSpec` return annotation.
- `highest_expressed_genes`: `size` is now annotated `float` (was untyped).
- `histograms`/`violins`/`boxplots`/`ridges`: `widths`/`heights` are now annotated
  `list[float] | None` (were bare `list | None`), matching the other grid plurals.
- `histograms`/`violins`: `share_ticks` docstrings now say `default=False` (matched
  the signature; were wrongly `default=True`), and the `share_ticks` description no
  longer reads "share the labels".
