# Changelog

All notable changes to cellestial are documented here. Breaking changes are
called out explicitly so users can migrate between versions.

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
