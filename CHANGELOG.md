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

### Fixed
- `stream`: `cutoff_percentile` is now annotated `float | None` (was `None`).
- `volcanos`: added the missing `-> SupPlotsSpec` return annotation.
- `highest_expressed_genes`: `size` is now annotated `float` (was untyped).
