# Changelog

All notable changes to cellestial are documented here. Breaking changes are
called out explicitly so users can migrate between versions.

## [Unreleased]

### Added
- Added multimodal (`MuData`) support. Frames and the dimensional, distribution,
  scatter and heatmap plot families now accept a multimodal container directly,
  reading metadata and embeddings at the container level so joint embeddings
  (WNN, MOFA) can be plotted and colored by a feature from any modality.
  - Variables resolve to the modality owning them. A bare name works when it is
    unique across modalities, and `modality:name` disambiguates otherwise,
    raising `AmbiguousVariableError` rather than silently picking one.
  - Observations absent from a modality surface as NaN, aligned through the
    container's observation map.
- Added a `modality` parameter to `markers`, `marker_genes`, `marker_genes_dict`,
  `volcano`, `volcanos`, `elbow`, `highest_expressed_genes`, `heatmap`,
  `dotplot`, `stacked_violin` and `annotated_heatmap`, selecting which modality's
  stored analysis results to use. Required for a multimodal object holding more
  than one modality, and not accepted otherwise.
- Added `cellestial.datasets.pbmc_cite()`, a small CITE-seq multimodal dataset,
  and an `extension` parameter to `from_url` so it can serve `.h5mu` files.
- Added `AmbiguousVariableError`.

### Fixed
- A variable name matching more than one column is now rejected instead of
  producing wrong values. Slicing on a duplicated name yielded an interleaving
  of the matching columns for multimodal inputs, and an opaque pandas
  `InvalidIndexError` for AnnData; both now raise `AmbiguousVariableError`.
- A key stored on both axes no longer silently selects the wrong one. Metrics
  such as `total_counts` are written per observation *and* per variable, and the
  variables axis always won, so `violin(data, key="total_counts", fill="leiden")`
  built a variables frame that could not contain the grouping column. The
  grouping and aesthetic keys of the same call now break the tie; with nothing
  to break it the variables axis still wins, as before. Affects AnnData as well
  as multimodal inputs.
- `group_by` is now validated before use, so an unknown value raises
  `KeyNotFoundError` instead of a raw polars `ColumnNotFoundError` naming the
  narrowed frame. When the name exists on a modality, the error suggests the
  qualified form (`rna:celltype`). Affects AnnData as well as multimodal inputs.
- Type annotations now admit multimodal inputs. Every plotting function that
  accepts one at runtime was still annotated `AnnData`, so type checkers
  rejected valid calls.
- Widened `dimensions` from `Literal["umap", "pca", "tsne"]` to also accept any
  other embedding name. Non-standard names always worked at runtime, and joint
  embeddings such as `wnn_umap` make the narrow annotation untenable.
- Variable names containing colons now resolve. A qualified `modality:name` key
  is still tried first, but a key that does not match that reading is now
  matched literally, so ATAC peak names (`chr1:1000-2000`) and datasets that
  store variable names already prefixed (`rna:SAMD11`) work. Previously every
  variable in such a dataset was unreachable.
- Embeddings stored as a `DataFrame` rather than an array no longer fail with
  `InvalidIndexError` when materialised into a frame. Affects AnnData as well as
  multimodal inputs.
- Marker-derived plots (`heatmap`, `dotplot`, `stacked_violin`) now read the
  group column a stored ranking was computed on as the container names it. A
  ranking records that column under the modality's own name, so a container
  carrying both a global `leiden` and a modality's `rna:leiden` previously
  grouped the ranked genes by the wrong clustering. Passing the container name
  explicitly as `group_by` is now accepted too.

### Notes
- Spatial plots do not accept a multimodal object and now say so explicitly,
  naming the single-modality form.
- Only containers whose modalities share observations (`axis=0`) are supported.

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
