# Changelog

All notable changes to cellestial are documented here. Breaking changes are
called out explicitly so users can migrate between versions.

Entries for 0.58.2 and earlier were reconstructed from the commit history after
the fact, so they summarize each release in a line or two rather than list every
change. Versions absent from the list were never released.

## [Unreleased]

### Breaking

- `anndata_observations_frame`, `anndata_variables_frame` and
  `anndata_variable_columns` are no longer exported from `cellestial.frames`
  (now private helpers), and `DeferredLayer` is no longer exported from
  `cellestial.layers`. Layer functions (`bracket`, `stream`, ...) behave as
  before. Migration: build frames with `build_frame`; do not import or
  construct `DeferredLayer`, add the layer returned by the layer function.
- `SpatialLibraryError` is removed; it was never raised. Migration: drop it from
  imports and `except` clauses.

- `groups` is keyword-only in `markers`, `marker_genes` and
  `marker_genes_dict`, matching the rest of the API (data and the plotted
  keys positional, everything else by name). Migration:
  `markers(data, ["0"])` -> `markers(data, groups=["0"])`, and likewise for
  `marker_genes` and `marker_genes_dict`.
- The stored ranking is chosen with `ranking_key` (was `key`) in `markers`,
  `volcano`, `volcanos`, `marker_genes` and `marker_genes_dict`, so `key`
  always means the column or feature to plot. The heatmap family (`heatmap`,
  `matrixplot`, `dotplot`, `stacked_violin`, `tracksplot`) takes
  `markers: bool` plus `ranking_key` (was `markers: bool | str`).
  `ranking_key` is ignored when `markers` is False. Old-style calls raise:
  `key=` in `markers`/`volcano(s)` raises `UnknownParameterError` (a
  `TypeError`) instead of reaching the geom, and a string `markers` raises
  `InvalidArgumentTypeError`. Migration: `key="x"` -> `ranking_key="x"`;
  `markers="x"` -> `markers=True, ranking_key="x"`.
- Unknown keys whose axis cannot be determined (e.g. `violin(data, "not_a_key")`),
  tooltip fields missing from the data and a missing stored PCA variance now
  raise `KeyNotFoundError` (a `KeyError`) instead of `ValueError`. Keys that
  mix observation-only and variable-only columns raise `InvalidArgumentError`.
  Migration: catch `KeyError` (or `KeyNotFoundError`) instead of `ValueError`
  for these cases.
- `build_frame` returns categorical metadata as `pl.Enum` in the source
  category order (was `pl.Categorical`, ordered by first appearance), so
  legends and group axes list categories as the data defines them (e.g.
  clusters `0, 1, 2, ...` instead of `0, 2, 3, ...`). Migration: code that
  checks `dtype == pl.Categorical` on a built frame should also accept
  `pl.Enum`; compare or join the column with strings after
  `.cast(pl.String)`.
- On a multimodal object, a bare `group_by` that names both a container column
  and a column of `modality` (e.g. `leiden` on `pbmc_cite`) raises
  `AmbiguousColumnError` in plots that use stored results (`markers`,
  `dendrogram`). `dotplot`, `matrixplot` and `stacked_violin` used to plot the
  container's column while clustering the modality's, and the per-cell plots
  failed with `DendrogramMismatchError`. Migration: pass the qualified name,
  `group_by="rna:leiden"`.
- A string other than `"none"` for `tooltips` raises `InvalidArgumentError`
  (it used to reach the renderer unchecked). Migration: pass a list for one
  column, `tooltips=["CD14"]`.

- `volcano` and `volcanos` now reject a negative `logfoldchange_threshold`,
  `top_n` or `nonsignificant_subsample` with `InvalidArgumentError`. A negative
  fold change threshold used to call features with negative fold changes `up`,
  and `top_n=-1` labelled every significant feature but one. Migration: pass
  the absolute threshold (`1.0`, not `-1.0`) and `0` or `None` to disable labels.

- `volcano` raises `KeyNotFoundError` (with a close match) instead of a bare
  `KeyError` for a missing ranking key or group. Migration: catch
  `KeyNotFoundError`; once it subclasses `KeyError`, `except KeyError` works
  again.
- `marker_genes(groups=[])` and `marker_genes_dict(groups=[])` raise
  `EmptySelectionError` instead of returning an empty result that failed later.
- `bracket` raises `InvalidArgumentError` for an unknown `test`, `correction`
  or `label`, and `InvalidComparisonError` when there are too few finite
  observations. Both subclass `ValueError`, so `except ValueError` still works.

- `dimensionals`, `umaps`, `tsnes` and `pcas` raise `KeyNotFoundError` for an
  unknown key instead of a `ValueError` about tooltip fields. Migration: catch
  `KeyNotFoundError` (or `Exception`) instead of `ValueError`.

- `spatial` and `spatials` raise `KeyNotFoundError` (with a close match) for an
  unknown `table_name`, `image_name`, `shapes_name`, `coordinate_system`,
  `library_id` or `spatial_key`, instead of `KeyError`. Ambiguous or missing
  components raise `InvalidArgumentError` or `EmptySelectionError` (both
  `ValueError` subclasses) instead of `ValueError`, and a non-SpatialData object
  passed where one is needed raises `UnsupportedDataTypeError` (a `TypeError`).
  Migration: catch `KeyNotFoundError` instead of `KeyError`.
- `spatial` and `spatials` raise `InvalidArgumentError` when `mapping` sets an
  aesthetic they map themselves (`x`, `y`, `group` for polygons, and `color`, or
  `fill` for polygons, when `key` is given), instead of a raw `TypeError`.
  Migration: drop the aesthetic from `mapping`, or pass `key=None` and map
  `color`/`fill` in `mapping` (which now works).
- `spatial(polygon=True)` with a `frame` whose rows differ from the table raises
  `ConflictingLengthError`; polygons are now matched to frame rows by position,
  as points already were. Migration: build `frame` from the same table without
  filtering rows, and use `groups`/`drop` to filter.
- `use_key` is renamed to `embedding_key` in `umap`, `pca`, `tsne`, `expression`,
  `dimensional`, their plurals `umaps`, `pcas`, `tsnes`, `expressions`,
  `dimensionals`, and in `embedding_density` and `blend`, matching the other
  stored-result keys (`pca_key`, `dendrogram_key`, ...). Passing `use_key=` or
  scanpy's `basis=` now raises `UnknownParameterError` (a `TypeError`)
  instead of being silently forwarded to the geom. Migration: replace
  `use_key="X_umap_harmony"` (or `basis=...`) with
  `embedding_key="X_umap_harmony"`.
- `save` is a thin wrapper over `ggsave`: `save(plot, filename, *, scale=2.0,
  **ggsave_kwargs)`. It no longer sets `dpi=300` and `unit="in"`, so outputs
  follow the `ggsave` defaults (a default `cl.umap` PNG is 1200x800 px at 96 dpi,
  was 3750x2500 px at 300 dpi). An unknown keyword raises `TypeError`, and
  `GGBunch` is dropped from the accepted types (ggrust never supported it).
  Migration: pass `dpi=300, unit="in"` to keep the old output; combine plots
  with `gggrid`.

### Added

- Public `cellestial.errors` module (also `cl.errors`) re-exporting every error
  class and `CellestialWarning`. `cellestial.util.errors` keeps working.
- `CellestialError` base class: every cellestial error subclasses it as well as
  its builtin (`KeyNotFoundError` is still a `KeyError`), so
  `except cl.errors.CellestialError` catches anything cellestial raised.

- `volcano` and `volcanos` take `min_fraction_expressed` (default `None`) to
  drop features detected in fewer than that fraction of the group's cells
  (e.g. `0.1`). Such features get huge fold changes from the pseudocount and
  used to dominate the plot. It reads the detection rates stored with the
  ranking; with `group_by` they are recorded when the ranking is (re)computed,
  otherwise a missing record raises `KeyNotFoundError` naming the fix.

### Changed

- `datasets.pbmc3k()` now downloads `pbmc3k_pped_v2.h5ad` with corrected
  `cell_type_lvl1` labels, following the scanpy tutorial's coarse annotation of
  `leiden_res_0.02`. The old file had "B Cells" and "Erythroid" swapped and left
  cluster 4 (1767 erythroblasts) null; those cells are now "Erythroid" and the
  column has no nulls. Nothing else in the file changed. A cached
  `pbmc3k_pped.h5ad` is no longer used, the new file name triggers a fresh
  download. Migration: code that filtered or counted null `cell_type_lvl1`
  cells on this dataset now finds none.
- Remaining built-in exceptions in frame building, layout and grid helpers,
  marker and dataset helpers are now `errors.py` classes that subclass the same
  built-in (`InvalidArgumentError`, the new `InvalidArgumentTypeError`,
  `ConflictingLengthError`, `GridIndexError`, `MissingPlotDataError`,
  `IncompleteDownloadError`), so existing `except` clauses keep working.
- `layout(widths=...)` accepts numpy arrays (1D, 2D, or one array per row).
- `build_frame` accepts a single string for `metadata_columns` and
  `dimension_keys` as one key (it used to be split into characters).
- `KeyNotFoundError` and `VariableNotFoundError` now subclass `KeyError`, so
  `except KeyError` catches them. Their message prints without quotes.
- Unknown keys, metadata columns, embeddings, variables and modalities name the
  closest matches ("Did you mean 'leiden'?"). `build_frame` raises
  `KeyNotFoundError` (a `KeyError`) for unknown `metadata_columns` and
  `dimension_keys`.

### Fixed

- An observation or variable metadata column named like the id column
  (`Barcode`, `Variable`) raises `InvalidArgumentError` naming
  `observation_id_column` / `variable_id_column` instead of a polars
  duplicate-column error.
- Reading variables from an object without a main expression matrix raises
  `KeyNotFoundError` explaining it, instead of a `TypeError`.
- `build_frame` rejects `axis=True` / `axis=False` with `InvalidArgumentError`
  instead of reading them as 1 / 0.
- The embedding fallback warning lists the other embeddings in sorted order, so
  it reads the same on every run.
- Singular plots reject sets and arrays passed as `key` (pointing to the plural
  function) like they reject lists, instead of failing later.
- `ridge` no longer fills the ridges by `group_by` on its own (no hidden
  aesthetic promotion). Migration: pass `mapping=aes(fill="<group_by column>")`
  to keep the colored ridges; the same applies to `ridges`.
- `ridge` and `ridges` no longer fill the ridges by `group_by` on their own (no
  hidden aesthetic promotion); they default to grey. Migration: pass
  `fill=group_by` (e.g. `fill="leiden"`) to keep the colored ridges.
- `ridge` and `ridges` take `fill` / `color` (column shortcuts for
  `aes(fill=...)` / `aes(color=...)`) and `geom_fill` / `geom_color`
  (constants), like `violin`. A constant `color="#..."` used to pass through
  to the geom; it now names a column and raises `KeyNotFoundError` pointing to
  `geom_color`. Migration: use `geom_color="#..."`. Setting `fill` / `color`
  both as a shortcut and in `mapping` raises `InvalidArgumentError`.
- `histogram` takes one key; a list raises `TypeError` pointing to
  `histograms`, like the other singular plots. Migration: use `histograms`
  for several keys.
- `color=` / `fill=` no longer pass through to the geom as constants in
  `dimensional`, `umap`, `tsne`, `pca`, `expression`, their plurals,
  `scatter`, `bar`, `xyplot`, `xyplots`, `spatial` and `spatials`; they raise
  `InvalidArgumentError`. These functions take `geom_color` / `geom_fill`
  for constants instead. `blend` and `embedding_density` compute their
  colors, so they reject `color=` / `fill=` and take no constant. A
  `geom_color` / `geom_fill` that would hide the coloring by `key` or a
  `mapping` column raises `InvalidArgumentError`. Migration:
  `umap(color="red")` -> `umap(geom_color="red")`,
  `scatter(..., color="grey")` -> `scatter(..., geom_color="grey")`.
- `elbow` and `highest_expressed_genes` rename the constant `color` to
  `geom_color` (same default), following the convention that `color` / `fill`
  name columns. Passing `color=` raises `InvalidArgumentError` pointing to
  `geom_color`. Migration: `color=` -> `geom_color=`.
- `highest_expressed_genes` no longer accepts `observation_id_column`, which
  had no effect. Migration: drop the argument.
- `pca` and `pcas` label the axes `PC1` / `PC2` instead of `PCA1` / `PCA2`.
  Migration: update any `labs` or text matching the old titles.
- A `mapping` that sets an aesthetic the function sets itself raises
  `InvalidArgumentError` naming both, instead of a raw `TypeError`:
  `dimensional`, `umap`, `tsne`, `pca`, `expression` and their plurals
  (`x`, `y`, and `color` when `key` is given), `violin` / `boxplot` (`x`, `y`,
  also in `point_mapping`), `histogram` (`x`), `ridge` (`x`, `y`),
  `cluster_outlines` and `stream` (`x`, `y`, `group`). Migration: remove the
  aesthetic from `mapping` (pass the column as `key` / `group_by` instead).
- Error classes changed from builtins to `cellestial` errors:
  `xy` of the wrong length in `dimensional` (and wrappers), `blend` and
  `embedding_density` raises `InvalidArgumentError` (was `KeyError`); an unknown
  `point_geom` or `axis_type` raises `InvalidArgumentError` (was `ValueError`);
  `stream` raises `KeyNotFoundError` for missing velocity columns (was
  `KeyError`) and `InvalidArgumentError` when `velocity_key` cannot name them
  (was `ValueError`); `cluster_outlines` raises `KeyNotFoundError` for an
  unknown group and `EmptySelectionError` when no outline can be computed (both
  were `ValueError`). Migration: catch the new classes.

- `violin`, `boxplot` and `histogram` raise `InvalidArgumentError` when `key`
  is also the `group_by` / `color` / `fill` column or is not numeric (booleans
  still plot as 0 / 1), instead of a polars `DuplicateError` or a late failure.
- `histograms` with a string `keys` raises a `TypeError` instead of splitting
  it into characters.
- `dimensional` (and wrappers) without `key` accepts `mapping=aes(color=...)`.
- `dimensionals`, `umaps`, `tsnes` and `pcas` name the categorical key that
  lacks a `groups` / `drop` value and explain that the filter applies to every
  categorical key. Numeric keys are still left unfiltered.
- `expression`, `cluster_outlines` and `stream` suggest close matches for an
  unknown gene, group or velocity column.
- `build_frame` keeps missing values of a numeric categorical column as null
  instead of turning them into a `"nan"` category.
- `build_frame` raises `DuplicateKeysError` naming the repeat for repeated
  `variable_keys`, `metadata_columns` and `dimension_keys` (case variants of one
  embedding count as a repeat) instead of failing on a duplicate column, reads
  sparse embeddings, and raises `InvalidArgumentError` for an `axis` other than
  0 or 1.
- Stored-result plots on a multimodal object with a single modality now find
  the group column when `modality` is omitted.
- `datasets.from_url` reads a URL whose filename has no `.h5ad` / `.h5mu`
  suffix in the format given by `extension`, and caches it with that suffix.
- `heatmap` (per cell) and `annotated_heatmap` with `dendrogram=True` no longer
  crash when the grouping column holds non-string categories (e.g. integers).
- `heatmap`, `matrixplot`, `dotplot`, `stacked_violin` and `annotated_heatmap`
  raise `DendrogramMismatchError` when a stored dendrogram lacks groups present
  in the data, or (per-cell `heatmap`, `annotated_heatmap`) includes groups with
  no cells, instead of an internal error. `stacked_violin` used to drop those
  groups silently.
- Computing a dendrogram for a grouping with unused categories raises
  `DendrogramMismatchError` naming them, instead of an internal `KeyError`.
- Duplicate `keys` raise `DuplicateKeysError` in `heatmap`, `matrixplot`,
  `dotplot`, `stacked_violin` and `annotated_heatmap`, instead of an internal
  error.
- `annotated_heatmap` leaves out cells with no `group_by` value whether or not
  rows are binned. They used to show as an unlabelled block only when the data
  fit in `max_rows`.
- `histogram` shows the plotted value as the x-axis title.
- Per-cell `heatmap` keeps cells apart when two groups share a barcode (e.g.
  concatenated samples without unique names). Such cells used to be duplicated,
  and binned rows averaged cells of both groups.

- `volcano` and `volcanos` reject a `pvalue_threshold` outside (0, 1] before
  building the plot, instead of failing in the renderer.
- `volcano` names the missing fields when the stored ranking has no fold
  changes or p-values (e.g. logistic regression), instead of a bare `KeyError`.
- `volcanos(group_by=...)` computes the ranking once, as documented, even when
  `rank_genes_kwargs` holds arguments the stored parameters do not record
  (e.g. `pts=True`). It used to recompute it for every panel.
- `markers(groups=[])` raises `EmptySelectionError` instead of a polars error.
- `bracket(y_position=...)` below the top of the data no longer gives the plot
  a negative height, and a `y` column without finite values raises a clear
  `ValueError` instead of a `TypeError`.

- `volcano` labels: features tied on p-value and fold change are picked by
  name, so the labelled set no longer depends on sort order.
- `volcanos(groups=[])` raises `EmptySelectionError`; `volcanos` and `markers`
  reject `ncol` below 1 with `InvalidArgumentError` (it used to fail in the
  grid or with `ZeroDivisionError`).
- `markers` drops non-finite scores before taking the top `n_genes`, so a
  panel keeps `n_genes` genes with gapless ranks.
- Unknown `groups` errors from the group filters suggest a close match.
- `bracket` on a constant `y` column stacks the brackets above the data
  (the range is taken as 1) instead of drawing them all on the data line, and
  an `x` or `y` naming a column the plot lacks raises `MissingAestheticError`
  with a close match instead of a polars error.
- `matrixplot` declares `modality` in its signature (it used to reach `heatmap`
  only through `**geom_kwargs`).
- Binned per-cell `heatmap` and `tracksplot` keep bins in barcode order within
  a group (bin names are zero-padded; they used to sort as 0, 1, 10, 11, ..., 2).
- `stacked_violin` draws a flat line for a key whose values are all equal in a
  group (e.g. all zeros) instead of leaving the cell empty, and rejects an
  unknown `fill_statistic` or `scale` with `InvalidArgumentError` (an unknown
  `fill_statistic` used to drop the fill silently).
- `annotated_heatmap` settles ties for the most frequent label of a binned
  categorical track the same way on every run, and raises
  `InvalidArgumentError` for a category -> color mapping on a numeric track
  instead of drawing it with the wrong scale.
- `tracksplot` raises `DendrogramMismatchError` when the dendrogram lacks a
  group present in the data; that group's cells used to be dropped silently.
- The `DendrogramMismatchError` hint also names a same-named column with
  different groups (e.g. a modality's own column) as a possible cause.

- `cl.spatial` and `cl.spatials` on SpatialData point (or polygon-centroid)
  shapes now place each table row at its own shape, matched on the table's
  instance key. Spots used to be matched by position, so a table ordered
  differently from its shapes colored the wrong spots, and a table covering only
  some shapes failed with a shape error.
- `cl.stream()` on an embedding stored without the `X_` prefix (e.g.
  `wnn_umap`) reads `velocity_wnn_umap`; it used to reuse the positions as
  velocities and draw meaningless streams.
- `dimensional` and its wrappers reject `xy` components that do not exist
  (0-based values or beyond the embedding) with an `InvalidArgumentError`
  instead of returning a plot that fails to render.
- `xyplot` raises `KeyNotFoundError` with close matches when `x` or `y` is not
  in the data, instead of a `ValueError` about tooltip fields.
- `highest_expressed_genes(n=...)` and `elbow(n_pcs=...)` reject values below 1;
  negative values used to show all but the last genes or components.
- `violins`, `boxplots` and `ridges` reject a single string for `keys`, which
  used to be split into characters.
- `violin`, `boxplot` and `histogram` warn when `groups` or `drop` is passed
  without `group_by`, where the filter is ignored.

- `spatial(polygon=True)` draws holes in polygons as holes; a hole wound the
  same way as its exterior used to be filled. The outline still draws a line
  from the exterior to each hole until ggrust splits polygon rings.
- `spatial(polygon=True)` keeps the instance key column, so `add_keys` and
  `tooltips` can use it; it used to fail with a column-not-found error.
- `spatial(polygon=True)` joins only table rows annotating the chosen shapes
  element. A table annotating several shapes elements with overlapping instance
  ids used to draw rows of the other elements on these shapes.
- `spatial` and `spatials` raise `DuplicateKeysError` naming the repeated ids
  when a SpatialData shapes element repeats an instance id, instead of an
  internal pandas error.
- `spatials` resolves the image and geometry once for all keys instead of once
  per key.
- `spatial` and `spatials` document that with several Visium libraries every
  spot is drawn on the chosen library's image.

## [0.62.1] - 2026-11-09

Fixes for the fixed-cell plots: stable group order, area-proportional dotplot
dots and tighter key-group bracket labels. There are no breaking changes.

### Changed
- `dotplot` dot area is now proportional to the share of expressing cells, so
  a group where no cell expresses a key shows an empty cell instead of a small
  dot, as in scanpy. Low shares draw smaller dots than before; 100% dots are
  unchanged.
- `stacked_violin`, `dotplot` and aggregated `heatmap` no longer leave empty
  space above the key-group bracket labels.

### Fixed
- `heatmap(aggregate=True)`, `matrixplot` and `dotplot` now order groups the
  same way on every call, by their first appearance in the data. The order
  used to change between identical calls.

## [0.62.0] - 2026-11-09

Three new plots: `tracksplot`, `blend` and `embedding_density`. The release
also moves to ggrust 0.8.1 and keeps `dotplot` dots inside their cells.
There are no breaking changes.

### Added
- Added `embedding_density`, an embedding plot with cells colored by the local
  density of cells around them, optionally with one panel per group:
  `cl.embedding_density(data, group_by="condition")` (issue #74). Each panel's
  density uses only that group's cells, and every cell is drawn in grey behind
  each panel.
- Added `blend`, a co-expression plot of two features on an embedding, after
  Seurat's blended feature plot: `cl.blend(data, keys=("CD3E", "CD8A"))`. It
  returns a grid of the two feature panels, the blended panel, and a
  two-dimensional color key. Cells expressing both features show the sum of
  both colors (yellow for the default red and green) (issue #75).
- Added `InvalidColorError`, raised when a color is not in a format the
  function can compute with.
- Added `tracksplot`: cells ordered along x by group, one bar or area track per
  key, a colored group bar marking the group spans, and optional dendrogram and
  key-group brackets. Shares its parameters with `heatmap`:
  `cl.tracksplot(data, keys=["CD3D", "LYZ"], group_by="leiden")` (issue #73).
- Added `InvalidArgumentError` (a `ValueError`) for missing or out-of-range
  parameter values, and `DendrogramMismatchError` (a `ValueError`) for a stored
  dendrogram that covers groups absent from the data.

### Fixed
- `dotplot` dots no longer spill into neighboring cells when the canvas is
  smaller than the plot's own size (for example `ggsave(plot, w=8, h=5,
  unit="in")` or `+ ggsize(...)`). Dot sizes are now fractions of a cell (a
  dot for 100% of cells is 80% of a cell wide), so they shrink with the cells.
  The size legend is drawn for the default cell size and does not shrink with
  a resized canvas. Dot size now covers a fixed 0 to 100% range, so genes
  expressed in few cells get smaller dots than before.
- `dotplot`, `stacked_violin` and `matrixplot` cells are now exactly 30 pixels
  wide. Before, a dendrogram squeezed them to 21 to 25 pixels and plots with
  few groups stretched them to about 34 pixels. A single key keeps the backend's
  50 pixel minimum panel instead of clipping the legend.

### Dependencies
- ggrust is raised to 0.8.1 (`ggrust>=0.8.1,<1.0`). It brings working
  `sampling=` on layers, `label_format` and `check_overlap` on `geom_text` and
  `geom_label`, `stat_centroid`, contour `breaks=`, and `scale_linewidth` /
  `scale_stroke`. Layers you add to cellestial plots can use all of them.
- Behavior changes from ggrust that can show in plots you extend: a mapped
  `linewidth` now uses a 1..7 range (lines get slightly thicker), and
  `ggsave(plot, filename)` called without `path` writes to `filename`
  directly instead of into a `lets-plot-images/` folder. `cl.save` always
  passes `path`, so it behaves as before.

## [0.61.0] - 2026-11-08

cellestial now builds and renders every plot with ggrust, a Rust backend that
keeps the lets-plot API. Statistics, density estimates, velocity grids,
outlines, axis dendrograms and arrow axes that cellestial used to compute or
draw by hand now come from ggrust. That shrinks the dependency tree: lets-plot,
scikit-image and tqdm are gone, spatialdata is optional, and the `extras` extra
is removed. The release also adds `highlight`, gives the matrix-style plots
fixed-size cells, turns `annotated_heatmap` into a single zoomable plot, and
renames several parameters for consistency. Most of the breaking changes are
renames with a one-line migration, listed below.

### Added
- Added `highlight`, a layer that draws the chosen groups in color over every
  other cell in grey: `cl.umap(data, key="leiden") + cl.highlight(["0", "3"])`
  (issue #38).
- Added `track_titles` to `annotated_heatmap` (see the `annotated_heatmap`
  entry under Breaking).

### Dependencies
- lets-plot is replaced by ggrust (`ggrust>=0.8.0,<1.0`).
- scikit-image is no longer a dependency: `cluster_outlines` uses ggrust's
  density outline.
- tqdm is no longer a dependency: dataset downloads print their own progress
  line.
- spatialdata moved to the `spatial` extra (see Breaking).
- The `extras` extra (matplotlib, scanpy, scvelo) is removed (see Breaking).
  None of the three is needed any more: `stream` builds its velocity grid and
  streamlines with ggrust instead of scVelo and matplotlib.

### Breaking
- Plots are built and rendered with ggrust instead of lets-plot. ggrust keeps
  the lets-plot API, but cellestial now returns ggrust plot objects, which do
  not combine with layers or themes made by `lets_plot`.
  - Migration: replace `from lets_plot import ...` with `from ggrust import ...`
    in code that adds layers, scales or themes to cellestial plots.
- `cellestial.setup_html` is removed, along with the ggrust `LetsPlot` shim it
  aliased. It was a no-op, and so is the setup call cellestial made on import
  in IPython.
  - Migration: delete `setup_html()` calls; use
    `ggrust.display("html" | "svg" | "png")` to choose notebook output.
- `spatialdata` is no longer installed by default. It is only needed to plot
  `SpatialData` objects with `spatial`, `spatials` or `build_frame`; AnnData
  inputs, spatial ones included, work without it. Dropping it from the core
  install keeps its dask and geopandas stack out of environments that do not
  use it.
  - Migration: install the `spatial` extra, `pip install "cellestial[spatial]"`.
- The `extras` extra is removed. cellestial no longer imports matplotlib,
  scanpy or scvelo.
  - Migration: install the packages you use directly, e.g.
    `pip install cellestial "scanpy[leiden]"` instead of
    `pip install "cellestial[extras]"`.
- `annotated_heatmap` returns one `PlotSpec` instead of a `gggrid` of
  `SupPlotsSpec` panels. Annotation tracks are ggside side layers
  (`geom_xsidetile`/`geom_ysidetile`), each with its own fill scale
  (`new_scale_fill`) and legend, so they zoom and pan with the heatmap in HTML
  output (`interactive=True`). Track titles beside the bands are hidden by
  default, since the legends name each track; the new `track_titles=True`
  shows them (a top or bottom band's names on its right, a left or right
  band's names above it). The group dendrogram is drawn in the observation
  band, outermost. `layers_all` acts as `layers` now that there is a single
  plot. `column_annotation_height`, `row_annotation_width` and the tree's 0.15
  width set the bands' share of the panel (`ggside_panel_scale_x`/`_y`).
  - Migration: index the plot's `as_dict()["layers"]` instead of
    `["figures"]`; combine it with other plots through `gggrid` as any
    `PlotSpec`.
- `dotplot`, `stacked_violin`, `matrixplot` and `heatmap(aggregate=True)` draw
  the group dendrogram with ggrust's axis dendrogram (`scale_y_dendro` +
  `guide_axis_dendro`) instead of a `geom_path` layer. The tree sits in a fixed
  band right of the panel rather than inside an extended x range, so the data
  area no longer shrinks to make room for it. `dendrogram_kwargs` is removed
  from these functions and from `heatmap`; the per-cell
  `heatmap(aggregate=False)` keeps its path layer, styled by `dendrogram_color`
  and `dendrogram_size`.
  - Migration: drop `dendrogram_kwargs`; style the tree with `dendrogram_color`
    and `dendrogram_size`, or with `theme(axis_ticks_y=element_line(...))`.
- `arrow_axis` returns a ggrust theme with arrow axis lines instead of a
  deferred layer of `geom_path`/`geom_segment`. It no longer takes `plot=`
  (the theme needs no data) or extra keyword arguments, which used to be
  forwarded to `arrow()`. Arrowhead options go through an `arrow_kwargs` dict,
  and `linetype=` is an explicit parameter. `length` is a fraction of the panel
  rather than of the data range, so arrows start at the panel corner.
  - Migration: drop `plot=`; `arrow_axis(type="closed")` becomes
    `arrow_axis(arrow_kwargs={"type": "closed"})`; pass `linetype=` directly.
    Other geom options such as `alpha` are no longer supported.
- `bracket` computes its tests with ggrust's `stat_compare_means` instead of
  scipy. P-values, the `none`/`bonferroni`/`fdr_bh` corrections and the stars
  agree for groups of 50 or more values; below that, tie-free groups get R's
  exact Wilcoxon test, so `mannwhitney` p-values change there.
  `test="ttest"` is now Welch's t-test rather than the pooled-variance one,
  matching R's `t.test()` and scanpy, so its p-values change too. Stars now use
  `<=` cutoffs, and p-value labels show 2 significant digits. Bracket tips are
  3% of the y range (ggpubr's default) instead of a fixed size. Removed:
  `test="student"`, `alternative`, `label="padj"` and label sequences,
  `label_format`, `prefix`, `prefix_style`, `separator` and `threshold`.
  - Migration: `label="padj"` becomes `label="pvalue"` (it shows the adjusted
    p-value); `threshold=0.05` becomes `hide_ns=True` with `label="stars"`;
    pass `tip_length=` for other tips. One-sided and pooled-variance tests are
    no longer available; use `test="ttest"` for Welch's test.
- `volcano` and `volcanos` name their p-value column after the values it holds.
  Under the default `use_adjusted_pvalue=True` the column, tooltip entry and
  y-axis title read `pvalue_adj` and `-log10(Padj)`; raw p-values are
  unaffected.
  - Migration: pass `pvalue_column="pvalue"` to keep the old column name.
- `volcano` and `volcanos` spell out the negative log p-value parameter and the
  column it names: `neg_log_pvalue_column` is now `negative_log_pvalue_column`,
  and its default changes from `neg_log_pvalue` to `negative_log_pvalue`.
  - Migration: rename the keyword. Code reading the output column by name, or
    naming it in `mapping`/`tooltips`, needs the new spelling too, or pass
    `negative_log_pvalue_column="neg_log_pvalue"` to keep the old column name.
- `groups` and `drop` raise `KeyNotFoundError` when given a category the
  grouping column does not have, rather than filtering to nothing. A name that
  matched nothing was silently ignored, so a typo produced a blank panel, and a
  typo among valid names produced a plausible figure quietly missing that
  condition. The error names the unknown values and lists the available ones,
  matching what `marker_genes` already did.
  - Migration: correct the name, or drop it from the list. To tolerate names
    that may be absent across datasets, intersect with the column's categories
    before the call.
- `spatial` and `spatials` no longer accept `scale_axis`. A spatial plot maps
  one value column onto one continuous colour scale, so min-max scaling it was
  absorbed by that scale and left the rendered plot unchanged, while leaking
  normalized values into the tooltips.
  - Migration: remove the argument. Plots render identically without it. Only
    the colour bar labels change, from 0-1 back to the values' own units, and a
    numeric `midpoint` is now given in those units.
- `stacked_violin` renames `color_by` to `fill_statistic`. The parameter picks
  what drives the fill aesthetic, not color, and its old name shadowed the
  backend's own `color_by` layer parameter, which could not be passed through
  `**geom_kwargs`.
  - Migration: `color_by="group"` becomes `fill_statistic="group"`. A
    `color_by` argument now reaches the violin layer instead.
- `dotplot` and `stacked_violin` name the parameters for their output frame
  columns with a `_column` suffix, matching `value_column` and
  `variable_column`: `percentage_key` is now `percentage_column`, `mean_key` is
  now `mean_column`, and `aggregate_key` is now `aggregate_column`. Defaults
  are unchanged.
  - Migration: rename the keywords.
- On/off toggle parameters drop their `show_` prefix, matching the bare-noun
  toggles elsewhere (`dendrogram`, `rectangle`, `legend`, `line`).
  `violin`, `violins`, `boxplot` and `boxplots` take `points` instead of
  `show_points`, and `volcano` and `volcanos` take `threshold_lines` instead of
  `show_threshold_lines`.
  - Migration: rename the keyword. The old names are no longer recognized and
    are passed through to the geom like any other extra keyword, so they no
    longer hide the points or threshold lines.
- `observations_name` and `variables_name` are renamed to
  `observation_id_column` and `variable_id_column` across the public API
  (`build_frame`, the frame helpers and every plot that takes them). Defaults
  (`"Barcode"`, `"Variable"`) and behaviour are unchanged.
  - Migration: e.g. `observations_name="Cell"` becomes
    `observation_id_column="Cell"`.

### Changed
- `dotplot`, `stacked_violin` and `matrixplot` draw cells of a fixed size
  (square, and 1.5 times taller than wide for violins), and the plot grows
  with the number of keys and groups instead of stretching cells to fill a
  fixed canvas. Dot sizes no longer shrink as keys or groups are added. A
  `ggsize` added to the plot still overrides the computed size.
- `bracket` stacks brackets at a fixed pixel step sized to the label, and the
  plot grows taller by the height of the stack, so labels no longer collide
  with the next bracket and many brackets no longer squash the data. The
  default `y_step` no longer follows `y_padding`. A `ggsize` added after the
  brackets still overrides the computed size.
- The per-cell `heatmap` fills the canvas instead of being locked to a square
  panel, so heatmaps with many keys are no longer a narrow column with
  overlapping key labels. Cell rows sit one y unit apart rather than being
  rescaled to the width of the key axis.
- `cluster_outlines` pads each cluster's density grid by the reach of its
  kernel instead of a fixed 1.5 data units, so outlines stay closed on t-SNE,
  spatial and rescaled embeddings, not just UMAP. `padding` now defaults to
  `None` (automatic); an explicit float works as before.
- `stream`, `cluster_outlines` and `stacked_violin` compute their velocity
  grid, density outlines and violin densities with ggrust. Plots look the
  same: violins are pixel-identical, outlines move by at most 5 px, and
  streams differ only by float32 rounding.
- `stacked_violin` warns when it skips a group's violin because its values are
  too uniform to estimate a density, instead of dropping it silently. Other
  errors from the density estimate are no longer swallowed.
- `heatmap`, `matrixplot` and `stacked_violin` order integer-like group labels
  (such as leiden clusters "0" to "16") numerically, smallest on top, as
  `dotplot` already did. They used to follow first-seen order.
- `markers` caps `ncol` at the number of groups, so a single group fills the
  canvas instead of taking one third of it.
- Extra keyword arguments forwarded to the underlying geom are documented as
  `**geom_kwargs` everywhere (previously `**point_kwargs` or `**text_kwargs`
  in some functions). Calls are unaffected.

### Fixed
- Single-key `violin`, `boxplot`, `histogram` and their plural variants no
  longer raise `DuplicateError: duplicate column name` with polars 1.44 and
  newer.
- `volcano` and `volcanos` break gene-label ties by absolute log fold change, so
  the labels picked among p-values that underflow to zero are the strongest
  features and stable between builds.
- `stream` no longer fails on a two-vertex streamline, and drops streamlines
  whose midpoint vertex is repeated.

### Notes
- `bracket` tests the rows the plot draws, so on a plot built with the plot
  function's own `threshold`, the comparison covers only the observations
  above it. Documented only; behaviour unchanged.

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
