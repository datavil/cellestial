# v1 public API review (2026-11-18)

Scope: everything in `cellestial.__all__`, `cellestial.datasets`, and the subpackages
users import (`cellestial.layers`, `cellestial.frames`, `cellestial.util`,
`cellestial.util.errors`). Base: `claude/fixes-combined` (0.62.1). Yardstick:
`AGENTS.md`, `philosophy.md`, `MEMORY.md`, `CHANGELOG.md` and the closed decisions in
`plans/done/api_audit_v1.md` (items decided there are not reopened; see the end).

Method: dumped every exported signature, compared singular/plural pairs
mechanically, ran ~70 probe calls on `cl.datasets.pbmc3k(cache_directory="data")`
(with `rank_genes_groups` on `leiden_res_0.02` for the differential plots), and ran
all 115 `jupyter-execute` docstring examples (all pass). The current surface is now
pinned by `tests/test_public_api_surface.py` + `tests/snapshots/public_api_signatures.json`,
so every later rename shows up as a deliberate snapshot update.

"Breaking" below means a renamed, removed or reordered parameter, a changed default,
or a removed export. Turning a silent no-op or a raw crash into a cellestial error is
marked non-breaking (only misuse is affected).

## 1. Findings, ranked

Rank = cost of fixing after the freeze x how often users hit it.

### F1. Two parameters for one concept: `variable_keys` and `add_keys` (breaking)

- Evidence: `cellestial/single/core/dimensional.py:68-69`, `cellestial/spatial/spatial.py:90-91`
  and `:377-378`; same pair on `umap(s)`, `pca(s)`, `tsne(s)`, `expression(s)`,
  `dimensionals`, `spatials` (12 functions). `add_keys` already accepts metadata
  columns *and* variable names (docstring at `dimensional.py:124`), so it is a strict
  superset of `variable_keys` (`dimensional.py:122`, "Variable keys to add").
- Why v1: two ways to do one thing, both frozen in 12 signatures; users will copy both
  from tutorials. Violates "explicit, one obvious way".
- Recommendation: remove `variable_keys` from the plot functions, keep `add_keys`.
  Keep `variable_keys` on `build_frame`, `plot`, `scatter`, `bar` (no `add_keys`
  there; it is the frame builder's own vocabulary). **Breaking.**

### F2. `key` means two different things; stored-result keys have four names (breaking)

- Status: `ranking_key` part done (A2, 2026-11-22): `markers`, `volcano(s)`,
  `marker_genes(_dict)` take `ranking_key`, the heatmap family takes
  `markers: bool` plus `ranking_key`, and old-style calls raise. `embedding_key`
  (was `use_key`) is still open.

- Evidence: in every plot `key` is the column/feature to plot (`umap`, `violin`,
  `spatial`, ...). In `markers`, `volcano(s)`, `marker_genes(_dict)` `key='rank_genes_groups'`
  is the stored ranking (`markers.py:37`, `volcano.py:48`). The heatmap family carries
  the same thing as `markers: bool | str` (`heatmap.py:74`, `:649`, also `dotplot`,
  `stacked_violin`, `tracksplot`, `matrixplot`). Other stored-result keys are
  `pca_key` (`elbow.py:29`), `use_key` (dimensional family, `embedding_density`, `blend`),
  `dendrogram_key`, `spatial_key`, `image_key`, `velocity_key`.
- Why v1: `cl.markers(data, key=...)` vs `cl.umap(data, key=...)` is the most likely
  user confusion in the whole API, and `markers=True` / `markers="name"` (bool | str)
  is a flag that is secretly a key. Renaming any of these after v1 breaks notebooks.
- Recommendation: one `<thing>_key` pattern for every stored result:
  `ranking_key` (markers, volcano(s), marker_genes(_dict), and heatmap family with
  `markers: bool`), `embedding_key` (was `use_key`), keep `pca_key`, `dendrogram_key`,
  `spatial_key`, `image_key`, `velocity_key`. **Breaking.**

### F3. Error classes have no stable public path and no common base (mostly non-breaking)

- Status: done (C1, C2). `cellestial.errors` is public, every error subclasses
  `CellestialError`, `SpatialLibraryError` is removed.

- Evidence: classes live in `cellestial/util/errors.py`; nothing re-exports them
  (`cellestial/__init__.py`, `cellestial/util/__init__.py`), so users write
  `from cellestial.util.errors import KeyNotFoundError`. No `CellestialError` base, so
  "catch anything cellestial raised" is impossible. `SpatialLibraryError(Exception)`
  (`errors.py:82`) is the only class not under a builtin category and is never raised.
  `CellestialWarning` exists but only one call site uses it (`util/utilities.py:170`).
- Why v1: the import path users put in `except` clauses is frozen API.
- Recommendation: expose `cellestial.errors` (public module, re-exporting the classes)
  and add `class CellestialError(Exception)` as a mixin base of every class (adding a
  base is non-breaking). Remove `SpatialLibraryError` (**breaking** only for code that
  imports it; it is never raised).

### F4. ~20 public raises still use bare builtins; some messages name AnnData/scanpy or a stale parameter (non-breaking)

- Evidence (all reachable from public calls):
  - `ValueError`: `complex/annotated_heatmap.py:547`, `single/differential/volcano.py:351`,
    `single/differential/utilities.py:322`, `:371`, `single/heatmap/heatmap.py:359`,
    `single/heatmap/dotplot.py:303`, `single/heatmap/stacked_violin.py:339`,
    `single/heatmap/utilities.py:775`, `:817`, `:883`, `:898`, `:1027`.
  - `TypeError`: `single/differential/volcano.py:679`, `single/differential/markers.py:227`,
    `single/core/subdimensionals.py:286` (also `:628`, `:968`, `:1307`, `:1640`),
    `single/core/distributions.py:225` (also `:533`, `:820`), `single/quick/ridge.py:469`,
    `spatial/spatials.py:284`, `single/heatmap/utilities.py:402`, `:739`.
  - `NotImplementedError`: `spatial/utilities.py:372`.
  - Stale name: `heatmap(data, markers=3)` raises "`rank_genes_groups` must be True or
    a string naming the `adata.uns` key" (`heatmap/utilities.py:733-739`); the
    parameter is `markers`. Same stale name in `:883` and `:898`.
  - AnnData/scanpy language in messages (against the AGENTS.md rule): `markers` says
    "`adata.uns['rank_genes_groups']` not found. Run `scanpy.tl.rank_genes_groups` ...
    `key_added`"; `volcano` says "not found in `data.uns`"; `n_genes` errors say
    "stored in `adata.uns[...]`".
  - Plural type messages differ and one is ungrammatical: "keys must be an Sequence of
    strings" (`subdimensionals.py:286`).
- Why v1: messages are not API, but the *classes* are: switching to the
  `InvalidArgumentError` / `InvalidArgumentTypeError` subclasses keeps `except
  ValueError/TypeError` working, so do it now while nobody pins on exact classes.
- Recommendation: replace with the matching `errors.py` classes, name the real
  parameter (`markers`), describe the stored ranking without slot names, and make
  plurals say "pass a list, or use `umap` for one key". **Non-breaking.**

### F5. Keyword-only boundary is inconsistent in a few signatures (breaking)

- Status: `groups` part done (B1, 2026-11-22): `markers`, `marker_genes` and
  `marker_genes_dict` take `groups` keyword-only. The rest of B1 stays as it is
  (`elbow(n_pcs)`, `highest_expressed_genes(n)`, `retrieve(index)`,
  `cache_directory`, `borders`, `get_figures`, `cluster_outlines`, `highlight`).

- Evidence (positional-or-keyword after `data`): `elbow(data, n_pcs)`,
  `highest_expressed_genes(data, n)`, `markers(data, groups)`, `marker_genes(data, groups)`,
  `marker_genes_dict(data, groups)`, `retrieve(plot, index)` (while
  `get_mapping(plot, *, index)` is keyword-only, `util/operations.py:326` vs `:364`),
  `get_figures(grid, indices, **kwargs)`, `borders(panel_color, legend_color, plot_color)`,
  `datasets.pbmc3k(cache_directory)` and siblings, `cluster_outlines(groups, ...)`,
  `highlight(groups)`.
- Why v1: once positional calls exist in the wild they cannot be made keyword-only.
  Everything else in the API already follows "data + the plotted key(s) positional,
  rest keyword-only".
- Recommendation: keyword-only for `n_pcs`, `n`, `groups` (markers, marker_genes(_dict)),
  `retrieve(index)`, `borders` colours, `cache_directory`. Keep positional:
  `data`, `key(s)`, `group` (volcano), `x`/`y`, `mapping` (scatter/bar),
  heatmap-family `keys, group_by`, `ridge(key, group_by)`, `highlight(groups)`,
  `cluster_outlines(groups)` (the layer's one subject). **Breaking.**

### F6. Default output column names mix styles and abbreviations (breaking)

- Evidence: `observation_id_column='Barcode'`, `variable_id_column='Variable'`
  (Capitalized) vs `value_column='value'`, `variable_column='variable'` (lower);
  `dotplot(percentage_column='pct_exp', mean_column='avg_exp')` (`dotplot.py:84-85`,
  abbreviations the naming rule forbids); `elbow(component_column='Principal Component',
  variance_column='Variance Ratio')` (`elbow.py:35-36`, spaces);
  `stacked_violin(aggregate_column='expression')`; volcano `negative_log_pvalue`,
  `logfoldchange`.
- Why v1: these strings become legend titles, tooltip labels and the names users map
  in `+ aes(...)` / `retrieve(...)`. Changing a default after v1 silently breaks
  downstream code.
- Recommendation: fix only the clear rule violations: `pct_exp` -> `percent_expressed`,
  `avg_exp` -> `mean_expression`. Leave the rest (they are readable and already
  documented). **Breaking** (default values).

### F7. `threshold` and `scale` each mean different things (breaking)

- Evidence: `threshold` drops rows below a value in `violin`/`boxplot`/`histogram`/`ridge`
  (+ plurals), `stacked_violin`, `highest_expressed_genes`; in `dotplot`
  (`dotplot.py:75`, default `0`) it is the "counted as expressed" cutoff.
  `scale` is a float overlap in `ridge(s)` (`ridge.py:67`), a method literal in
  `stacked_violin` (`stacked_violin.py:71`) and a raster factor in `save`.
  `ridge(data, "CD14", "leiden", scale="width")` is accepted silently (probe).
- Why v1: same name, different meaning and type, so muscle memory produces wrong plots
  without an error.
- Recommendation: `dotplot(expression_threshold=0)`, `ridge(s)(overlap=2.0)`; validate
  numeric `overlap`. **Breaking** (two renames).

### F8. `markers()` styling parameters (breaking)

- Evidence (`markers.py:37-49`, docs `:99-114`): `text_color`, `text_size`, but
  `fontface`, `angle` without the `text_` prefix; `rank_color: bool` reads as a colour
  value and overrides `text_color` when True; `line: bool` toggles a score curve
  styled by `line_color` / `line_size` / `line_alpha` / `line_kwargs`.
- Why v1: the only boolean in the API named like a colour; prefix drift inside one
  function.
- Recommendation: `text_fontface`, `text_angle`, `rank_gradient: bool`,
  `score_line: bool` (`line_*` styling unchanged). **Breaking.**

### F9. `save` freezes a copy of `ggsave` with abbreviations and changed defaults (breaking)

- Evidence: `util/save.py:14`: `w`, `h` (abbreviations), `GGBunch` in the type
  (a lets-plot legacy class), defaults differ from `ggrust.ggsave` (`scale=2.0`,
  `unit='in'`, `dpi=300` passed even for `unit='px'`; ggsave defaults are `None`).
  It adds no logic. The dpi docstring claimed a unit-dependent default; fixed in this
  branch to describe what the code does.
- Why v1: every future `ggsave` change must be mirrored by hand, and `w`/`h` get frozen.
- Recommendation: remove `save` and point users to `ggsave` (one fewer frozen
  function), or keep it as `save(plot, filename, **ggsave_kwargs)` with only the
  `scale=2.0` default. **Breaking** either way.

### F10. `blend` is a singular name that returns `SupPlotsSpec` (decision)

- Evidence: `single/core/blend.py:55`, Returns "Grid of four panels" (probe:
  `SupPlotsSpec`). It always returns one type, so predictability holds, but it breaks
  the singular -> `PlotSpec` naming rule (`markers` is a plural noun, so fine).
- Recommendation: keep and document it as the one composite exception (same footing as
  `annotated_heatmap`), or move it to `cellestial.complex`. Non-breaking if kept.

### F11. Deferred layers crash with raw errors on the wrong receiver (non-breaking)

- Evidence: `layers/_deferred.py` `DeferredLayer.__radd__` calls the builder without a
  type check. `cl.umaps(...) + cl.bracket(...)` -> `TypeError: 'NoneType' object is not
  a mapping`; `ggplot() + cl.ondata_legend()` -> `IndexError: list index out of range`
  (probe). `highlight` does check (`layers/highlight.py:86`).
- Also: `DeferredLayer` is in `cellestial.layers.__all__` (a public class users never
  construct).
- Recommendation: in `__radd__`, raise `InvalidPlotSpecError` for a `SupPlotsSpec`
  ("add the layer through `layers=`") and `MissingPlotDataError` for a plot with no
  data. Drop `DeferredLayer` from `__all__` (**breaking** only for importers).

### F12. Singular and plural input validation gaps (non-breaking)

- `xyplot(data, ["a", "b"], "c")` -> raw `TypeError: cannot use 'list' as a dict key`
  (`common/xyplot.py:52` never calls `_reject_sequence_key`).
- `scatter(data, None)` / `bar(data, None)` -> `AttributeError: 'NoneType' object has
  no attribute 'as_dict'`.
- `ridge(data, key, None)` -> raw polars `TypeError: invalid input for col`.
- Recommendation: `_reject_sequence_key` in `xyplot`, `MissingAestheticError` for a
  `None` mapping, `InvalidArgumentError` for `group_by=None` in `ridge`.

### F13. Silent no-ops users will rely on (non-breaking now, breaking later)

- `umap(data, groups=["0"])` with no `key` ignores `groups` silently
  (`dimensional.py:350`); a continuous `key` only warns.
- `umap(data, "CD14", midpoint="foo")` is accepted when `color_mid` is None
  (`util/utilities.py:475` is only reached with `color_mid`).
- `heatmap(data, genes, group, axis=1)` errors with "`metadata_columns` not found in
  variables", naming an internal parameter; with `axis=1` the keys must be variable
  annotation columns, which the message should say.
- Recommendation: raise for `groups`/`drop` without `key`, validate `midpoint` always,
  reword the `axis=1` message. Do before v1: after it, raising on accepted input is a
  break.

### F14. Spatial image parameters copy matplotlib / lets-plot vocabulary (breaking)

- Evidence: `spatial.py:70-74`: `cmap`, `norm: bool | None`, `vmin`, `vmax`
  (abbreviations, `geom_imshow` passthrough names), `greyscale` (British; the rest of
  the API is `color`). `crop=(left, right, top, bottom)` is an unusual order.
- Why v1 / ggrust: these four names tie the signature to `geom_imshow`; if ggrust's
  image layer differs, the frozen names stay.
- Recommendation: `grayscale`, `image_colormap`, `image_normalize: bool`,
  `image_min`, `image_max`; keep `crop` but document the order prominently.
  **Breaking.** (Acceptable alternative: keep them as deliberate lets-plot names, but
  decide it explicitly.)

### F15. `stream` has three ways to style arrows (breaking)

- Evidence: `layers/stream.py:32`: `arrows: FeatureSpec | None` (the lets-plot
  `arrow()` spec), `arrow_kwargs`, and `arrow_color` / `arrow_size` / `arrow_alpha`;
  plus `density` and `grid_density` side by side.
- Recommendation: drop `arrows` (keep the explicit `arrow_*` and `arrow_kwargs`);
  document `density` vs `grid_density` or rename to `line_density`. **Breaking.**

### F16. `cellestial.frames` exports AnnData-named builders (breaking)

- Evidence: `cellestial/frames/__init__.py` `__all__` has `anndata_observations_frame`,
  `anndata_variables_frame`, `anndata_variable_columns`; tests import them directly.
- Why v1: backend-specific names in a public namespace contradict the
  backend-agnostic rule and freeze internals of `build_frame`.
- Recommendation: make them private (`_` prefix or drop from `__all__`); `build_frame`
  stays the public entry point. **Breaking** for importers.

### F17. Styling parameters that duplicate layers (decision, breaking if changed)

- Evidence: the 6 dimensional functions (and plurals) carry `axis_type` + 4 `arrow_*`
  (duplicating `cl.arrow_axis()`), and `legend_ondata` + 8 `ondata_*` / `halo_*`
  (duplicating `cl.ondata_legend()`): 13 params x 12 signatures.
- Why v1: largest single block of frozen surface; philosophy point 1 ("modularity over
  abstraction") argues for `+ cl.arrow_axis(...)`.
- Recommendation: keep the two toggles (`axis_type`, `legend_ondata`) and drop the
  styling copies in favour of the layers. Reasonable alternative: keep all (AGENTS.md
  allows exceptions for very common layers).

### F18. Grid parameters copy `gggrid` (decision)

- Evidence: 9 grid params (`ncol`, `sharex`, `sharey`, `widths`, `heights`, `hspace`,
  `vspace`, `fit`, `align`, `guides`) on 13 plurals. ggrust's `gggrid` already has more
  (`design`, `tag_levels`, `axes`, `axis_titles`). `get_figures(**kwargs)` forwards to
  `gggrid` unlisted.
- Recommendation: keep the explicit list (matches lets-plot, so porting stays an
  import swap); state in the docs that new `gggrid` options will be added, never
  renamed. Non-breaking.

### F19. Dataset naming and loader flags (breaking)

- Evidence: `datasets.pbmc3k` is 17,041 bone-marrow cells, not the 10x pbmc3k
  (`plans/pbmc3k_labels.md`); its `cell_type_lvl1` labels are wrong. `bring: bool` on
  every loader (`datasets.py:66`) does not say what it brings.
- Recommendation: fix the hosted labels before v1 (data change, not API), and decide
  whether the loader name stays (it is in every tutorial). Rename `bring` to
  `copy_from_global_cache`. **Breaking** for the rename.

### F20. Docstring gaps (non-breaking)

- No `Raises` section: `umap`, `pca`, `tsne`, all 13 plurals, `scatter`, `matrixplot`,
  `save`, `get_mapping`, `arrow_axis`. `borders` has no NumPy docstring and no type
  hints (`layers/utilities.py:21`). Dataset loaders have no `Examples`.
- 71 links to lets-plot.org in 18 files; they will point at the wrong backend once
  ggrust is the only backend.
- All 115 docstring examples run on current data.

### F21. Smaller naming items (low, breaking if changed)

- Top-level colour constants (`BLUE`, `CHERRY`, ..., `show_colors`): 10 names frozen in
  the root namespace. Option: move to `cellestial.colors`.
- `retrieve(plot)` returns the plot's data frame; the verb does not say what it gets.
  Option: `get_frame` (matches `get_figure`, `get_mapping`).
- `include_dimensions: bool | int` doubles as a count. Option: leave.

## 2. Decisions for the user

Recommended choice first. Approve batch by batch.

**Batch A: consistency renames (breaking, cheap now)**
- A1. Drop `variable_keys` from plot functions, keep `add_keys` (F1). / keep both.
- A2. (`ranking_key` done 2026-11-22; `embedding_key` open) Stored-result keys as `<thing>_key`: `ranking_key`, `embedding_key`; heatmap `markers: bool` (F2). / keep `key`, `use_key`, `markers: bool | str`.
- A3. `dotplot(expression_threshold=)`, `ridge(s)(overlap=)` (F7). / keep `threshold`, `scale`.
- A4. `markers()`: `text_fontface`, `text_angle`, `rank_gradient`, `score_line` (F8). / keep.
- A5. `pct_exp` -> `percent_expressed`, `avg_exp` -> `mean_expression` (F6). / keep.
- A6. Spatial image: `grayscale`, `image_colormap`, `image_normalize`, `image_min`, `image_max` (F14). / keep lets-plot names.
- A7. `stream`: drop `arrows` (F15). / keep.

**Batch B: signature shape (breaking)**
- B1. (`groups` part done 2026-11-22; the rest kept positional) Make `n_pcs`, `n`, `groups` (markers, marker_genes(_dict)), `retrieve(index)`, `borders` colours, `cache_directory` keyword-only (F5). / leave positional.
- B2. Drop the `arrow_*` and `ondata_*` styling copies, keep the toggles (F17). / keep all.
- B3. `save`: remove in favour of `ggsave` (F9). / thin `save(plot, filename, **ggsave_kwargs)` / keep.

**Batch C: exports and errors**
- C1. Public `cellestial.errors` module + `CellestialError` base (F3). / leave in `cellestial.util.errors`. **Done.**
- C2. Remove unused `SpatialLibraryError` (F3). / keep. **Done.**
- C3. Make `anndata_*` frame builders and `DeferredLayer` private (F11, F16). / keep public.
- C4. Colour constants: keep at top level. / move to `cellestial.colors` (F21).
- C5. `retrieve`: keep. / rename to `get_frame` (F21).

**Batch D: non-breaking hardening (approve as one)**
- D1. Replace the ~20 bare raises, fix stale `rank_genes_groups` and AnnData/scanpy wording (F4).
- D2. Deferred layers raise `InvalidPlotSpecError` on a grid or data-less plot (F11).
- D3. `xyplot` sequence rejection, `None` mapping, `ridge(group_by=None)` errors (F12).
- D4. Raise for `groups`/`drop` without `key`; always validate `midpoint`; reword heatmap `axis=1` error (F13).
- D5. Add missing `Raises` sections, a NumPy docstring for `borders`, swap lets-plot links for ggrust docs (F20).

**Batch E: identity**
- E1. `blend`: keep, documented as a composite exception (F10). / move to `cellestial.complex`.
- E2. Fix the hosted pbmc3k labels before v1; keep the loader name (F19). / rename the loader too.
- E3. `bring` -> `copy_from_global_cache` (F19). / keep.
- E4. Grid params: keep the explicit `gggrid` list, add new options additively (F18). / add `grid_kwargs`.

## 3. Checked and fine

- Return types: every singular in the 12 pairs is annotated and returns `PlotSpec`,
  every plural `SupPlotsSpec` (probe and pinned test); `markers`, `layout`,
  `get_figures` return `SupPlotsSpec`; `annotated_heatmap` returns `PlotSpec`.
- Singular/plural parity: all 12 pairs share the same non-grid parameters in the same
  order with the same defaults; plurals only lack `frame` (additive later).
- Singulars reject sequences with `InvalidArgumentTypeError` naming the plural:
  `dimensional`, `umap`, `pca`, `tsne`, `expression`, `histogram`, `ridge`, `spatial`
  (`violin`/`boxplot` accept sequences by design). Plurals reject a bare string.
- Keyword-only: everything after the data and plotted key(s) is keyword-only in all
  plot functions except the F5 list.
- No `width`/`height` sizing parameters on plot functions; `layout(widths, heights)`
  are relative ratios.
- `color`/`fill` are column shortcuts and `geom_color`/`geom_fill` constants in every
  key-coloured plot; `umap(color="red")` raises with guidance; heatmap family,
  `volcano(s)`, `markers` pass through (user's choice).
- Colour word order follows one rule: `<element>_color` for a drawn element
  (`arrow_color`, `point_color`, `line_color`) and `color_<level>` for scale stops or
  classes (`color_low/mid/high`, `color_up/down/nonsignificant`). Worth stating in
  AGENTS.md, no change.
- Boolean toggles use the ggplot noun style (`legend`, `dendrogram`, `points`,
  `rectangle`, `image`) consistently; only `rank_color` and `line` (F8) read badly.
- `tooltips` accepts `'none'`, a list or `layer_tooltips()` and rejects other strings
  with `InvalidArgumentError`; `layer_tooltips` exists in ggrust.
- `interactive`, `modality`, `group_by`, `mapping`, `observation_id_column`,
  `variable_id_column`, `value_column`, `variable_column`, `<element>_kwargs` are
  named and defaulted consistently wherever they appear.
- `modality` is present on every function that reads stored results or a matrix per
  modality; per-cell plots use qualified `modality:name` keys instead.
- Error classes subclass the matching builtin (`KeyNotFoundError` is a `KeyError`,
  `InvalidArgumentError` a `ValueError`, ...); names are specific and worth keeping.
- `get_figure` index errors raise `GridIndexError`; `retrieve` and `get_mapping` work
  on built plots.
- Deferred layers (`bracket`, `stream`, `cluster_outlines`, `ondata_legend`,
  `highlight`) compose with `+` on singular plots; the `plot=` escape hatch is
  documented.
- Subpackage `__all__` lists contain no private names (existing test).
- No lets-plot imports in the package; the only backend ties are the passthrough
  names in F9, F14, F15, F18 and the doc links in F20.
- All 115 `jupyter-execute` docstring examples run on current data.
- Not reopened (closed in `plans/done/api_audit_v1.md`): `borders` debug defaults,
  `n` / `n_genes` / `top_n`, the `share_*` families and `markers(share_labels=True)`,
  `groups` accepted types, `midpoint` family defaults, positional `ridge(group_by)`,
  `volcano(group)` / `volcanos(groups)`, heatmap `axis` + `scale_axis`, singular
  distributions accepting sequences.
