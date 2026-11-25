# API consistency findings before v1

Collected on 2026-11-18 from the pre-v1 API review (`plans/v1_api_review.md` on branch
`claude/v1-api-review-2f395f`) and the colour-parameter work of the same day. Nothing
here is implemented yet. Each item gives the problem, the reasoning, and the
recommendation. "Breaking" means existing user code would need a change.

Rule of thumb behind every item: after v1, any rename or signature change breaks
notebooks, so anything that is inconsistent should be settled now or kept on purpose.

---

## 1. Naming

### 1.1 `variable_keys` and `add_keys` do the same job (breaking)

- **Problem:** 12 plot functions (`umap(s)`, `pca(s)`, `tsne(s)`, `expression(s)`,
  `dimensional(s)`, `spatial(s)`) take both. `add_keys` already accepts metadata
  columns and variable names, so it covers everything `variable_keys` does.
- **Reasoning:** two ways to do one thing, both frozen in 12 signatures. Tutorials will
  mix them. This goes against "one obvious way".
- **Recommendation:** remove `variable_keys` from those plot functions, keep `add_keys`.
  Keep `variable_keys` on `build_frame`, `scatter` and `bar`, where it is the frame
  builder's own vocabulary and there is no `add_keys`.

### 1.2 `key` means two different things (breaking)

- **Problem:** in every plot, `key` is the column or gene to plot. In `markers`,
  `volcano(s)` and `marker_genes(_dict)`, `key` is the name of the stored marker
  ranking. The heatmap family carries the same stored ranking as `markers: bool | str`,
  a flag that is secretly also a key. Other stored-result keys are named `pca_key`,
  `use_key`, `dendrogram_key`, `spatial_key`, `image_key` and `velocity_key`.
- **Reasoning:** `cl.markers(data, key=...)` next to `cl.umap(data, key=...)` is the
  most likely confusion in the whole API.
- **Recommendation:** one `<thing>_key` pattern for every stored result:
  `ranking_key` (markers, volcano(s), marker_genes(_dict), heatmap family, with
  `markers` becoming a plain bool), `embedding_key` instead of `use_key`. Keep the
  others as they are.

### 1.3 `threshold` and `scale` each mean several things (breaking)

- **Problem:** `threshold` filters values in most plots but is the "counted as
  expressed" cutoff in `dotplot`. `scale` is a float overlap in `ridge(s)`, a method
  name in `stacked_violin`, and a raster factor in `save`. `ridge(..., scale="width")`
  is accepted silently.
- **Reasoning:** same name, different meaning and type, so habit produces wrong plots
  without an error.
- **Recommendation:** `dotplot(expression_threshold=)` and `ridge(s)(overlap=)`, with
  `overlap` validated as a number.

### 1.4 `markers()` styling names drift (breaking)

- **Problem:** `fontface` and `angle` lack the `text_` prefix the other text options
  have. `rank_color` is a boolean named like a colour, and overrides `text_color` when
  True. `line` is a boolean toggling a score curve.
- **Reasoning:** the only boolean in the API named like a colour value, and prefix
  drift inside one function.
- **Recommendation:** `text_fontface`, `text_angle`, `rank_gradient: bool`,
  `score_line: bool`. `line_*` styling stays.

### 1.5 Default output column names use abbreviations (breaking)

- **Problem:** `dotplot` names its computed columns `pct_exp` and `avg_exp`.
- **Reasoning:** these strings become legend titles, tooltip labels and names users map
  in `aes(...)`. The naming rule forbids abbreviations, and changing a default after
  v1 silently breaks downstream code.
- **Recommendation:** `percent_expressed` and `mean_expression`. Leave the other
  output names (`Barcode`, `value`, `Principal Component`, ...) as they are; they are
  readable and documented.

### 1.6 Spatial image parameters use matplotlib vocabulary (breaking)

- **Problem:** `spatial` takes `cmap`, `norm`, `vmin`, `vmax` (abbreviations copied
  from the image layer) and `greyscale` (British spelling, while the rest of the API
  says `color`). `crop=(left, right, top, bottom)` is an unusual order.
- **Reasoning:** the abbreviations break the naming rule and tie the signature to one
  backend layer.
- **Recommendation:** `grayscale`, `image_colormap`, `image_normalize: bool`,
  `image_min`, `image_max`; keep `crop` but document its order prominently.
  Acceptable alternative: keep the current names as a deliberate choice.

### 1.7 `stream` has three ways to style arrows (breaking)

- **Problem:** `arrows` (an arrow spec), `arrow_kwargs`, and `arrow_color` /
  `arrow_size` / `arrow_alpha`. `density` and `grid_density` sit side by side without
  a clear difference.
- **Reasoning:** three overlapping routes to one setting.
- **Recommendation:** drop `arrows`, keep the explicit `arrow_*` and `arrow_kwargs`;
  document `density` vs `grid_density` (or rename to `line_density`).

### 1.8 Datasets: `bring` says nothing (breaking)

- **Problem:** every dataset loader has `bring: bool`.
- **Reasoning:** the name does not say what is brought from where.
- **Recommendation:** rename to `copy_from_global_cache`.

### 1.9 Colour word order (no change)

- **Observation:** names follow `<element>_color` for a drawn element (`arrow_color`,
  `point_color`, `line_color`) and `color_<level>` for scale stops or classes
  (`color_low/mid/high`, `color_up/down/nonsignificant`).
- **Reasoning:** both forms are consistent within their meaning.
- **Recommendation:** keep, and state the rule in `AGENTS.md`.

---

## 2. Signatures

### 2.1 Keyword-only boundary is inconsistent (breaking)

- **Problem:** almost every function makes everything after the data and the plotted
  key(s) keyword-only, except `elbow(n_pcs)`, `highest_expressed_genes(n)`,
  `markers(groups)`, `marker_genes(_dict)(groups)`, `retrieve(plot, index)` (while
  `get_mapping(plot, *, index)` is keyword-only), `borders(...)` colours and the
  dataset loaders' `cache_directory`.
- **Reasoning:** once positional calls exist in the wild, they can never be made
  keyword-only.
- **Recommendation:** make those keyword-only. Keep positional: `data`, `key(s)`,
  `group` (volcano), `x`/`y`, `mapping` (scatter/bar), heatmap `keys, group_by`,
  `ridge(key, group_by)`, `highlight(groups)`, `cluster_outlines(groups)`.

### 2.2 `save` copies `ggsave` with abbreviations (breaking)

- **Problem:** `cl.save` has `w`/`h`, mentions a legacy lets-plot type, and changes
  `ggsave` defaults (`scale=2.0`, `unit="in"`, `dpi=300` even for pixels) without
  adding any logic.
- **Reasoning:** every future `ggsave` change must be mirrored by hand, and `w`/`h`
  get frozen.
- **Recommendation:** keep it, slimmed to `save(plot, filename, **ggsave_kwargs)` with
  only the `scale=2.0` default. Alternative: remove it and point users to `ggsave`.

### 2.3 Styling copies of layers on the dimensional plots (decision)

- **Problem:** the 12 dimensional functions carry `axis_type` plus 4 `arrow_*`
  parameters copying `cl.arrow_axis()`, and `legend_ondata` plus 8 `ondata_*` /
  `halo_*` parameters copying `cl.ondata_legend()`: about 150 frozen parameters.
- **Reasoning for dropping:** the largest block of frozen surface; "modularity over
  abstraction" argues for `+ cl.arrow_axis(...)`.
- **Reasoning for keeping:** these are the most common plots, and the user prefers
  convenience in common plots (see `highest_expressed_genes` keeping its fill).
- **Recommendation:** keep them all.

### 2.4 Grid parameters copy `gggrid` (no change)

- **Problem:** 13 plurals list `gggrid` options explicitly; ggrust's `gggrid` already
  has more.
- **Reasoning:** the explicit list matches lets-plot, so porting stays an import swap.
- **Recommendation:** keep, and document that new `gggrid` options will be added,
  never renamed.

### 2.5 `blend` is singular but returns a grid (no change)

- **Problem:** `blend` returns `SupPlotsSpec`, breaking the "singular returns
  `PlotSpec`" naming rule.
- **Reasoning:** it always returns one type, so it stays predictable.
- **Recommendation:** keep, documented as the one composite exception (like
  `annotated_heatmap`).

---

## 3. Colour parameters (done on `claude/fixes-combined`)

- `color` / `fill` are column shortcuts and `geom_color` / `geom_fill` are constants in
  every key-coloured plot: `dimensional`, `umap`, `tsne`, `pca`, `expression`, their
  plurals, `scatter`, `bar`, `xyplot(s)`, `spatial(s)`.
  **Reasoning:** `umap(key="leiden", color="red")` silently painted every cell red
  and dropped the key colouring.
- `ridge(s)` gained `fill`, `color`, `geom_fill`, `geom_color` like `violin`; no
  automatic fill by group. **Reasoning:** the old automatic fill was a hidden
  aesthetic promotion, and `fill=group_by` is a one-word shortcut.
- `elbow` and `highest_expressed_genes` renamed the constant `color` to `geom_color`.
  `highest_expressed_genes` keeps its per-gene fill. **Reasoning:** convenience plots
  should look finished with no extra arguments.
- Left as passthrough by choice: heatmap family, `volcano(s)`, `markers`.
  **Reasoning:** their main colours encode values or significance, so a passed
  constant only affects outlines.
- Layers (`stream`, `bracket`, `arrow_axis`, `ondata_legend`, `cluster_outlines`) keep
  a constant `color`. **Reasoning:** layers never map data, so there is no clash.

---

## 4. Exports and errors

### 4.1 No public path or common base for error classes (mostly non-breaking)

- **Problem:** users must write `from cellestial.util.errors import ...`, and there is
  no `CellestialError` base to catch everything the library raises.
- **Reasoning:** the import path users put in `except` clauses is frozen API.
- **Recommendation:** a public `cellestial.errors` module re-exporting the classes,
  and a `CellestialError` base class on all of them (adding a base breaks nothing).

### 4.2 Unused `SpatialLibraryError` (breaking for importers only)

- **Problem:** never raised, and the only class not under a builtin category.
- **Recommendation:** remove it.

### 4.3 About 20 public raises still use bare builtins (non-breaking)

- **Problem:** some `ValueError`/`TypeError`/`NotImplementedError` remain, and some
  messages name AnnData/scanpy or a stale parameter (`heatmap(markers=3)` mentions
  `rank_genes_groups`). Plural type messages differ and one is ungrammatical.
- **Reasoning:** messages are not API, but classes are. Switching to the
  `errors.py` subclasses keeps `except ValueError` working, so do it before anyone pins
  exact classes.
- **Recommendation:** replace them, name the real parameter, drop slot names.

### 4.4 Backend-named frame builders are public (breaking for importers)

- **Problem:** `cellestial.frames` exports `anndata_observations_frame`,
  `anndata_variables_frame` and `anndata_variable_columns`.
- **Reasoning:** AnnData-specific names in a public namespace contradict the
  backend-agnostic rule and freeze internals.
- **Recommendation:** make them private; `build_frame` stays the public entry point.

### 4.5 `DeferredLayer` is public (breaking for importers)

- **Recommendation:** drop it from `__all__`; it is an implementation detail of the
  `+` layers.

### 4.6 Colour constants and `retrieve` (no change)

- **Recommendation:** keep the colour constants at the top level and keep the name
  `retrieve`.

---

## 5. Robustness (non-breaking)

- **Deferred layers on the wrong plot:** `cl.umaps(...) + cl.bracket(...)` gives
  `TypeError: 'NoneType' object is not a mapping`, `ggplot() + cl.ondata_legend()` an
  `IndexError`. **Recommendation:** raise `InvalidPlotSpecError` naming the fix.
- **Validation gaps:** `xyplot` accepts a sequence key; a `None` mapping crashes;
  `ridge(group_by=None)` gives an unclear error. **Recommendation:** clear errors.
- **Silent no-ops:** `groups` / `drop` without `key` do nothing; `midpoint` is only
  validated with `color_mid`; the heatmap `axis=1` error names an internal parameter.
  **Reasoning:** raising on accepted input after v1 is itself a break, so do it now.
  **Recommendation:** raise, always validate, reword.
- **Docstrings:** missing `Raises` sections, `borders` lacks a NumPy docstring, and
  some links point to lets-plot instead of ggrust docs. **Recommendation:** fill them.

---

## 6. Data and other open items

- **pbmc3k labels:** "B Cells" and "Erythroid" are swapped and the unlabelled cluster
  is erythroblasts; the data is bone marrow (`plans/pbmc3k_labels.md`).
  **Recommendation:** fix the hosted file before v1 (only the user can re-upload);
  keep the loader name, which is in every tutorial.
- **Duplicate helpers** written by two parallel sessions: `_reject_mapping_conflicts`
  (spatial) and `_reject_mapping_clash` (core); `_did_you_mean` and
  `_close_matches_sentence` (errors). **Recommendation:** keep one of each.
- **`histogram` with a list** raises a plain `TypeError`. **Recommendation:** use
  `InvalidArgumentTypeError`, like the other singulars.

---

## 7. Checked and fine

- Every singular returns `PlotSpec`, every plural `SupPlotsSpec`; all 12 pairs share
  parameters in the same order with the same defaults.
- Singulars reject sequences with an error naming the plural; plurals reject a bare
  string.
- No `width` / `height` sizing parameters on plot functions.
- `tooltips`, `interactive`, `modality`, `group_by`, `mapping` and the id/value column
  parameters are named and defaulted consistently.
- Error classes subclass the matching builtin.
- Deferred layers compose with `+` on singular plots.
- No lets-plot imports in the package.
- All 115 docstring examples run.
