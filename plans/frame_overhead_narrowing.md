# Frame-build overhead: narrowing + sparse-slice bug

## Summary of changes

Cold-start benchmark on `breast_cancer_atlas` (378k cells, 22 obs columns, two embeddings):

| format | original | pass 1 (dimensional) | pass 2 (full rollout) | scanpy |
| --- | --- | --- | --- | --- |
| html | 10.38s | 2.62s | **1.64s** | n/a |
| svg | 11.82s | 4.18s | **3.47s** | 10.77s |
| pdf | 23.60s | 16.15s | 15.35s | 12.93s |
| png | 23.53s | 15.99s | 15.33s | 7.79s |

Cellestial is now **3.1× faster than scanpy** for SVG and faster for HTML;
PDF/PNG remain rasteriser-bound (the lets-plot export stage dominates, not
cellestial code).

## Pass 3: narrowing-correctness fixes + `add_columns`

An audit found four regressions where the narrowing dropped columns a downstream
consumer needed. Root cause in every case: the narrowing must enumerate *all*
columns the frame's consumers read, and some were missed. Fixes:

- **`_resolve_tooltips` is now axis-aware.** It routes tooltip fields through
  `_collect_aes_columns` (obs metadata -> `metadata_columns` on axis 0, var
  metadata on axis 1, gene names -> `variable_keys` on axis 0). Previously it
  only checked `data.obs.columns`, so variable-axis tooltips were dropped.
- **Plural wrappers resolve tooltips into the shared frame** before building it
  (`dimensionals`/`umaps`/`tsnes`/`pcas`/`expressions`, `xyplots`, `spatials`,
  `ridges`). Each subplot validates tooltips against the shared frame, so the
  shared frame must contain custom tooltip columns up front.
- **`ridge` resolves tooltips before `build_frame`** (was after), so custom
  metadata tooltips reach `metadata_columns`.
- **`heatmap(axis=1)`** now materialises `keys` via `metadata_columns` (on the
  variable axis they are metadata columns, not genes pulled from X, and the
  variables-frame builder ignores `variable_keys`).

- **`_distribution` (violin/boxplot) resolves tooltips before the unpivot and
  adds tooltip-referenced metadata columns to the unpivot `index`.** These plots
  melt to long form with `index=[group_by, *add_keys]`, which drops any column
  not in the index. So a custom obs tooltip must be (a) materialised into the
  frame and (b) kept in the unpivot index. `violins`/`boxplots` additionally
  resolve tooltips into the shared frame before building it, mirroring the other
  plural wrappers.

### `add_columns` escape hatch

Added an `add_columns: Sequence[str] | str | None = None` parameter to the
per-observation plot functions (`dimensional` + `umap`/`tsne`/`pca`/`expression`,
their plural variants, `xyplot`/`xyplots`, `spatial`/`spatials`). It is the
user-facing seed of the same accumulator the auto-derived columns flow into:
its entries are routed through `_collect_aes_columns` alongside `key`/`mapping`,
so obs/var metadata land in `metadata_columns` and gene names in
`variable_keys`. Use it when an added layer reads a column the plot itself does
not reference. `ridge`/`ridges` and `violins`/`boxplots` already expose
`add_keys`, which now serves the same role (it is routed through
`_collect_aes_columns` too), so no separate `add_columns` was added there.

Two changes in this pass, applied to `dimensional` only:

1. **`build_frame` learnt column narrowing.** Added two opt-in params on
   `build_frame`, `anndata_observations_frame`, `anndata_variables_frame`:
   - `metadata_columns: Sequence[str] | None = None` — restrict which
     observation/variable metadata columns are materialised
     (`None` = current behaviour, all columns)
   - `dimension_keys: Sequence[str] | None = None` — restrict which embeddings
     are materialised, case-insensitive (`None` = all embeddings)

   `_select_embedding_keys` is a private helper in
   [cellestial/frames/build.py](../cellestial/frames/build.py) that does the
   case-insensitive lookup and raises `KeyError` on unknown keys.

2. **`_resolve_tooltips` mirrors the variable-keys mutator.** Now also takes an
   optional `metadata_columns: list[str]` and appends any tooltip field that is
   an observation metadata column, so callers requesting a narrow frame don't
   accidentally drop user-referenced tooltip columns.

3. **`dimensional` assembles the minimum set.**
   - Adds `key` (when it is a metadata column) to `metadata_columns`
   - Scans `mapping.as_dict()` values for column references (covers
     `mapping=aes(shape="leiden")` etc.)
   - Lets `_resolve_tooltips` extend both `variable_keys` and `metadata_columns`
   - Passes `metadata_columns=metadata_columns` and `dimension_keys=[prefix]`
     to `build_frame`

4. **One-line bug fix in `anndata_observations_frame`.**
   `if variable_keys is not None:` → `if variable_keys:`. `dimensional`
   normalises `None` → `[]` internally, so the old guard always evaluated true
   and called `data[:, []].X`, which triggered a full sparse-matrix slice on
   the entire X matrix. Profiling showed this single line cost 8 seconds out of
   the 8.2s total plot build on the benchmark dataset. The narrowing work in
   (1)-(3) saved ~0.15s on its own; this fix is what unlocked the
   ~7.5s remaining gap.

Backward compatibility: `cl.build_frame` public API is preserved — every new
param defaults to `None`, which keeps the original "all columns / all
embeddings" behaviour. The `frame=` pre-built path in `dimensional` is
unchanged.

## Pass 2: rollout to the rest of the plot family

Same pattern applied to every direct `build_frame` caller, using a new shared
helper `_collect_aes_columns` in [cellestial/util/utilities.py](../cellestial/util/utilities.py)
that routes key+mapping references into `metadata_columns` (obs/var metadata)
and `variable_keys` (gene names) for the chosen axis.

Functions updated in this pass:

- [cellestial/single/core/dimensional.py](../cellestial/single/core/dimensional.py) — `dimensional`
- [cellestial/single/core/subdimensionals.py](../cellestial/single/core/subdimensionals.py) — `dimensionals`, `umaps`, `tsnes`, `pcas`, `expressions`
- [cellestial/single/core/distributions.py](../cellestial/single/core/distributions.py) — `violins`, `boxplots`
- [cellestial/single/core/utilities.py](../cellestial/single/core/utilities.py) — `_distribution` (shared violin/boxplot impl)
- [cellestial/single/common/xyplot.py](../cellestial/single/common/xyplot.py) — `xyplot`
- [cellestial/single/common/xyplots.py](../cellestial/single/common/xyplots.py) — `xyplots`
- [cellestial/single/quick/ridge.py](../cellestial/single/quick/ridge.py) — `ridge`, `ridges`
- [cellestial/single/heatmap/heatmap.py](../cellestial/single/heatmap/heatmap.py) — `heatmap`
- [cellestial/single/heatmap/dotplot.py](../cellestial/single/heatmap/dotplot.py) — `dotplot`
- [cellestial/single/heatmap/stacked_violin.py](../cellestial/single/heatmap/stacked_violin.py) — `stacked_violin`
- [cellestial/spatial/spatial.py](../cellestial/spatial/spatial.py) — `spatial` (preserves `instance_key` for polygon mode)
- [cellestial/spatial/spatials.py](../cellestial/spatial/spatials.py) — `spatials` (preserves `instance_key` for polygon mode)

Deliberately **not** updated:
- [cellestial/single/base/base.py](../cellestial/single/base/base.py) `cl.plot` — generic user-extensible entry point; users may add geoms with aes refs we cannot inspect at call time, so narrowing here would silently break their layered code. Kitchen-sink frame stays.

### Embedding narrowing reverted (stream regression)

An earlier version of this pass also narrowed `dimension_keys` to just the
plotted embedding (e.g. `["X_UMAP"]`). This broke `cl.stream()`: the deferred
velocity layer reads a *different* embedding (`velocity_umap` ->
`VELOCITY_UMAP1/2`) from the already-built frame, which the narrowing excluded.
Because `cl.stream()` lets users name velocity columns arbitrarily
(`velocity_key=` / `velocity_prefix=`), the plot call cannot predict which
embeddings a later layer will need.

Resolution: **do not narrow `dimension_keys` in any plot function.** Embeddings
are cheap (each capped at `max(xy)` columns by `include_dimensions`), so this
costs almost nothing, and it keeps all deferred layers working. The
`dimension_keys` parameter still exists on `build_frame` as an opt-in for power
users, but no internal caller uses it.

The big wins all come from `metadata_columns` narrowing (skipping unused obs
columns), the `observations_name=None` Barcode skip, and the
`if variable_keys:` sparse-slice bug fix — none of which touch embeddings.

Audited the other deferred layers (`arrow_axis`, `ondata_legend`,
`cluster_outlines`, `bracket`): they only read `x`/`y`/`color`, which are the
plot's own aesthetics and always present, so they are unaffected by
`metadata_columns` narrowing.

## How to roll this out to the other plot functions

The pattern is the same for any plot that calls `build_frame` and currently
pulls all observation/variable columns. Per plot function:

1. **Identify the minimum data the plot consumes.** Usually:
   - the colour/grouping `key` (if it lives in obs/var metadata)
   - any explicit `variable_keys` argument (already handled)
   - any `mapping` aes references
   - any tooltip field references
   - one or more embedding keys (for plots that use embeddings)

2. **Build a `metadata_columns: list[str]` locally** the same way `dimensional`
   does. Append `key` if it is an obs/var metadata column. Scan
   `mapping.as_dict()` values; route strings either to `metadata_columns` (if
   it is a metadata column) or to `variable_keys` (if it is a variable name).

3. **Pass `metadata_columns=metadata_columns` to `_resolve_tooltips`** so it
   can append tooltip-referenced metadata columns too. Use the existing
   `metadata_columns` kwarg — no signature change needed on the helper.

4. **Pass `metadata_columns=metadata_columns` to `build_frame`.** Do **not**
   pass `dimension_keys` — see the "Embedding narrowing reverted" note above.
   Embeddings stay governed by `include_dimensions` so deferred layers like
   `cl.stream()` can still read velocity embeddings from the frame.

5. **For variable-axis plots** (heatmaps, dotplots, etc.), the same pattern
   applies but against `data.var.columns` / `data.varm`. `_resolve_tooltips`
   currently only checks `data.obs.columns`; if you need it to also handle
   var-axis tooltip refs, add an `axis` parameter to the helper (out of scope
   for the current pass).

Candidates ordered by likely benefit (largest first), based on how aggressively
they pull from obs/obsm today:

- `cellestial/single/core/subdimensional.py::umap` etc. (these are thin
  wrappers around `dimensional` and *already inherit* this fix — confirm by
  reading the wrappers; no action needed if they just call `dimensional`)
- `cellestial/single/core/subdimensionals.py::dimensionals`, `::umaps`
  (multi-panel variants; share the same overhead)
- `cellestial/single/core/scatter.py` (if it exists; same pattern)
- `cellestial/spatial/spatial.py`, `spatials.py`
- `cellestial/single/heatmap/heatmap.py`, `dotplot.py`, `stacked_violin.py`
  (var-axis — may need the helper change in (5))
- `cellestial/single/core/distribution.py`, `distributions.py`

---

## Instructions for a follow-up agent

You are picking up after a perf pass that landed on `cellestial/single/core/dimensional.py`. Your job is to apply the same narrowing pattern to the rest of the plot functions that call `build_frame`.

### Pre-reads (do these first)

1. [plans/frame_overhead_narrowing.md](frame_overhead_narrowing.md) (this file) — the summary above.
2. [cellestial/single/core/dimensional.py](../cellestial/single/core/dimensional.py) — the reference implementation. Look at how `metadata_columns` is assembled and how `_resolve_tooltips` and `build_frame` are called.
3. [cellestial/frames/build.py](../cellestial/frames/build.py) — the new `metadata_columns` and `dimension_keys` params on `build_frame`, `anndata_observations_frame`, `anndata_variables_frame`, and the `_select_embedding_keys` helper.
4. [cellestial/util/utilities.py](../cellestial/util/utilities.py) — `_resolve_tooltips` now accepts a mutating `metadata_columns: list[str] | None = None`.
5. [MEMORY.md](/Users/zaf4/.claude/projects/-Users-zaf4-datavil-cellestial/memory/MEMORY.md) — especially the feedback memories on naming, docstring style, no AnnData internals in docstrings, expressive names.

### What to do, per plot function

Apply the recipe in "How to roll this out" above. Keep edits surgical — don't refactor unrelated code.

### Constraints

- Public API of `cl.build_frame` must stay backward compatible (all new params default to `None` = current behaviour). Already done; don't break it.
- Docstrings: no AnnData internal slot names (`data.obs`, `data.obsm`); say "observation metadata", "embeddings", etc.
- Use full words for new variables, not abbreviations (`metadata_columns`, not `meta_cols`).
- Don't touch `cl.build_frame` callers that pass their own pre-built `frame=` — keep that path unchanged.

### Verification per function

1. `NUMBA_DISABLE_JIT=1 poetry run pytest tests/test_<function>.py -x -q`
2. For perf-sensitive functions, profile against the breast_cancer_atlas dataset (see [notebooks/benchmark_cold.py](../notebooks/benchmark_cold.py) for the pattern). Confirm `anndata_observations_frame` / `anndata_variables_frame` shows as a small fraction of total time in cProfile.

### Pitfall to avoid

The `if variable_keys is not None:` → `if variable_keys:` fix is already landed. If you see code elsewhere that calls `anndata_variable_columns(..., keys=[])` or `data[:, []].X`, that is the same sparse-slice trap — flag it.

### Out of scope

- PDF/PNG export still trails scanpy. That cost lives in the lets-plot rasteriser, not in cellestial. Do not chase it as part of this rollout.
- Adding axis-awareness to `_resolve_tooltips` for var-axis plots — only do this if a heatmap/dotplot in scope actually needs it; otherwise leave for a separate pass.

### Logging

Add a row in [plans/audit_AI.md](audit_AI.md) for each function you modify.
