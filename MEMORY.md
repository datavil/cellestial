

## MEMORY

# Memory Index

- [Project: Cellestial architecture & design](project_cellestial_architecture.md) — Single-cell viz library built on Lets-Plot + AnnData; layered modular design, Polars-first, grammar of graphics
- [Project: Cellestial function inventory](project_cellestial_functions.md) — What every function does: frames, single plots, layers, util, themes
- [Project: Cellestial tooling](project_cellestial_tooling.md) — Uses Poetry (not uv/pip) for deps and running commands
- [Project: Cellestial code patterns](project_cellestial_patterns.md) — Naming conventions, parameter patterns, singular/plural pairs, type annotations, data transformation patterns
- [Feedback: Test code changes](feedback_test_code.md) — Run actual test code against data/pbmc3k_pped.h5ad before reporting success

- [Feedback: Sync repo MEMORY.md](feedback_sync_repo_memory.md) — Mirror every auto-memory change in repo's `MEMORY.md` (index line + inlined source block)
- [Feedback: Avoid shortened variable names](feedback_naming.md) — No abbreviations like `grp_idx`/`obs`/`cfg`; use full words (only `n`-style counts OK)
- [Feedback: Docs avoid AnnData internals](feedback_docs_no_anndata_internals.md) — Don't reference AnnData slots OR scanpy APIs (`scanpy.tl.dendrogram`, `key_added`) in user-facing docstrings; abstract them
- [Feedback: Docstring backtick discipline](feedback_docstring_backticks.md) — Don't sprinkle double backticks; match neighboring params (single backticks for identifiers, plain text otherwise)
- [Feedback: Docstring single-line summary](feedback_docstring_single_line_summary.md) — One-line summary then Parameters; extra context goes into a Notes section, never floating prose
- [Feedback: Don't overengineer key resolvers](feedback_dont_overengineer_resolvers.md) — `use_key | None -> default` is a one-liner; no scanning, no candidate ranking, no shape helpers unless there's real ambiguity
- [Feedback: Report only findings](feedback_report_only_findings.md) — When asked for weaknesses/problems, list those only; no "Not weaknesses (for completeness)" filler
- [Feedback: Never use em dashes](feedback_no_em_dashes.md) — User hates em dashes; never write them in code, docs, comments, or chat
- [Feedback: `if isinstance(data, AnnData)` block stays](feedback_isinstance_anndata_block.md) — Never flatten to `if not isinstance: raise`; positive branch hosts AnnData-specific ops for future multi-backend support
- [Feedback: No AnnData language in plot bodies](feedback_no_adata_language_in_plots.md) — Plot functions and their error messages stay backend-agnostic; adata.uns/obs/etc. live in helpers under `if isinstance(data, AnnData):`
- [Feedback: No hidden aesthetic promotion](feedback_no_hidden_aesthetic_promotion.md) — Don't auto-promote `group_by` to `fill`/`color`; user wires aesthetics explicitly, prefer warn+facet over hidden magic
- [Project: Lets-plot scale_size shrinks geom_text](project_letsplot_scale_size_text_quirk.md) — `scale_size(range=...)` silently shrinks `geom_text(size=N)` constants ~5x; bypass with `size_unit="x"`/`"y"`. Used in `_key_groups.py` dual-mode design.
- [Feedback: Explicit list+extend over splat](feedback_explicit_list_extend.md) — Prefer `x = list(x); x.extend(...)` over `[*a, *b]` when merging a sequence param with computed items
- [Feedback: Record AI-written functions in audit_AI.md](feedback_audit_ai_tracking.md) — Any function an agent generates or significantly modifies gets a row in `plans/audit_AI.md`; leave verification columns blank for the user to fill
- [Project: Polars drop/exclude null gotcha](project_polars_drop_null_gotcha.md) — Exclude filters need `~col.is_in(values).fill_null(False)` or they silently drop null-category rows too
- [Project: Deferred layers read the built frame](project_deferred_layers_read_frame.md) — Don't narrow embeddings in plot functions; `cl.stream()` reads velocity embeddings from the frame after the plot call
- [Feedback: Track breaking changes in CHANGELOG.md](feedback_changelog_breaking_changes.md) — Record breaking API changes under the current poetry version in repo-root `CHANGELOG.md` (`### Breaking` + migration note)
- [Project: geom_raster aspect lock](project_letsplot_geom_raster_aspect.md) — lets-plot geom_raster forces square pixels; cl.heatmap rescales y, cl.annotated_heatmap (cellestial/complex) uses geom_tile + gggrid(align, guides)
- [Project: annotated_heatmap overflow](project_annotated_heatmap_overflow.md) — cl.annotated_heatmap layout: dendrogram clip fixed (expand left margin), legend overflow fixed (bottom horizontal compact, set on subplots+grid); per-track brewer palettes; bar-end labels tried+reverted (leftmost row-label column clips). ggsize is not a fix.


---

## Source: project_deferred_layers_read_frame.md

---
name: project-deferred-layers-read-frame
description: Don't narrow embeddings (obsm/dimension_keys) in plot functions; deferred layers like cl.stream() read extra embedding columns from the built frame
type: project
---
Plot functions must NOT narrow which embeddings get materialised into the frame (i.e. do not pass `dimension_keys` to `build_frame` in `dimensional`, `umap`, etc.). Embeddings stay governed by `include_dimensions`.

**Why:** Deferred layers added after the plot call (notably `cl.stream()`, the velocity layer in `cellestial/layers/stream.py`) read columns from the *already-built* frame via `retrieve(plot)`. `cl.stream()` needs a *different* embedding than the plotted one — e.g. plotting `X_umap` but reading `velocity_umap` -> `VELOCITY_UMAP1/2`. Since `cl.stream()` lets users name velocity columns arbitrarily (`velocity_key=` / `velocity_prefix=`), the plot call cannot predict which embeddings a later layer will need. Narrowing `dimension_keys=[prefix]` caused `KeyError: Velocity columns not found`.

**How to apply:** When optimizing frame builds, narrow `metadata_columns` (unused obs/var columns) and skip the identifier column (`observations_name=None`) freely, but leave embeddings alone. Embeddings are cheap anyway (each capped at `max(xy)` columns). The other deferred layers (`arrow_axis`, `ondata_legend`, `cluster_outlines`, `bracket`) only read `x`/`y`/`color`, which are the plot's own aesthetics and always present, so `metadata_columns` narrowing is safe for them. See plans/frame_overhead_narrowing.md.

## Source: feedback_sync_repo_memory.md

---
name: Keep repo's MEMORY.md in sync with auto-memory
description: Whenever auto-memory is added/updated/removed, mirror the change in the repo's MEMORY.md at the project root (both the index line and the inlined source block)
type: feedback
---
Whenever you write, update, or delete a memory in the auto-memory directory (`~/.claude/projects/<project-slug>/memory/`), mirror the same change in the repo's `MEMORY.md` at the project root.

**Why:** The user maintains a checked-in `MEMORY.md` in the project root that aggregates the same memories alongside their inlined contents. The two should not drift; the repo file is the shareable / reviewable copy and goes to a public repo, so do not write absolute paths or usernames into it.

**How to apply:** The repo's `MEMORY.md` has two sections that both need updating in lockstep:
1. **Index lines** at the top — same format as auto-memory's `MEMORY.md`: `- [Title](file.md) — one-line hook`.
2. **Inlined source blocks** below, each under `## Source: <filename>.md` with the full frontmatter + body.

On every memory write/update/delete:
- Add/edit/remove the corresponding index line.
- Add/edit/remove the corresponding `## Source:` block (full content, not just a link).
Keep ordering consistent between the index and the source blocks.
Never include absolute home paths (e.g. `/Users/<name>/...`) in repo content; use `~` or relative descriptions.


---

## Source: feedback_docs_no_anndata_internals.md

---
name: Docs avoid AnnData-internal references
description: Don't reference AnnData/scanpy internals (uns['spatial'], obsm['X'], scanpy.tl.dendrogram, key_added) in user-facing docs and docstrings
type: feedback
originSessionId: 38fa3aa3-d285-4124-9b4f-be3c011e598b
---
Never name backend-specific structure or APIs in user-facing docstrings, Notes blocks, or error messages aimed at end users. This covers two categories:

1. **AnnData container slots:** `uns['spatial']`, `obsm['spatial']`, `obs['cluster']`, `var_names`, `layers['counts']`.
2. **Scanpy / other backend function names and parameter names:** `scanpy.tl.dendrogram`, `sc.tl.rank_genes_groups`, `key_added`, etc.

Use abstracted language instead: "image data", "spot coordinates", "the precomputed dendrogram", "the ranking", "the dataset".

**Why:** Cellestial aims to abstract over the data container (AnnData today, possibly others later — the architecture already accepts non-AnnData per `plans/TODO.md`). Surfacing AnnData or scanpy internals in docs leaks the implementation, dates the docs, and confuses users on alternative backends. Telling users "useful when `scanpy.tl.dendrogram` was run with custom `key_added`" assumes both the backend and the workflow.

**How to apply:** When documenting parameters, behavior, or notes, describe what the input *represents* (image, coordinates, gene expression, precomputed dendrogram) rather than where it lives or which library produced it. Internal code comments and private helpers referencing `uns/obsm/scanpy/etc.` are fine — this rule is for public docstrings and user-visible text only.


---

## Source: feedback_docstring_backticks.md

---
name: Docstring backtick discipline
description: Don't sprinkle RST-style double backticks in user-facing docstrings; use single backticks for python identifiers, plain text otherwise
type: feedback
---
Keep docstring formatting plain. Match the style of neighboring parameters in the same function — most existing cellestial docstrings use single backticks for python identifiers (`group_by`, `keys`, `aes()`) and plain text for everything else. Avoid wrapping prose phrases or default-value descriptions in double backticks reflexively.

**Why:** Reflexively reaching for RST-style double backticks (a habit borrowed from sphinx-heavy projects) clutters the rendered docstring with backtick noise and breaks tonal consistency with the rest of the file. The user flagged this directly: "you are using double backticks so freely." See `feedback_docs_no_anndata_internals.md` — the same review pass.

**How to apply:** Before submitting a docstring change, glance at the surrounding params. If they use plain prose with occasional single-backtick identifiers, match that. Reserve double backticks for true code-block-like literals where single backticks would be ambiguous, and even then prefer the shorter form when the rest of the docstring uses it.


---

## Source: feedback_docstring_single_line_summary.md

---
name: docstring-single-line-summary
description: Cellestial docstrings must have a single-line summary then go directly to Parameters; any extra context goes into a Notes section
type: feedback
---

Cellestial function docstrings should consist of exactly one summary line followed directly by the Parameters section. Do not add a paragraph below the summary line. If the function needs additional explanation that provides important context, place it under a `Notes` section (NumPy docstring style), not floating between the summary and Parameters.

**Why:** The user wants consistent, scannable docstrings. Free-floating prose between the summary and Parameters block clutters the rendered API docs and breaks the NumPy convention used in cellestial. Important context is fine, but it belongs in `Notes` where it's properly labeled.

**How to apply:** When writing or editing any cellestial docstring:
- Keep the first line a single concise summary sentence.
- Go straight to `Parameters` after a blank line.
- If there is meaningful context (caveats, behavior notes, references to related plots/keys), put it in a `Notes` section after `Returns`/`Examples`.
- Pair with `feedback_docstring_backticks.md` (reserve double backticks for true literals) and `feedback_docs_no_anndata_internals.md` (don't leak AnnData/scanpy internals).


---

## Source: feedback_dont_overengineer_resolvers.md

---
name: Don't overengineer key resolvers
description: For "use_key | None -> default" parameters, just do `key if key is not None else default` — no fuzzy matching, scanning, candidate ranking, or shape-detection helpers
type: feedback
---
When adding a "specify a key, default to something" parameter, the body should be one line: `key = use_key if use_key is not None else default_key`. Don't build resolvers that scan the container for similarly-named candidates, validate the value's stored shape, rank candidates by length, or warn about alternatives. Don't add structural-check helpers (`_looks_like_X`) or field-extractors (`_stored_Y`) as scaffolding. If the named key doesn't exist, fall through to whatever fallback the existing code had (recompute, raise, etc.).

**Why:** The user explicitly called this out: *"if dendrogram_key is None just use the default else use what is given. do not overcomplicate it."* The fancy resolver I built (`_resolve_dendrogram_key` with prefix/groupby matching, tiebreak rules, multi-candidate warnings, helper functions for shape detection and field unwrapping) was ~70 lines plus 9 tests for what should be 1 line plus 0 extra tests. The earlier `_resolve_embedding_key` (the user's own commit) IS more elaborate, but only because embedding lookups have a real ambiguity (multiple obsm keys with shared prefix, dimensionality varies). Dendrogram lookup has no such ambiguity — there's exactly one key the user named, or the canonical default. Don't generalize from one resolver's complexity to all resolvers.

**How to apply:** When introducing a `foo_key: str | None = None` parameter, first ask "is there genuine ambiguity to resolve, or just a default?" If it's just a default, write the one-liner. Only build a resolver when the search space actually has multiple plausibly-correct candidates and the user benefits from automatic disambiguation. See `feedback_docstring_backticks.md` and `feedback_docs_no_anndata_internals.md` for adjacent docstring lessons from the same review pass.


---

## Source: feedback_report_only_findings.md

---
name: Report only findings, no "for completeness" filler
description: When asked for weaknesses/problems/issues, list those only — don't pad with "things I checked but were fine"
type: feedback
---
When the user asks to scan for weaknesses, bugs, or problems, the report contains only what was found. Do not add a "Not weaknesses (worth noting for completeness)" / "Things I checked that look OK" / "Out of scope" section. Do not list code paths you inspected and cleared.

**Why:** The user called this out directly: *"why did you tell me this, what is the problem??"* — pointing at a section I'd labeled "Not weaknesses (worth noting for completeness)" in a weakness scan. Padding a findings report with non-findings:
- Inflates the report with content the user didn't ask for.
- Forces them to re-read and dismiss each item to confirm it's not a problem.
- Reads as self-congratulatory ("look at everything I checked"), not service.
- Dilutes the signal of the real findings.

**How to apply:** If asked "find X", return only the X you found. If something looked suspicious but turned out fine, drop it from the report. The exception is when *not finding* something is the answer itself ("no issues found in <area>") — that's a finding, not filler.


---

## Source: feedback_isinstance_anndata_block.md

---
name: Keep AnnData operations under `if isinstance(data, AnnData):` blocks
description: Never flatten the `if isinstance(data, AnnData):` block into an early-return type check; the block hosts AnnData-specific logic for future multi-backend support
type: feedback
originSessionId: 38fa3aa3-d285-4124-9b4f-be3c011e598b
---
When code uses `if isinstance(data, AnnData):` followed by AnnData-specific operations, **never** rewrite it into the early-return form `if not isinstance(data, AnnData): raise TypeError(...)`. Keep the positive-branch block.

**Why:** The library is designed to support more data containers in the future (per `plans/TODO.md` "ACCEPT possibly different objects other than AnnData"). The `if isinstance(data, AnnData): ...` shape exists so a future `elif isinstance(data, OtherType): ...` branch can be added without restructuring. Flattening it to a guard erases that hook and forces a refactor when a new backend lands.

**How to apply:** Anywhere a function dispatches on data type, structure as:

```python
if isinstance(data, AnnData):
    # AnnData-specific operations
    ...
else:
    msg = f"Unsupported data type: `{type(data)}`"
    raise TypeError(msg)
```

The trailing `else` raises on unknown types but the positive branch stays available for extension. Do NOT collapse to `if not isinstance: raise` even when it looks tidier; the user has explicitly chosen the verbose form for extensibility.


---

## Source: feedback_naming.md

---
name: Avoid shortened variable names
description: User dislikes abbreviated/shortened variable names in cellestial code; prefer full words
type: feedback
originSessionId: e066ee95-b168-470f-88b2-cdfc6623abda
---
Do not use shortened/abbreviated variable names (e.g. `grp_idx`, `obs`, `cfg`). Use full descriptive names like `group_index`, `group_indices`, `group_positions`.

**Why:** User explicitly stated they do not use shortened names unless the abbreviation is extremely obvious (like `n` for count). Readability over brevity.

**How to apply:** When writing or reviewing code in the cellestial project, expand abbreviated identifiers to full words. Only `n`-style single-letter counts are acceptable shorthand. Applies to local variables, parameters, and column names.


---

## Source: feedback_no_em_dashes.md

---
name: Never use em dashes
description: User strongly dislikes em dashes; never write them in code, docs, comments, or chat
type: feedback
originSessionId: 38fa3aa3-d285-4124-9b4f-be3c011e598b
---
Never use em dashes (—). User explicitly said "I hate them".

**Why:** Personal stylistic preference, stated directly.

**How to apply:** Applies everywhere: chat replies, docstrings, comments, commit messages, plan docs. Restructure the sentence instead. Use a period, comma, semicolon, parentheses, or colon. En dashes (–) and hyphens (-) are fine.


---

## Source: feedback_no_hidden_aesthetic_promotion.md

---
name: No hidden aesthetic promotion in plot APIs
description: Don't auto-promote one aesthetic (e.g. group_by) to another (e.g. fill) inside cellestial plot functions; the user wires aesthetics explicitly.
type: feedback
originSessionId: 663f107b-1049-48c8-9838-82cc5f88f988
---
When fixing or designing plot function APIs in cellestial, do not auto-promote a parameter like `group_by` to a different aesthetic slot (e.g. `fill`/`color`) when the situation would otherwise be ambiguous (multi-key + explicit `group_by`, etc.).

**Why:** The user designed `distribution.py` (and similar) to keep aesthetic mapping explicit on purpose. Auto-promoting `group_by` to `fill` for multi-key calls (the seaborn `hue=` style) is hidden behavior the user deliberately avoided. They prefer that the user wires `fill=`/`color=` themselves so the plot's aesthetics match exactly what was written.

**How to apply:**
- When the chosen `group_by` cannot be honored on the x-axis (e.g. multi-key forces `variable` onto x), warn and discard rather than silently re-routing it to `fill`/`color`.
- For multi-axis layouts (multi-key + explicit grouping), prefer faceted/grid variants (`violins`, `boxplots`, `volcanos`, etc.) over auto-dodging in the single-plot function.
- This applies to the whole plot family: `violin`/`boxplot`/`ridge`/`dotplot`/etc. — don't introduce hue-style auto-magic.


---

## Source: feedback_test_code.md

---
name: Test code changes before reporting success
description: User expects me to run the actual test code to verify changes work, not just syntax-check
type: feedback
---

Test code changes by running them against the actual dataset before reporting success.

**Why:** User called out that I should have tested the code myself instead of just checking syntax. The project has test data at `data/pbmc3k_pped.h5ad`.

**How to apply:** After implementing changes to plot functions, run the user's example code (or a minimal equivalent) via Bash to confirm it executes without error. Don't rely on syntax checks or linting alone.


---

## Source: feedback_no_adata_language_in_plots.md

---
name: No AnnData-specific language in main plotting functions
description: Plotting function bodies and error messages must stay backend-agnostic; AnnData-specific language (adata.uns, obs, obsm, etc.) belongs in helpers under `if isinstance(data, AnnData):`
type: feedback
originSessionId: 4a073345-7f3f-4538-b5fe-681d8d0358a1
---
Main plotting functions (heatmap, dotplot, etc.) must not reference AnnData internals (`adata.uns`, `adata.obs`, `adata.obsm`, `adata.X`, etc.) in code paths, error messages, or warnings.

**Why:** The library is designed to support multiple backends. Plotting functions describe what the user sees; AnnData-specific extraction belongs in helpers that get dispatched from inside `if isinstance(data, AnnData):` blocks. Leaking AnnData language into the plotting layer entangles the public API with one backend.

**How to apply:**
- Put AnnData reads (and the error messages that mention them) inside helpers (e.g. `_rank_genes_groups.py`).
- In the plotting function, wrap calls to those helpers in `if isinstance(data, AnnData):` so future backends can dispatch differently.
- Error messages in the plotting function should reference user-facing parameter names (`keys`, `group_by`, `markers`), not backend internals.


---

## Source: project_letsplot_scale_size_text_quirk.md

---
name: Lets-plot scale_size silently shrinks geom_text constants
description: When a plot has scale_size with an explicit range, geom_text(size=...) constants get re-routed through the size scale and render tiny (~5x smaller). Use size_unit="x" or "y" to bypass.
type: project
originSessionId: f7c1e22b-92de-4d63-a280-da2817eac2f7
---
When a lets-plot plot includes `scale_size(range=[a, b])` (range explicitly set),
ANY `geom_text(size=N)` constant in that plot gets re-mapped through the size
scale and renders much smaller than expected (empirically ~5-6x shrinkage with
typical dotplot ranges). `scale_size` with only `trans=` or `breaks=` does NOT
have this effect; `scale_size_continuous`, `scale_size_area`, `scale_size_identity`
are untested but suspected to behave similarly when `range` is set.

**Why:** lets-plot treats the size aesthetic as plot-global, even when `geom_text`
passes `size` as a constant rather than via `aes(size=...)`. The size scale's
output range applies to all size values, including text constants.

**How to apply:** Whenever a layer adds rotated/scaled text on top of a dotplot
(or any plot with `scale_size(range=...)`), pass `size_unit="x"` or `size_unit="y"`
to `geom_text`. That makes the text size be in axis units and bypasses the
scale_size routing. Calibrate the size value differently:
- Without size_unit: size is in lets-plot text units (~pt-like, e.g. 4-6 typical).
- With size_unit="y": size is in y-axis units (e.g. 0.2 means cap height = 0.2 y).

This is exactly why `cellestial/single/heatmap/_key_groups.py` has dual-mode
sizing: heatmap uses absolute units (no scale_size in plot), dotplot and
stacked_violin pass `size_unit="y"` because dotplot adds
`scale_size(range=[size_max*0.1, size_max])` for the dot sizing.

**To reproduce / verify:** Build a plot with `geom_point(aes(size="value"))` +
`scale_size(range=[3, 30])` + `geom_text(size=8.6)`. The text renders at ~1px.
Add `size_unit="y"` to geom_text and the text renders at proper visual size.


---

## Source: project_cellestial_architecture.md

---
name: Cellestial architecture & design philosophy
description: Overview of cellestial's architecture, design philosophy, and how layers/plots/API fit together
type: project
---

Cellestial is a **single-cell visualization library** built on **Lets-Plot** and **AnnData**, designed for bioinformatics workflows.

**Why:** Purpose-built to give single-cell bioinformaticians a clean, composable plotting API that understands AnnData structure natively.

**How to apply:** New features should follow the layered, composable grammar-of-graphics model; plots are Lets-Plot `PlotSpec` objects that support `+` operator chaining.

## Architecture layers

1. **Base Layer** — `frames/` — AnnData → Polars DataFrame construction
2. **API Layer** — `single/` — Plot-generation functions by type
3. **Enhancement Layer** — `layers/` and `themes/` — additive plot layers and theme specs
4. **Utility Layer** — `util/` — helpers for operations on plots and data

## Data flow

```
User call (e.g., cl.umap(...))
  → Validation & preprocessing
      (axis inference, mapping merge, tooltip prep)
  → build_frame() → Polars DataFrame
  → ggplot() + geom_X() + scale + theme + optional interactive
  → Optional layers: arrow_axis / stream / cluster_outlines
  → Optional grid wrapper: gggrid() → SupPlotsSpec
```

## Module structure

```
cellestial/
├── __init__.py          # Public API exports (flat namespace)
├── _version.py
├── frames/              # build_frame: AnnData → Polars DF
├── single/
│   ├── base/            # plot() — bare ggplot wrapper
│   ├── basic/           # scatter, bar, heatmap
│   ├── common/          # xyplot, xyplots
│   ├── core/            # violin/boxplot, dimensional reduction
│   ├── special/         # dotplot
│   └── quick/           # (quick plotting utilities)
├── layers/              # arrow_axis, stream, cluster_outlines
├── themes/              # _THEME_DIMENSION, _THEME_SCATTER, etc.
└── util/                # get_mapping, retrieve, slice, colors, errors
```

## Core design philosophy (from philosophy.md)

Borrowed from ggplot, Zen of Python, and Rust.

1. **Modularity over abstraction** — Don't add params (e.g., `width`, `height`) for things the user can do themselves with `+ ggsize(800, 600)`. Accepting fewer params eliminates complexity and the risk of conflicting arguments. Especially valuable in notebooks/EDA.

2. **Predictability over flexibility** — Strict return types. `cl.umap(key: str) → PlotSpec` always. `cl.umaps(keys: Sequence[str]) → SupPlotsSpec` always. Never accept both a single key and a sequence in the same function to avoid inconsistent return types. Exception: `violin(keys: Sequence[str])` still returns `PlotSpec` (merges multiple geoms on one plot); `violins()` returns `SupPlotsSpec`.

3. **Explicitness over implicity** — Full names always: `violin()` not `vln()`, `include_dimensions=` not `inc_dims=`. Users must not need to remember the programmer's abbreviation choices.

4. **Simplicity & expressiveness** — No `plot` suffix in function names (`umap` not `umap_plot`), no sub-namespaces. The API is flat and self-evident.

## Additional design principles

- **Composability**: plots and layers combine with `+` (Lets-Plot)
- **AnnData-aware**: native understanding of obs, var, obsm, varm structure
- **Polars-first**: all DataFrame operations use Polars (not pandas)
- **Pre-configured themes**: one theme constant per plot family ensures visual consistency
- **Flat public API**: everything re-exported from `cellestial/__init__.py`


---

## Source: project_cellestial_functions.md

---
name: Cellestial function inventory
description: What every function in cellestial does — frames, single plots, layers, util, themes
type: project
---

## frames/build.py

- `build_frame(data, axis, variable_keys, include_dimensions)` — Main orchestrator: AnnData → Polars DataFrame
- `anndata_observations_frame()` — Extracts axis=0 (cells/observations)
- `anndata_variables_frame()` — Extracts axis=1 (genes/variables)
- Handles: categorical integer conversion, dimension inclusion (coords from .obsm/.varm), variable key lookup

## single/base/base.py

- `plot(data, mapping, axis, variable_keys, include_dimensions)` — Bare ggplot() wrapper; builds frame, no geom added

## single/basic/

- `scatter()` — geom_point(); optional `interactive=True` adds ggtb()
- `bar()` — geom_bar() for categorical/count data
- `heatmap()` — geom_tile() for 2D matrices, applies _THEME_HEATMAP

## single/common/

- `xyplot(data, x, y, mapping, tooltips, ...)` — Scatter with explicit x/y; auto tooltip generation; applies _THEME_SCATTER
- `xyplots(data, x, y, ...)` — Grid of xyplots; x/y broadcasting (single val replicated to match lengths); returns SupPlotsSpec

## single/core/distribution.py + distributions.py

- `_distribution(geom, data, key, ...)` — Core impl: unpivots data to long format, supports point overlays (jitter/point/sina), threshold filtering, position_jitterdodge
- `violin()` — Calls _distribution(geom="violin")
- `boxplot()` — Calls _distribution(geom="boxplot")
- `violins(data, keys, ...)` — Grid: one violin per key; share_ticks/share_axis remove redundant labels
- `boxplots(data, keys, ...)` — Grid: one boxplot per key

## single/core/dimensional.py + subdimensional.py + subdimensionals.py

- `dimensional(data, key, dimensions, xy, ...)` — Core dim-reduction plot; handles categorical (Set2) vs continuous (gradient) color; axis_type: None/axis/arrow; legend_ondata at centroids
- `umap()`, `tsne()`, `pca()` — Wrappers calling dimensional() with preset dimensions param
- `expression(key)` — Like umap but validates key is a gene; defaults axis_type="arrow"
- `dimensionals(data, keys, ...)` — Grid: one dimensional plot per key
- `umaps()`, `tsnes()`, `pcas()` — Wrappers for dimensionals()
- `expressions(data, keys, ...)` — Grid of expression plots

## single/special/dotplot.py

- `dotplot(data, keys, group_by, ...)` — Marker expression across groups; data ops: unpivot → groupby → mean expression + % expressing; size ∝ percentage, color ∝ mean expression; sort_by support

## layers/

- `arrow_axis(plot, size, length, angle, color, ...)` — Replaces standard axes with custom arrows; extracts data/mapping from existing plot
- `_modify_axis()` — Internal: removes std axis elements, draws geom_segment arrows at data min/max
- `stream(plot, velocity_prefix, ...)` — Velocity field visualization; requires scvelo; computes streamlines on grid; returns FeatureSpecArray
- `cluster_outlines(plot, groups, padding, level, grid_size, ...)` — KDE + contour extraction (scipy + skimage) → geom_path outlines around clusters

## themes/

- `_THEME_DIMENSION` — 500x400, Arial, Set2 colors
- `_THEME_SCATTER` — classic + Set2 brewer
- `_THEME_DIST` — classic, for violin/boxplot
- `_THEME_DOTPLOT` — classic + border, custom axis text
- `_THEME_HEATMAP` — blank grid, custom text

## util/

- `get_mapping(plot, index=0)` — Extracts combined mapping as dict from a plot
- `retrieve(plot, index=0)` — Extracts underlying DataFrame from a plot
- `slice(grid, index)` — Slices SupPlotsSpec by index/indices (exported as `slice`)
- `_determine_axis(data, keys)` — Infers axis 0 vs 1 by key location
- `_is_variable_key()` / `_is_observation_key()` — Type checking helpers
- `_select_variable_keys(data, keys)` — Filters to only gene names
- `_color_gradient()` — 2-color or 3-color continuous scale; midpoint: "mean"/"median"/"mid"/numeric
- `_legend_ondata(frame, x, y, group_by, ...)` — geom_text labels at cluster centroids
- `show_colors()` — Displays the predefined color palette grid
- Color constants: TEAL, BLUE, RED, CHERRY, LIGHT_GRAY, SNOW, PURPLE, PINK, ORANGE
- Custom exceptions: ConflictingKeysError, KeyNotFoundError, ConfilictingLengthError

## complex/annotated_heatmap.py

- `annotated_heatmap(data, keys, group_by=None, column_annotations, row_annotations, annotation_colors, scale_axis, max_rows=1000, ...)` — gggrid composite (returns SupPlotsSpec). Rows=cells (optionally grouped by `group_by`), cols=`keys`; var-metadata tracks on top, obs-metadata tracks on LEFT. Numeric track->gradient, categorical->discrete; `annotation_colors[name]` overrides per track (dict=manual, list=gradient/palette). `max_rows` bins contiguous rows (mean numeric / mode categorical). Uses geom_tile (see [[project_letsplot_geom_raster_aspect]]). No `observations_name` param (row id is `position_y`). Also supports `dendrogram` (clusters groups via `_get_dendrogram`, reorders, draws a left panel), plus heatmap passthroughs `mapping`, `group_lines_kwargs`, `dendrogram_color/size/key/kwargs`, `interactive` (ggtb), `**geom_kwargs`. Excludes `group_bars*`/`aggregate`/`key_labels*`/`geom`/`markers` (contradict the composite). All vertical panels share explicit y-limits so dendrogram leaves align with group blocks.
- `_order_observations(frame, group_by, group_order=None)` — sorts rows into contiguous group blocks (explicit group_order e.g. from dendrogram, else first-appearance; group_by=None -> unchanged); returns (frame, group_order|None)
- `_bin_observations(frame, group_by, max_rows)` — caps rows; reuses heatmap's `_bin_within_groups` proportional per-group allocation formula (not shared code); numeric->mean, categorical->mode
- `_assign_position_y(frame)` — adds `position_y`, first row at top
- `_annotation_scale(series, colors)` — resolves a track's fill scale
- `_annotation_strip(frame, position, value, horizontal, legend, colors, position_limits)` — one-row/col track plot
- `_dendrogram_panel(paths, group_centers, n_groups, position_limits, color, size, kwargs)` — group dendrogram as a left vertical panel (geom_path; leaves at group centers, root extends left)
- `_blank_strip_theme(legend)` — void theme shared by tracks + dendrogram
- `_as_list(keys)` — str|Sequence|None -> list


---

## Source: project_cellestial_patterns.md

---
name: Cellestial code patterns and conventions
description: Naming conventions, parameter patterns, type annotations, singular/plural pairs, data transformation idioms
type: project
originSessionId: 60712229-021e-44be-be24-48d71e77fc8b
---
## Singular vs. plural function pairs

Every plot type has a singular (single PlotSpec) and plural (SupPlotsSpec grid) version:

| Singular | Plural | Notes |
|----------|--------|-------|
| violin | violins | keys: Sequence[str] |
| boxplot | boxplots | |
| dimensional | dimensionals | |
| umap | umaps | wraps dimensional |
| tsne | tsnes | |
| pca | pcas | |
| expression | expressions | genes only, axis_type="arrow" default |
| xyplot | xyplots | x/y broadcasting |

No plural versions for: scatter, bar, heatmap, dotplot, plot

## Parameter patterns

### Key/keys
```python
# Single: optional key
dimensional(data, key=None, ...)
# Plural: required Sequence
dimensionals(data, keys: Sequence[str], ...)
```

### Axis inference
- Usually automatic via `_determine_axis()` — checks if keys in var_names (axis=1) or obs.columns (axis=0)
- Explicit: `axis: Literal[0, 1]`

### Mapping merge pattern
```python
_mapping = aes(x=x, y=y)
merged = _mapping.as_dict() | mapping.as_dict()  # user mapping wins
```

### Tooltip parameter
```python
tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None
# None    → auto-generate from mapped keys
# "none"  → disable tooltips
# Sequence[str] → wrap in layer_tooltips()
# FeatureSpec   → use as-is
```

### Interactive parameter
```python
interactive: bool = False
# True → adds ggtb(size_zoomin=-1) for zoom
```

### Color/fill parameters
- Geom level: `geom_fill`, `geom_color`, `point_color`
- Scale level: `color_low`, `color_mid`, `color_high`, `mid_point`
- Discrete: `scale_color_brewer(palette="Set2")` — always Set2 for categoricals
- Continuous: `_color_gradient()` for expression/numeric data

### Default naming parameters
```python
observations_name: str = "Barcode"   # cell identifier column
variables_name: str = "Variable"      # gene identifier column
```

## Type annotation style
- Uses Python 3.10+ union syntax: `str | Sequence[str]`
- `TYPE_CHECKING` block for Lets-Plot type imports (avoid runtime circular)
- Literal types for constrained params: `Literal["umap", "pca", "tsne"]`
- Polars types used directly: `DataFrame`, `Series`

## Data transformation idioms (Polars)
- `.unpivot()` — wide → long format for multi-key distribution plots
- `.group_by().agg()` — aggregation for dotplot stats (mean expr, pct expressing)
- `.filter()`, `.with_columns()`, `.cast()` — standard transforms
- `.drop_nulls()` — clean after unpivot
- Categorical integer conversion in `build_frame()` — important for proper Polars typing

## Grid helper functions
- `_share_ticks()` — removes axis text on non-edge plots
- `_share_axis()` — removes axis elements except at edges
- `_share_labels()` — removes axis titles except at edges
- All applied before `gggrid()` call in plural functions

## Color logic
- **Categorical key** → `scale_color_brewer(palette="Set2")`
- **Continuous key** → `_color_gradient(low, mid, high, midpoint)`
  - 2-color: `scale_color_continuous(low, high)`
  - 3-color: `scale_color_gradient2(low, mid, high, midpoint)`

## Plot construction pattern
```python
plot = ggplot(frame, mapping) + geom_X(...) + scale + theme
if interactive:
    plot += ggtb(size_zoomin=-1)
# Optional layers added by user afterward with +
```

## Layer addition (deferred layer pattern)
- Layers (`arrow_axis`, `bracket`, `cluster_outlines`, `stream`) return a `DeferredLayer`
  (subclass of `PlotSpec` in `cellestial/layers/_deferred.py`) — they do **not** take
  `plot` positionally. The receiving plot is delivered by `__radd__` at `+` time.
- Primary usage: `umap + cl.arrow_axis()` (no `plot` arg needed).
- Advanced/escape-hatch: `ggplot() + cl.stream(plot=umap)` — the `plot` keyword-only
  argument lets callers pin an explicit data source when the receiving plot is
  different (e.g. drawing stream on an empty canvas beside the source).
- `DeferredLayer` subclasses `PlotSpec` so Python's reflected-operator priority
  rule fires `__radd__` before lets-plot's `PlotSpec.__add__` (which otherwise
  raises `TypeError` for unknown right operands — it does NOT return `NotImplemented`).


---

## Source: project_cellestial_tooling.md

---
name: Cellestial uses Poetry
description: Package/dependency management for cellestial is Poetry, not uv or pip
type: project
originSessionId: 4a2f596c-0e94-46e7-8439-87c79a139128
---
Cellestial uses **Poetry** for dependency management and running commands.

**Why:** User explicitly confirmed on 2026-04-13 after I suggested `uv` commands.

**How to apply:** Use `poetry add`, `poetry add --group dev`, `poetry run <cmd>`, `poetry install` — not `uv`, `pip install`, or `pip install -e .`. When suggesting dev tool installation or test commands, default to Poetry syntax.


---

## Source: feedback_explicit_list_extend.md

---
name: feedback-explicit-list-extend
description: Prefer explicit `list(x); x.extend(...)` over unpacking `[*a, *b]` when combining a user-supplied sequence with extra items
type: feedback
---

When combining a user-supplied sequence (possibly None) with additional items, prefer the explicit two-line form:

```python
variable_keys = list(variable_keys)
variable_keys.extend(_select_variable_keys(data, keys))
```

Do NOT collapse into `variable_keys = [*(variable_keys or []), *_select_variable_keys(data, keys)]`.

**Why:** User finds the splat-unpacking form ugly. The explicit form reads top-to-bottom, mirrors patterns already used elsewhere in the codebase (see `cellestial/spatial/spatial.py` around the `variable_keys` handling), and the variable's role is clearer.

**How to apply:** Whenever you reach for `[*a, *b]` to merge a sequence parameter with computed items in this codebase, write `x = list(x); x.extend(...)` instead.


---

## Source: feedback_audit_ai_tracking.md

---
name: Record AI-written functions in audit_AI.md
description: Whenever an agent generates or significantly modifies a function, add a row for it to plans/audit_AI.md
metadata:
  type: feedback
---
Whenever an agent (Claude or any other model) writes a new function or modifies a significant portion of an existing one in cellestial, add a row for that function to `plans/audit_AI.md`.

**Why:** The user tracks every AI-touched function there so each can later be verified (behavioral, visual, edge cases, line-by-line review). Functions that get AI-written but never logged silently escape that review pass. The audit table replaced the old inline `# AI-GENERATED:` / `# AI-MODIFIED:` source comments (removed in commit "replace AI notice"), so the markdown table is now the single source of truth.

**How to apply:** The table columns are `Function | AI status | Source | Behavioral | Visual | Edge cases | Line-by-line`.
- `AI status`: `GENERATED` for brand-new functions, `MODIFIED` for significant edits to existing ones.
- `Source`: the model that did the work (e.g. `Claude 4.7`, `Claude 4.6`, `Mixed`).
- Leave the verification columns (`Behavioral`, `Visual`, `Edge cases`, `Line-by-line`) blank when adding the row; the user fills `✓` after they review. Do not mark them verified yourself unless you actually ran/verified that check.
Place the new row near related functions (e.g. singular/plural pairs and their helpers together). Key private helpers count too (the table already lists `_spatial_components`, `_get_dendrogram`, `_build_markers_frame`, etc.).


---

## Source: project_polars_drop_null_gotcha.md

---
name: project_polars_drop_null_gotcha
description: Polars exclude/drop filters must use `.fill_null(False)` or they silently drop null rows too
type: project
---

When implementing a `drop`/exclude filter on a categorical column in polars, use
`frame.filter(~pl.col(col).is_in(values).fill_null(False))`, NOT plain
`~pl.col(col).is_in(values)`.

`is_in` returns `null` (not `False`) for null entries, and `filter` drops null
rows. So `~col.is_in([...])` removes unannotated/`None`-category rows in addition
to the requested groups. `.fill_null(False)` keeps the null rows.

This is how the `drop` parameter is implemented across the plotting functions
(spatial, dimensional/_distribution/ridge). The positive `groups` keep-filter
does not need this since dropping nulls there is acceptable.

## Source: feedback_changelog_breaking_changes.md

---
name: feedback_changelog_breaking_changes
description: Record breaking API changes in CHANGELOG.md under the current poetry version
type: feedback
---

When making a breaking API change (renaming/removing/repurposing a public
parameter or function), record it in the repo-root `CHANGELOG.md` under a
`### Breaking` heading, with a one-line migration note (e.g. "replace `old=` with
`new=`). Keep a Changelog style; non-breaking fixes go under `### Fixed`.

**Why:** The user tracks breaking changes by version so users can migrate; this
matters more as the project approaches v1.0.

**How to apply:** Run `poetry version` to get the current version and date the
section `## [X.Y.Z] - YYYY-MM-DD`. If the version hasn't been bumped yet, put
entries under `## [Unreleased]` and rename the heading once bumped. Related:
feedback_audit_ai_tracking.

---

## Source: project_letsplot_geom_raster_aspect.md

---
name: project_letsplot_geom_raster_aspect
description: lets-plot geom_raster locks square pixels (data aspect); heatmaps must dodge it
metadata:
  type: project
---

In lets-plot (4.10), `geom_raster` renders with square pixels: the panel aspect
is forced to the data dimensions (n_x : n_y). For a heatmap with few columns and
many rows this collapses to a thin line. Two ways the repo handles it:

- `cl.heatmap` (single ggplot): rescales per-cell `position_y` to span `[0, n_x-1]`
  (same range as x) in `_assign_positions` (`y_step = (n_x-1)/(n_y-1)`), so raster's
  square pixels yield a square-ish plot regardless of cell count.
- `cl.annotated_heatmap` (in `cellestial/complex`, a `gggrid` composite): now uses
  `geom_raster` + `coord_cartesian()`. `coord_cartesian()` FREES raster's forced
  square-pixel aspect (without it the few-column many-row heatmap collapses to a thin
  sliver, verified), so the grid widths/heights control the shape. This is a simpler
  dodge than `cl.heatmap`'s y-rescaling. Annotation strips still use `geom_tile`.

Composite alignment relies on `gggrid(align=True)` (aligns panel areas across the
grid despite differing axis labels) and `guides="collect"` (gathers each track's
legend). `aes()` objects do not combine with `+`; build one `aes(...)` per geom.
See [[project_cellestial_functions]] and [[project_annotated_heatmap_overflow]].

---

## Source: project_annotated_heatmap_overflow.md

---
name: project_annotated_heatmap_overflow
description: cl.annotated_heatmap layout fixes (dendrogram clip, legend overflow) and why bar-end track labels were reverted
metadata:
  type: project
---

`cl.annotated_heatmap` (`cellestial/complex/annotated_heatmap.py`, a `gggrid`
composite) layout notes:

- **Dendrogram clip (fixed)**: the root segment sat flush on the panel/canvas edge.
  Fixed by giving `_dendrogram_panel`'s `scale_x_continuous` a small left margin
  (`expand=[0.05, 0]`). ggsize does NOT fix this (it only rescales proportionally).
- **Legend overflow (fixed)**: collected legends stacked into one tall right-side
  column that clipped off the bottom. Fixed by moving them to one bottom row.
  KEY lets-plot 4.10 gotcha for `gggrid(guides="collect")`: the grid theme controls
  ONLY legend PLACEMENT/arrangement (`legend_position`, `legend_box`,
  `legend_direction`); legend ELEMENT properties (`legend_position` per subplot,
  `legend_key_size`, `legend_text`, `legend_title` sizes) must be set on the SUBPLOTS
  that own each legend, or they are ignored. So: subplots (heatmap +
  `_blank_strip_theme`) carry `legend_position="bottom"` + `_LEGEND_THEME`
  (key_size=10, text size 9, title size 10); the grid carries only
  `theme(legend_position="bottom", legend_box="horizontal", legend_direction="horizontal")`.
  Other facts: collected legends arrange as ONE row (`legend_box="horizontal"`) or ONE
  column (`"vertical"`) only, no N-per-row / ncol / wrapping (verified; manual
  carrier-plot hack looks bad). No legend-title-position param either, but
  `legend_direction="vertical"` puts each title above its keys.
- Each annotation track gets a distinct ColorBrewer palette cycled from
  `_BREWER_PALETTES` via `scale_fill_brewer` (categorical tracks only; numeric keep
  the gray gradient). Palette order is qualitative-first (Spectral, Accent, Dark2,
  Paired, Pastel1/2, Set1/2/3) then diverging. Strip x-axis lines removed via
  `axis_line=element_blank()`.
- **Bold legend titles**: a legend title's `face` is inherited from the parent
  `title` element, NOT from `legend_title` (setting `face` on `legend_title` does
  nothing). So `_LEGEND_THEME` sets `title=element_text(face="bold")` (safe because
  axis titles are blanked). This bolds the categorical track titles; the continuous
  colorbar ("value") title does NOT bold under gggrid collection (lets-plot quirk,
  unresolved).
- **`layers` param**: `annotated_heatmap(..., layers=...)` adds extra lets-plot
  layers to the heatmap; accepts a single `FeatureSpec` OR a sequence of them. A
  `scale_fill_*` like `scale_fill_viridis()` overrides the default value gradient
  (added after it, last wins, no warning).
- **Heatmap geom is `geom_raster` + `coord_cartesian()`** (was `geom_tile`). Raster
  is fast for many cells; `coord_cartesian()` is required to free raster's square-
  pixel aspect lock (else the heatmap collapses to a sliver). See
  [[project_letsplot_geom_raster_aspect]].

**Bar-end track labels were tried and REVERTED.** The idea: dedicated label lanes
in the grid (top lane for row-track names rotated vertical, right lane for
column-track names). Column (horizontal) labels worked. The vertical row labels did
not: they live in the thin row-annotation columns (`row_annotation_width` ~0.025) and
the LEFTMOST row-label cell collapses to near-zero width and clips the text to
slivers, independent of `align` (False too) and of text length (swapping order moves
the clip to whichever name is leftmost). A real fix needs the labels decoupled from
the thin strip columns (e.g. one spanning label panel), not per-strip cells.

See [[project_letsplot_geom_raster_aspect]].
