

## MEMORY

# Memory Index

- [Project: Cellestial architecture & design](project_cellestial_architecture.md) — Single-cell viz library built on Lets-Plot + AnnData; layered modular design, Polars-first, grammar of graphics
- [Project: Cellestial function inventory](project_cellestial_functions.md) — What every function does: frames, single plots, layers, util, themes
- [Project: Cellestial tooling](project_cellestial_tooling.md) — Uses Poetry (not uv/pip) for deps and running commands
- [Project: Cellestial code patterns](project_cellestial_patterns.md) — Naming conventions, parameter patterns, singular/plural pairs, type annotations, data transformation patterns
- [Feedback: Test code changes](feedback_test_code.md) — Run actual test code against data/pbmc3k_pped.h5ad before reporting success
- [Feedback: Avoid shortened variable names](feedback_naming.md) — No abbreviations like `grp_idx`/`obs`/`cfg`; use full words (only `n`-style counts OK)
- [Feedback: Docs avoid AnnData internals](feedback_docs_no_anndata_internals.md) — Don't reference `uns['spatial']`/`obsm[...]`/`obs[...]` in user-facing docstrings; use abstracted terms
- [Feedback: Never use em dashes](feedback_no_em_dashes.md) — User hates em dashes; never write them in code, docs, comments, or chat
- [Feedback: `if isinstance(data, AnnData)` block stays](feedback_isinstance_anndata_block.md) — Never flatten to `if not isinstance: raise`; positive branch hosts AnnData-specific ops for future multi-backend support
- [Feedback: No hidden aesthetic promotion](feedback_no_hidden_aesthetic_promotion.md) — Don't auto-promote `group_by` to `fill`/`color`; user wires aesthetics explicitly, prefer warn+facet over hidden magic


---

## Source: feedback_docs_no_anndata_internals.md

---
name: Docs avoid AnnData-internal references
description: Don't reference AnnData container slots like uns['spatial'] / obsm['X'] in user-facing docs and docstrings
type: feedback
originSessionId: 38fa3aa3-d285-4124-9b4f-be3c011e598b
---
Never name AnnData-internal structure (e.g. `uns['spatial']`, `obsm['spatial']`, `obs['cluster']`, `var_names`, `layers['counts']`) in user-facing docstrings, Notes blocks, or error messages aimed at end users. Use abstracted language: "image data", "spot coordinates", "spatial coordinates", "the dataset", etc.

**Why:** Cellestial aims to abstract over the data container (AnnData today, possibly others later — the architecture already accepts non-AnnData per `plans/TODO.md`). Surfacing AnnData internals in docs leaks the implementation, dates the docs, and confuses users on alternative backends.

**How to apply:** When documenting parameters, behavior, or notes, describe what the input *represents* (image, coordinates, gene expression) rather than where it lives in the AnnData object. Internal code comments referencing `uns/obsm/etc.` are fine — this rule is for docstrings and user-visible text only.


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

