# AGENTS.md

Read this file fully before touching the code. It is the contract for every agent
(human or AI) working on **cellestial**. Design philosophy lives in
[philosophy.md](philosophy.md). Plans and audits live in [plans/](plans/).
Lessons learned live in [MEMORY.md](MEMORY.md).

## Vision

cellestial is an interactive single-cell and spatial omics visualization library,
built on the grammar of graphics.

- Inputs: AnnData, MuData and SpatialData. Plots return lets-plot `PlotSpec`
  (singular functions) or `SupPlotsSpec` (plural functions). Never both.
- Backend: lets-plot today. lets-plot is no longer maintained, so
  [ggrust](https://github.com/datavil/ggrust) is being built as its replacement with a
  lets_plot-compatible API. Porting should stay (almost) an import swap: use the
  lets_plot API surface ggrust supports (`aes`, `ggplot`, `geom_*`, `scale_*`, `theme`,
  `gggrid`, `ggtb`, `+` composition). When cellestial needs something ggrust lacks,
  record it as a request in ggrust's `plan.md` instead of working around it silently.
- Polars does the data work. AnnData internals stay inside helpers.
- Users are biologists in notebooks: explicit names, predictable return types,
  composable with `+` (see philosophy.md: modularity, predictability, explicitness).

## Golden rules

1. **Think before coding.** State assumptions. If a request has more than one
   reading, say so and ask. Do not pick one silently. Push back if a simpler way exists.
2. **Simplicity first.** Minimum code that solves the problem. No speculative features,
   no single-use abstractions, no configurability nobody asked for. If 200 lines could
   be 50, rewrite it.
3. **Be surgical.** Every changed line traces to the request. Do not "improve" adjacent
   code, comments or formatting. Mention unrelated dead code, do not delete it. Remove
   only the orphans your own change created. If you must touch anything outside the
   requested part, **ask the user first**.
4. **Goal-driven.** Turn fuzzy tasks into verifiable targets ("fix the bug" -> "write a
   test that reproduces it, then make it pass"). For multi-step tasks state a short
   plan with a check per step.
5. **Test everything before handing over. Not just smoke tests.** Run the real code
   against real data (a dataset from `cellestial.datasets`) and the test suite.
6. **Verify visuals.** Render the plot to PNG and look at it (Read the image). Check
   that labels, legends and geoms stay inside the plot area.
7. **Bug-hunting sub agents.** After a non-trivial change, spin up a sub agent (cheap
   model) whose job is to assume the code is faulty and find the bug. Fix what it
   confirms, surgically. Re-verify.
8. **Record what you did.** Breaking API changes go in `CHANGELOG.md` under the current
   version (`### Breaking` + migration note). Functions an agent wrote or significantly
   changed get a row in `plans/audit_AI.md` (leave the verification columns blank for
   the user). Lasting lessons go in `MEMORY.md`.
9. **Commit each significant step, then push.** Run `ruff format` and `ruff check --fix`
   first. Tests must pass at each commit.
10. Robustness over flexibility. Ask the user when unsure on design decisions.

## Naming and style

- Full words, no abbreviations: `group_index` not `grp_idx`, `frame` not `df`,
  `include_dimensions` not `inc_dims`. Well-known terms are fine (`n`, `X`, `umap`, `pca`).
- Singular/plural pairs: `cl.umap()` takes one key and returns `PlotSpec`;
  `cl.umaps()` takes a sequence and returns `SupPlotsSpec`. Every function returns
  one type.
- No sizing parameters (`width`, `height`) on plot functions: users add `+ ggsize()`.
- No hidden aesthetic promotion: do not auto-map `group_by` to `fill`/`color`. Users
  wire aesthetics explicitly.
- Keep `color`/`fill` as column mappings and `geom_color`/`geom_fill` as constants.
- Polars over pandas. Use LazyFrames where they pay off. Exclude filters need
  `~col.is_in(values).fill_null(False)` (otherwise null rows vanish).
- Python: ruff formatting and import sorting.
- Type hints on every public and private function, parameter and return value, with
  precise types (`Literal[...]`, `X | None`, typed containers). Checked with `ty`.
- NumPy docstrings on every public function: one-line summary, then `Parameters`,
  `Returns`, `Raises`, `Examples`. Extra context goes in `Notes`, not floating prose.
  Single backticks for identifiers. Never name AnnData slots (`adata.obs`, `uns`) or
  scanpy APIs in user-facing docs.
- No em dashes anywhere: code, docs, comments, chat.

## Backend-agnostic plot bodies

Plot functions and their error messages never speak AnnData. AnnData-, MuData- or
SpatialData-specific work lives in helpers under a positive branch:

```python
if isinstance(data, AnnData):
    ...  # backend-specific ops here
```

Keep the positive branch. Do not flatten it to `if not isinstance(...): raise`, it is
where future backends plug in. MuData routes through `frames/_container.py`.

## Errors are first-class

- Every error says **what** was wrong, **where** (which parameter, key, layer),
  **what is accepted**, and **how to fix it** (a close match for typos:
  `key 'leidn' not found; did you mean 'leiden'?`).
- Use the specific classes in `cellestial/util/errors.py` (they subclass the matching
  builtin). Add a new class with a docstring when none fits; no bare `ValueError`.
- Fail early, before building the plot.
- Tests assert on the error type and the key parts of the message (`tests/test_errors.py`).

## Architecture

```
cellestial/
  frames/      build a polars frame from AnnData/MuData (build.py, _container.py)
  single/      one plot per call and plural grid versions
    core/          dimensional (umap, pca, ...), distribution (violin, boxplot, ...)
    basic/         scatter, bar
    common/        xyplot(s)
    heatmap/       heatmap, dotplot, stacked_violin
    differential/  markers, volcano
  complex/     multi-panel figures (annotated_heatmap)
  layers/      `+` composable layers (arrow, bracket, outline, stream, ondata_legend)
  spatial/     SpatialData plots
  themes/      theme presets
  util/        errors, shared helpers
  datasets/    bundled example datasets
tests/         pytest (browser tests behind the `browser` marker)
SKILLS/        how-to notes for recurring plotting patterns (read before similar work)
plans/         roadmap, TODO, feature gaps, audits
```

Adding a plot = one file in the right `single/` subpackage + export in `__init__.py`
+ tests. If it needs edits in many unrelated files, stop and tell the user.

## Cost discipline

- **Read narrowly.** Search before reading. Read the line range you need.
- **Test narrowly while iterating** (`poetry run pytest -q -k <name>`). Run the full
  suite once before each commit.
- **Keep output short.** Pipe long output through `tail`/`grep`. Never dump full
  plot specs or HTML into context.
- **Cheap sub agents for cheap work** (searches, mechanical checks).
- **No open-ended loops.** If a fix fails twice the same way, stop and ask.

## Workflow per change

1. Read the relevant `plans/` file, `MEMORY.md` entries and `SKILLS/` notes.
2. Check the lets-plot behavior you rely on (and that ggrust supports it).
3. Implement minimally, with tests.
4. Render, look, fix.
5. Bug-hunting sub agent. Fix confirmed bugs.
6. Update `CHANGELOG.md` / `plans/audit_AI.md` if applicable.
7. Format, test, commit, push.

## Commands

Package management is **Poetry** (not uv, not pip).

```bash
poetry install                                    # dev env
poetry run pytest                                 # tests (parallel, excludes browser)
poetry run pytest tests/test_browser_plots.py -m browser -n 2   # browser tests
poetry run ruff format cellestial && poetry run ruff check --fix cellestial
poetry run ty check cellestial                    # type check
./scripts/release-check.sh                        # everything, before a release
```
