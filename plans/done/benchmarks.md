# Benchmarks: cellestial vs scanpy.pl

Head-to-head, end-to-end timing of cellestial plot functions against their `sc.pl.*`
counterparts. Two measured phases (construction, rendering), three replicas per case,
results as feather, comparison figures as SVG via lets-plot `theme_bw()`.

Skip plots with no scanpy counterpart. Skip atlas-scale datasets.

Runnable via `python -m benchmarks`.

---

## 1. Layout

```
benchmarks/
  __init__.py
  __main__.py         # forwards to cli.main()
  cli.py              # argparse: --datasets, --cases, --replicas, --output, --skip-render
  datasets.py         # dataset registry (lazy loaders + cached metadata)
  cases.py            # case registry: (name, cl_fn, sc_fn, params_grid, key_kind)
  runner.py           # timing harness: warmup + 3 replicas, construct & render phases
  io.py               # feather writer; SVG writer
  figures.py          # build comparison figures from results.feather
  results/            # *.feather (one consolidated file + per-run snapshots)
  figures/            # *.svg comparison plots
  svg/                # raw SVG outputs per case for equality / visual check
    <case>/<lib>/replica_{0,1,2}.svg
```

No `[project.scripts]` entry — runs via `poetry run python -m benchmarks`.

---

## 2. Pairs (cellestial ↔ scanpy.pl)

Only pairs with a real counterpart are benchmarked. The rest (`ridge`, `ridges`,
`volcano`, `volcanos`, `bar`, `stream`, `arrow_axis`, `cluster_outlines`,
`elbow` vs `pca_variance_ratio` — keep this one) are skipped.

| Case id              | cellestial                    | scanpy                          | Key kind        |
| -------------------- | ----------------------------- | ------------------------------- | --------------- |
| `umap_gene`          | `umap(adata, key=gene)`       | `sc.pl.umap(color=gene)`        | gene (n=1)      |
| `umap_category`      | `umap(adata, key=cluster)`    | `sc.pl.umap(color=cluster)`     | categorical (n=1) |
| `tsne_gene`          | `tsne(adata, key=gene)`       | `sc.pl.tsne(color=gene)`        | gene (n=1)      |
| `tsne_category`      | `tsne(adata, key=cluster)`    | `sc.pl.tsne(color=cluster)`     | categorical (n=1) |
| `pca_gene`           | `pca(adata, key=gene)`        | `sc.pl.pca(color=gene)`         | gene (n=1)      |
| `pca_category`       | `pca(adata, key=cluster)`     | `sc.pl.pca(color=cluster)`      | categorical (n=1) |
| `expression`         | `expression(adata, key=gene)` | `sc.pl.umap(color=gene)`        | gene (n=1)      |
| `dotplot`            | `dotplot(adata, keys, group_by)` | `sc.pl.dotplot(adata, keys, group_by)` | n_genes |
| `heatmap`            | `heatmap(adata, keys, group_by)` | `sc.pl.heatmap(adata, keys, group_by)` | n_genes |
| `matrixplot`         | `matrixplot(adata, keys, group_by)` | `sc.pl.matrixplot(adata, keys, group_by)` | n_genes |
| `stacked_violin`     | `stacked_violin(adata, keys, group_by)` | `sc.pl.stacked_violin(adata, keys, group_by)` | n_genes |
| `violin`             | `violin(adata, key)`          | `sc.pl.violin(adata, key)`      | n=1             |
| `violins`            | `violins(adata, keys)`        | `sc.pl.violin(adata, keys)`     | n_keys          |
| `highest_expressed`  | `highest_expressed_genes(adata, n_top)` | `sc.pl.highest_expr_genes(adata, n_top)` | n_top |
| `markers`            | `markers(adata, n_genes)`     | `sc.pl.rank_genes_groups(adata, n_genes)` | n_genes |
| `scatter`            | `scatter(adata, x, y)`        | `sc.pl.scatter(adata, x, y)`    | n=1             |
| `xyplot`             | `xyplot(adata, x=gene, y=gene)` | `sc.pl.scatter(adata, x=gene, y=gene)` | n=1 |
| `spatial`            | `spatial(adata, key=gene)`    | `sc.pl.spatial(adata, color=gene)` | n=1          |

Grid variants (`umaps`, `tsnes`, `expressions`, `boxplots`, etc.) where the
scanpy equivalent is "pass a list" are folded into the underlying single case
with `n_keys > 1` rather than getting their own row.

---

## 3. Datasets

Lazy registry — only loaded if a case references them. All cached under
`benchmarks/data/` (or the user's existing `~/.cache/cellestial/datasets`).

| Dataset key            | Loader                                              | Source     | Approx (cells × genes) | Use for                 |
| ---------------------- | --------------------------------------------------- | ---------- | ---------------------- | ----------------------- |
| `pbmc3k`               | `cl.datasets.pbmc3k()`                              | cellestial | ~7k × ~17k             | full grid (scRNA)       |
| `pancreas`             | `cl.datasets.pancreas()`                            | cellestial | ~3.7k × ~2k            | full grid (scRNA, velocity) |
| `pbmc68k_reduced`      | `sc.datasets.pbmc68k_reduced()`                     | scanpy     | ~700 × ~765            | small/fast              |
| `paul15`               | `sc.datasets.paul15()`                              | scanpy     | ~2.7k × ~3.5k          | small/fast              |
| `human_lymph_node`     | `cl.datasets.human_lymph_node()`                    | cellestial | ~4k spots (Visium)     | spatial pairs           |
| `visium_hne`           | `sq.datasets.visium_hne_adata()`                    | squidpy    | ~2.7k spots            | spatial pairs           |

Excluded: `breast_cancer_atlas`, any explicit atlas. Spatial cases run only
against spatial datasets; everything else runs on scRNA-seq datasets.

A small helper precomputes per-dataset metadata once (n_cells, n_vars, default
`group_by` column, a fixed list of marker genes drawn from `highly_variable_genes`
top-N) so cases pull deterministic inputs.

---

## 4. Scenarios (parameter grids)

Day-to-day → high-coverage. `n_keys` lists are capped to available HVGs per
dataset.

- **Dim reduction** (`umap_*`, `tsne_*`, `pca_*`, `expression`): n_keys = 1.
  Two variants per cell: `key_kind="gene"` and `key_kind="category"`.
- **Heatmap variants** (`dotplot`, `heatmap`, `matrixplot`, `stacked_violin`,
  `markers`): n_keys ∈ {5, 20, 50, 100, 200}.
- **Violins**: `violin` n=1; `violins` n_keys ∈ {1, 5, 20}.
- **highest_expressed**: n_top ∈ {10, 30, 60, 100}.
- **scatter / xyplot**: n=1 (gene-vs-gene and obs-vs-obs once each).
- **spatial**: n_keys ∈ {1, 5, 20} (gene channel).

Cases run on every dataset where they are well-defined (skipped silently with
status="skip" if a dataset lacks the required keys / group_by / spatial info).

---

## 5. Timing harness

`lets_plot.LetsPlot.setup_html(no_js=True, isolated_frame=True)` once at import
so lets-plot stays static. SVG export uses `lets_plot.export.ggsave(plot, path,
iframe=False)`. Matplotlib uses `Agg` backend (`matplotlib.use("Agg")`) so no
display is opened.

For every (case × dataset × params) tuple:

1. Resolve concrete keys/group_by from the dataset metadata.
2. Warmup: one full construct+render, result discarded.
3. Replicas 0..2:
   - `t0 = perf_counter()` → call the **construct** path → `t1`.
     - cellestial: `plot = cl.<fn>(adata, ...)` returning a `PlotSpec`.
     - scanpy: `fig = sc.pl.<fn>(adata, ..., show=False, return_fig=True)`.
       Where `return_fig` is unsupported, fall back to `ax = sc.pl.<fn>(..., show=False)`
       and grab `ax.figure`; note in README that scanpy's build phase includes some
       layout work that lets-plot defers to render.
   - `t2 = perf_counter()` → **render** → SVG file → `t3`.
     - cellestial: `ggsave(plot, "replica_N.svg", path=svg_dir, iframe=False)`.
     - scanpy: `fig.savefig(svg_dir / "replica_N.svg", format="svg", bbox_inches="tight")`.
     - close the matplotlib figure (`plt.close(fig)`).
   - Record `construct = t1 - t0`, `render = t3 - t2`, `total = construct + render`.

Failures: wrap each phase in try/except; record `status="error:<message>"`,
seconds = NaN, continue.

---

## 6. Results schema (feather)

One row per (case, dataset, params, library, replica, phase). Polars frame,
written to `benchmarks/results/results.feather` (atomic write via `.part` rename).
Each invocation also writes a timestamped snapshot
`benchmarks/results/runs/run_<UTC>.feather`.

| Column        | Type     | Notes                                              |
| ------------- | -------- | -------------------------------------------------- |
| `case`        | `str`    | e.g. `dotplot`                                     |
| `library`     | `str`    | `cellestial` or `scanpy`                           |
| `dataset`     | `str`    | dataset registry key                               |
| `n_cells`     | `i64`    | rows of the adata used                             |
| `n_keys`      | `i64`    | how many genes/keys plotted (1 for dim-reduction)  |
| `key_kind`    | `str`    | `gene`, `category`, `obs_obs`, `gene_gene`, `spatial`, `""` |
| `phase`       | `str`    | `construct` or `render`                            |
| `replica`     | `i64`    | 0..2                                               |
| `seconds`     | `f64`    | wall time                                          |
| `svg_bytes`   | `i64`    | size of generated SVG (0 for construct rows)       |
| `status`      | `str`    | `ok`, `skip:<reason>`, `error:<class>`             |
| `started_at`  | `datetime[μs, UTC]` | start timestamp                          |

---

## 7. SVG outputs (equality)

Every render writes a real SVG. Layout:

```
benchmarks/svg/<case>/<dataset>/<param_slug>/<library>/replica_{0,1,2}.svg
```

Both libraries write into sibling directories so a human can spot-check by
diffing folders. The runner records `svg_bytes`; a separate check just asserts
both files exist and are non-empty — full pixel-level equality is out of scope.

---

## 8. Figures (lets-plot, `theme_bw()`)

`figures.py` reads `results/results.feather` and produces:

1. **`scaling_by_data_size.svg`** — faceted by `case`, x = `n_cells`,
   y = mean `seconds`, color = `library`, linetype/dodge = `phase`. Shows how
   wall time grows with dataset size at a fixed `n_keys` (use the default of
   each case).
2. **`scaling_by_key_count.svg`** — faceted by `case` (only the heatmap-variant
   cases that vary `n_keys`), x = `n_keys`, y = mean `seconds`, color = `library`,
   linetype = `phase`, facet row = `dataset`.
3. **`construct_vs_render.svg`** — stacked-bar per (case, library) at default
   params on `pbmc3k`, x = case, y = mean `seconds`, fill = `phase`.
4. **`overall_summary.svg`** — single comparison: total time across all cases,
   bar dodged by library, with error bars from 3 replicas (sd).

All figures use `theme_bw()` and the project's existing color tokens
(`TEAL`, `CHERRY`) for `cellestial` vs `scanpy`. Polars groupby/agg for stats.

---

## 9. CLI

```
poetry run python -m benchmarks               # everything
poetry run python -m benchmarks --datasets pbmc3k pancreas
poetry run python -m benchmarks --cases dotplot heatmap
poetry run python -m benchmarks --replicas 5
poetry run python -m benchmarks --skip-render # construct-only (fast smoke)
poetry run python -m benchmarks --only-figures # rebuild figures from existing feather
```

Sensible defaults: all cases, all datasets, replicas=3. Progress via `tqdm`
over the (case × dataset × params) cartesian product. Datasets are loaded
once and reused across all cases that need them.

---

## 10. Implementation order

1. Scaffold `benchmarks/` package + `__main__.py` no-op CLI.
2. `datasets.py` — lazy loaders, metadata cache (n_cells, default `group_by`,
   marker gene list). Verify each dataset loads under `poetry run python -m benchmarks --check`.
3. `cases.py` — case registry mapping ids to `(cl_fn, sc_fn, params_grid, key_kind)`.
4. `runner.py` — timing harness, warmup, replicas, SVG export, status capture.
5. `io.py` — feather + SVG path resolution.
6. End-to-end smoke: `--cases umap_gene --datasets pbmc68k_reduced --replicas 1`,
   verify both SVGs render and feather row is written.
7. `figures.py` — build the four figures from `results.feather`.
8. Full run on `pbmc68k_reduced` + `paul15` to validate scaling shapes before
   running on the larger datasets.
9. README in `benchmarks/` documenting the construct/render caveat for scanpy
   and how to interpret the figures.

---

## 11. Open risks / notes

- **scanpy phase split is not perfect.** `sc.pl.dotplot/matrixplot/stacked_violin`
  build a `DotPlot`/`MatrixPlot`/`StackedViolin` object before rendering; for
  these we can call `obj = sc.pl.DotPlot(...)` then `obj.make_figure()` to get
  a cleaner split. For `sc.pl.umap` etc. the construct phase is inherently
  "build the matplotlib Figure" — note this in the README.
- **`sc.pl.spatial` and lets-plot spatial rendering speeds are not directly
  comparable** (raster vs vector). Record both, mention the caveat.
- **lets-plot SVG export requires a JVM**; cellestial already depends on
  `lets-plot`, so this should be fine, but the harness should fail loudly if
  `ggsave` errors.
- **Dataset load times are excluded from timings** — loads happen before
  `perf_counter()` starts; only plot calls are measured.
- **`highest_expressed_genes` was recently rewritten** (commit `e015f7b`,
  5–7× faster, 100× less memory) — benchmark should capture this clearly.
