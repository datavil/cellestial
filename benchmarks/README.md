# Cellestial vs scanpy.pl benchmarks

End-to-end timing of every cellestial plot function against its `scanpy.pl`
counterpart. Splits each measurement into a *construct* phase (function call)
and a *render* phase (SVG export). Output is a polars feather file plus four
lets-plot comparison figures, all using `theme_bw()`.

## Run it

```
poetry run python -m benchmarks                          # everything
poetry run python -m benchmarks --datasets pbmc68k_reduced
poetry run python -m benchmarks --cases umap dotplot
poetry run python -m benchmarks --replicas 5
poetry run python -m benchmarks --skip-render            # construct-only smoke
poetry run python -m benchmarks --only-figures           # rebuild figures from existing feather
poetry run python -m benchmarks --check                  # load datasets only, report metadata
poetry run python -m benchmarks --list                   # list available datasets and cases
```

On Apple-silicon installs where Numba JIT triggers segfaults inside scvelo,
prepend `NUMBA_DISABLE_JIT=1`.

Set `BENCHMARKS_DEBUG=1` to print full tracebacks for failing cases. Without
it, errors are reported as a single status line per replica.

## Output layout

```
benchmarks/
  results/
    results.feather               # consolidated across runs (appended)
    runs/run_<timestamp>.feather  # per-invocation snapshot
  figures/
    scaling_by_data_size.svg
    scaling_by_key_count.svg
    construct_vs_render.svg
    overall_summary.svg
  svg/
    <case>/<dataset>/<param_slug>/<library>/replica_{0,1,2}.svg
```

## Cases

Each entry pairs a cellestial function with its scanpy counterpart and a
parameter grid. Plots with no scanpy counterpart (`ridge`, `volcano`, `bar`,
`stream`, `arrow_axis`, `cluster_outlines`) are not included.

| case               | cellestial                  | scanpy                       | key axis        |
| ------------------ | --------------------------- | ---------------------------- | --------------- |
| `umap` / `tsne` / `pca` | `cl.umap/tsne/pca`     | `sc.pl.umap/tsne/pca`        | n=1, gene + category |
| `expression`       | `cl.expression`             | `sc.pl.umap`                 | n=1, gene       |
| `dotplot`          | `cl.dotplot`                | `sc.pl.dotplot`              | n ∈ {5,20,50,100,200} |
| `heatmap`          | `cl.heatmap`                | `sc.pl.heatmap`              | n ∈ {5,20,50,100,200} |
| `matrixplot`       | `cl.matrixplot`             | `sc.pl.matrixplot`           | n ∈ {5,20,50,100,200} |
| `stacked_violin`   | `cl.stacked_violin`         | `sc.pl.stacked_violin`       | n ∈ {5,20,50,100} |
| `violin`           | `cl.violin`                 | `sc.pl.violin`               | n=1, gene       |
| `highest_expressed`| `cl.highest_expressed_genes`| `sc.pl.highest_expr_genes`   | n_top ∈ {10,30,60,100} |
| `markers`          | `cl.markers`                | `sc.pl.rank_genes_groups`    | n ∈ {5,20,50}   |
| `scatter_obs_obs`  | `cl.scatter`                | `sc.pl.scatter`              | n=1, obs vs obs |
| `xyplot_gene_gene` | `cl.xyplot`                 | `sc.pl.scatter`              | n=1, gene vs gene |
| `spatial`          | `cl.spatial`                | `sc.pl.spatial`              | n ∈ {1,5,20}    |

## Datasets

Cached on first use under `data/` (or the user's existing
`~/.cache/cellestial/datasets/`). No atlas-scale datasets.

| key                | kind     | source              |
| ------------------ | -------- | ------------------- |
| `pbmc3k`           | scrna    | `cl.datasets.pbmc3k` |
| `pancreas`         | scrna    | `cl.datasets.pancreas` |
| `pbmc68k_reduced`  | scrna    | `sc.datasets.pbmc68k_reduced` |
| `paul15`           | scrna    | `sc.datasets.paul15` (normalize + UMAP + clusters + rank_genes_groups added inline) |
| `human_lymph_node` | spatial  | `cl.datasets.human_lymph_node` |
| `visium_hne`       | spatial  | `sq.datasets.visium_hne_adata` |

Spatial cases run only on spatial datasets and vice versa.

## Methodology

For every `(case, dataset, params, library, replica)` tuple the runner:

1. Performs one un-recorded warmup pass per library (skippable with `--no-warmup`).
2. Measures `construct = perf_counter()` while calling the library function.
3. Measures `render = perf_counter()` while serializing the artifact to an SVG
   under `benchmarks/svg/...`.
4. Closes all matplotlib figures between replicas and runs `gc.collect()`.

Default replicas = 3.

### Construct/render split caveats

The split is best-effort, because the two libraries draw the line in different
places:

- **cellestial** returns a `PlotSpec` that defers most rendering work; the
  *construct* phase is therefore very fast and *render* dominates.
- **scanpy** functions build the matplotlib `Figure` (and frequently lay it
  out) at call time, so *construct* includes layout work that cellestial
  defers to *render*. For `DotPlot` / `MatrixPlot` / `StackedViolin` we use
  the scanpy `BasePlot.savefig()` path, which itself calls `show()`
  internally, so the *render* row is what's interpretable for those.

Compare *total* (construct + render) across libraries; treat the phase split
as informational, not normative.

### SVG export

Both libraries write a real `.svg` file. cellestial uses
`lets_plot.export.ggsave(plot, ..., iframe=False)`, with matplotlib forced to
the headless `Agg` backend. scanpy plot calls go through
`Figure.savefig(format="svg", bbox_inches="tight")` (or the equivalent
`BasePlot.savefig`).

## Results schema

| Column        | Type     | Notes                                               |
| ------------- | -------- | --------------------------------------------------- |
| `case`        | str      | e.g. `dotplot`                                      |
| `library`     | str      | `cellestial` or `scanpy`                            |
| `dataset`     | str      | dataset registry key                                |
| `n_cells`     | i64      | rows of the adata used                              |
| `n_keys`      | i64      | number of genes/keys plotted (1 for dim-reduction)  |
| `key_kind`    | str      | `gene`, `category`, `obs_obs`, `gene_gene`, `spatial_gene`, `n_top`, `n_genes` |
| `phase`       | str      | `construct` or `render`                             |
| `replica`     | i64      | 0..(replicas-1)                                     |
| `seconds`     | f64      | wall time (null if status != `ok`)                  |
| `svg_bytes`   | i64      | size of the generated SVG (0 for construct rows)    |
| `status`      | str      | `ok`, `skip:<reason>`, `error:<class>`              |
| `started_at`  | datetime | UTC timestamp                                       |

The consolidated `results.feather` is appended on each run; a per-invocation
snapshot is also written to `results/runs/`.

## Figures (lets-plot, `theme_bw()`)

- `scaling_by_data_size.svg` — faceted by case, x = `n_cells`, y = mean
  seconds, color = library, linetype = phase. Uses the default (smallest)
  `n_keys` per case.
- `scaling_by_key_count.svg` — heatmap-variant cases only (`dotplot`,
  `heatmap`, `matrixplot`, `stacked_violin`), faceted by `dataset` × `case`,
  x = `n_keys`, y = mean seconds.
- `construct_vs_render.svg` — stacked bars per `(case, library)` at the
  smallest dataset / `n_keys` per case; fill = phase.
- `overall_summary.svg` — total time per case dodged by library, error bars
  from replica sd.
