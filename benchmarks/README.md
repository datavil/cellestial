# Cellestial benchmarks

Claim under test: **cellestial (Polars + Lets-Plot) is fast and scales well**
for both pre-rendering (building the plot spec) and rendering (emitting SVG).

## Methodology

Each benchmark case is the **same visualization** expressed in both
cellestial and scanpy (matplotlib). We time two phases separately:

| phase        | cellestial                          | scanpy                              |
| ------------ | ----------------------------------- | ----------------------------------- |
| `construct`  | `cl.<plot>(...)` → `PlotSpec`       | `sc.pl.<plot>(..., return_fig=True)` → `Figure` |
| `render`     | `PlotSpec.to_svg()`                 | `Figure.savefig(buf, format="svg")` |
| `total`      | construct + render                  | construct + render                  |

- `warmup` runs are discarded; `repeats` runs are kept. Report median + IQR.
- Peak memory via `tracemalloc` on a single construct+render run.
- `gc.collect()` between runs.

### Cases

- `umap_gene` — UMAP colored by continuous gene expression
- `umap_cluster` — UMAP colored by categorical cluster label
- `dotplot` — marker × cluster dotplot (tests groupby aggregation)
- `heatmap` — marker × cluster heatmap
- `violin` — distribution of 4 genes
- `boxplot` — distribution of 4 genes (scanpy side: `pandas.DataFrame.boxplot`,
  since scanpy has no native boxplot)

### Datasets

**Real (scanpy).** Preprocessed datasets keep their real UMAP/clusters; raw
datasets get synthetic UMAP/cluster labels patched in (we benchmark plotting,
not dimred):
- `pbmc68k_reduced` — ~700 cells, real UMAP + `bulk_labels`
- `pbmc3k_processed` — ~2.7k cells, real UMAP + `louvain`
- `pbmc3k` — raw ~2.7k cells, patched embeddings
- `paul15` — ~2.7k cells, real `paul15_clusters`, patched UMAP
- `ebi_expression_atlas` — downloads accession `E-MTAB-4888` by default,
  patched embeddings

**Synthetic scaling.** `sc.datasets.blobs(n_observations=N)` with UMAP/PCA/cluster
labels patched in as random per-cluster gaussians. We do **not** benchmark
dimensionality reduction — only the plotting pipeline given preprocessed input.
Default sizes: 5k, 20k, 100k cells.

## Caveats

- **Not strictly apples-to-apples SVG.** Lets-Plot renders SVG via its Kotlin
  backend; matplotlib renders via its own SVG backend. Both produce SVG but
  the work is not identical. We report both and separate `construct` from
  `render` so the pre-render (Polars-driven) path can be read independently.
- The first lets-plot call in a process pays a one-shot setup cost
  (JVM-style bootstrap). `warmup >= 1` hides this.
- Memory numbers are allocator peak (tracemalloc), not RSS.
- `peak_mem_mb` is only populated on the `total` phase row; `construct` and
  `render` rows carry `0.0`. tracemalloc itself distorts timings, so we measure
  memory in one dedicated run rather than mixing it into every phase.

## Running

```bash
# smoke test
poetry run python -m benchmarks.run --quick

# full run (default sizes: 5k, 20k, 100k)
poetry run python -m benchmarks.run --repeats 5

# targeted
poetry run python -m benchmarks.run --cases heatmap dotplot --sizes 10000 50000

# plot results
poetry run python -m benchmarks.plot_results
```

Outputs: `benchmarks/results/results.csv`, `benchmarks/plots/*.png`.
