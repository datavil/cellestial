# Cellestial performance & memory audit

Findings from a static audit of `cellestial/` aimed at caching, parallelism,
laziness, memory, and algorithmic wins. Each item is a self-contained task.

## 1. Caching

- [ ] **1a.** `_color_gradient` / `_fill_gradient` compute `series.min()` and `series.max()` twice when `mid_point='mid'`. [cellestial/util/utilities.py:224-287](../cellestial/util/utilities.py#L224-L287)
- [ ] **1b.** `_share_labels` / `_share_axis` / `_share_ticks` each recompute the same `(nrow, ncol, left_places, bottom_places)`. [cellestial/util/utilities.py:290-366](../cellestial/util/utilities.py#L290-L366)
- [ ] **1c.** Repeated `float()` cast per element in nested `icoord`/`dcoord` loop. [cellestial/util/dendrogram.py:37-43](../cellestial/util/dendrogram.py#L37-L43)
- [ ] **1d.** Same column extracted via `.to_list()` three times for star / pvalue / padj labels. [cellestial/layers/bracket.py:210-237](../cellestial/layers/bracket.py#L210-L237)
- [ ] **1e.** `_resolve_embedding_key()` re-runs per panel in multi-plot grids; `@functools.cache` candidate. [cellestial/single/core/dimensional.py:200](../cellestial/single/core/dimensional.py#L200)

## 2. Polars laziness / fewer passes

- [ ] **2a.** `unpivot` then `drop_nulls` then `filter` is three eager passes; chain into one. [cellestial/single/core/distribution.py:116-134](../cellestial/single/core/distribution.py#L116-L134)
- [ ] **2b.** Eager `pl.DataFrame(columns)` build forfeits lazy optimization across multi-column construction. [cellestial/frames/build.py:94-133](../cellestial/frames/build.py#L94-L133)
- [ ] **2c.** Per-key `data.obs_vector(key)` instead of a single batched AnnData slice. [cellestial/frames/build.py:18-44](../cellestial/frames/build.py#L18-L44) **(implemented in [build_nextgen.py](../cellestial/frames/build_nextgen.py); see [perf_report.md](perf_report.md))**

## 3. Memory

- [ ] **3a.** Vectorized numpy result then materialized back into Python floats in a loop. [cellestial/util/dendrogram.py:27-35](../cellestial/util/dendrogram.py#L27-L35)
- [ ] **3b.** `.to_numpy()` on selected columns when downstream may accept Polars directly. [cellestial/layers/stream.py:81-82](../cellestial/layers/stream.py#L81-L82)
- [ ] **3c.** `frame[y].to_numpy()` just to call `.min()` / `.max()`; call on the Series. [cellestial/layers/bracket.py:241-247](../cellestial/layers/bracket.py#L241-L247)
- [ ] **3d.** Full sparse `X.multiply(...)` for top-N gene normalization; slice `X[:, top_idx]` first. [cellestial/frames/operations.py:22-39](../cellestial/frames/operations.py#L22-L39)
- [ ] **3e.** `keys.copy()` then mutating the copy while iterating the original. [cellestial/single/common/xyplot.py:167](../cellestial/single/common/xyplot.py#L167)

## 4. Algorithmic

- [ ] **4a.** `key in column_names` (list) inside loop; convert to set once. [cellestial/frames/build.py:32-39](../cellestial/frames/build.py#L32-L39)
- [ ] **4b.** `isinstance(include_dimensions, int) and not isinstance(..., bool)` re-checked every iteration. [cellestial/frames/build.py:110-121](../cellestial/frames/build.py#L110-L121)
- [ ] **4c.** Grid-layout list comprehensions rebuilt three times across helper trio. [cellestial/util/utilities.py:290-366](../cellestial/util/utilities.py#L290-L366)
- [ ] **4d.** `obsm.keys()` looped twice with repeated `.upper()` calls. [cellestial/util/utilities.py:600-627](../cellestial/util/utilities.py#L600-L627)

## 5. Parallelism

- [ ] **5a.** Sequential `violin()` per key in grid builder. [cellestial/single/core/distributions.py:199-235](../cellestial/single/core/distributions.py#L199-L235)
- [ ] **5b.** Sequential `dimensional()` per key; frame build + plot independent. [cellestial/single/core/subdimensionals.py:223-250](../cellestial/single/core/subdimensionals.py#L223-L250)
- [ ] **5c.** Blocking `urlopen` download in `from_url` / `from_cellxgene`; async / threaded variant possible. [cellestial/datasets/datasets.py:424-523](../cellestial/datasets/datasets.py#L424-L523)

## 6. Rendering / lets-plot

- [ ] **6a.** Confirm `_THEME_DIST` is not deep-copied per `+` in the multi-plot loop. [cellestial/single/core/distributions.py:153](../cellestial/single/core/distributions.py#L153)
- [ ] **6b.** Categorical color scale recomputed per call; could cache by `(key, n_unique)`. [cellestial/single/core/dimensional.py:248-250](../cellestial/single/core/dimensional.py#L248-L250)
- [ ] **6c.** `scale_size(trans="sqrt", breaks=[25,50,75,100])` hardcoded; no override path. [cellestial/themes/_scatters.py:71](../cellestial/themes/_scatters.py#L71)

## 7. AnnData

- [ ] **7a.** `data.obs_vector(key)` per-key fetch, no caching across calls. [cellestial/frames/build.py:36](../cellestial/frames/build.py#L36) **(subsumed by 2c)**
- [ ] **7b.** Extra `.tocsr()` after sparse multiply — wasted copy. [cellestial/frames/operations.py:37-38](../cellestial/frames/operations.py#L37-L38)

## 8. Utility redundancy

- [ ] **8a.** Identical `isinstance(data, AnnData)` + error block duplicated across `_is_*_key` helpers. [cellestial/util/utilities.py:401-504](../cellestial/util/utilities.py#L401-L504)
- [ ] **8b.** Categorical-integer handling duplicated between observations / variables frames. [cellestial/frames/build.py:98-104](../cellestial/frames/build.py#L98-L104), [cellestial/frames/build.py:180-187](../cellestial/frames/build.py#L180-L187)

## 9. Imports / startup

- [ ] **9a.** Top-level `from anndata import read_h5ad`; defer into loader bodies. [cellestial/datasets/datasets.py:1-9](../cellestial/datasets/datasets.py#L1-L9)
- [ ] **9b.** Inconsistent in-function vs top-level imports across the codebase. [cellestial/util/dendrogram.py:14-19](../cellestial/util/dendrogram.py#L14-L19)

## Top 10 leverage ranking

1. **4a.** list to set in `anndata_variable_columns` (LOW effort, hot loop)
2. **1b. / 4c.** extract `_compute_grid_layout()` (MEDIUM, removes 3x duplication)
3. **2a.** collapse three-pass filter in distributions (LOW)
4. **1e.** `@cache` on `_resolve_embedding_key` (MEDIUM, multi-plot win)
5. **7b.** drop redundant `.tocsr()` (LOW)
6. **3c.** Polars `.min()` / `.max()` instead of `.to_numpy()` in brackets (LOW)
7. **3d.** slice sparse `X` before normalize in `_highest_expressed_genes_frame` (MEDIUM)
8. **2c.** batched Polars `.select` for AnnData var extraction (MEDIUM) **(done)**
9. **5a.** parallelize per-key violins (HIGH, needs lets-plot thread-safety check)
10. **8a.** extract `_validate_data_type()` helper (LOW, cleanup with reuse)
