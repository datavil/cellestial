# Cellestial performance & memory audit

Findings from a static audit of `cellestial/` aimed at caching, parallelism,
laziness, memory, and algorithmic wins. Each item is a self-contained task.

## 1. Caching


- [x] **1b.** `_share_labels` / `_share_axis` / `_share_ticks` each recompute the same `(nrow, ncol, left_places, bottom_places)`. [cellestial/util/utilities.py:290-366](../cellestial/util/utilities.py#L290-L366)



## 2. Polars laziness / fewer passes


- [x] **2c.** Per-key `data.obs_vector(key)` instead of a single batched AnnData slice. [cellestial/frames/build.py:18-44](../cellestial/frames/build.py#L18-L44) **(implemented in [build_nextgen.py](../cellestial/frames/build_nextgen.py); see [perf_report.md](perf_report.md))**

## 3. Memory

- [x] **3c.** `frame[y].to_numpy()` just to call `.min()` / `.max()`; call on the Series. [cellestial/layers/bracket.py:241-247](../cellestial/layers/bracket.py#L241-L247)
- [ ] **3d.** Full sparse `X.multiply(...)` for top-N gene normalization; slice `X[:, top_idx]` first. [cellestial/frames/operations.py:22-39](../cellestial/frames/operations.py#L22-L39)


## 4. Algorithmic

- [x] **4a.** `key in column_names` (list) inside loop; convert to set once. [cellestial/frames/build.py:32-39](../cellestial/frames/build.py#L32-L39)



## 7. AnnData

- [x] **7a.** `data.obs_vector(key)` per-key fetch, no caching across calls. [cellestial/frames/build.py:36](../cellestial/frames/build.py#L36) **(subsumed by 2c)**

## 9. Imports / startup


- [x] **9b.** Inconsistent in-function vs top-level imports across the codebase. [cellestial/util/dendrogram.py:14-19](../cellestial/util/dendrogram.py#L14-L19)

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
