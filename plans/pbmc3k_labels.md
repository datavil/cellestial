# pbmc3k cell type labels: check against canonical markers (2026-11-18)

Raised by the 2026-11-18 differential audit ("B Cells" has HBB/HBA as top markers).
Scope: check every `cell_type_lvl1` label of `cl.datasets.pbmc3k()` (17041 cells,
23427 features, samples `s1d1` 8713 and `s1d3` 8328) and propose a corrected
mapping. **No change to the dataset or its loader**; this is a proposal.

There is no `cell_type_lvl2` column. The only label column is `cell_type_lvl1`.

## Finding 1: `cell_type_lvl1` is a 1:1 relabel of `leiden_res_0.02`

| `leiden_res_0.02` | cells | current `cell_type_lvl1` |
| --- | ---: | --- |
| 0 | 9721 | Lymphocytes |
| 1 | 1510 | Monocytes |
| 2 | 2300 | Erythroid |
| 3 | 1743 | B Cells |
| 4 | 1767 | null |

So every label error is a whole-cluster error and the fix is a relabel of five
cluster ids.

## Finding 2: "B Cells" and "Erythroid" are swapped

Mean log-normalized expression (fraction of cells with expression > 0 in brackets),
`layers` default `X`:

| label | MS4A1 | CD79A | CD3E | NKG7 | GNLY | LYZ | CD14 | FCGR3A | HBB | HBA1 | PPBP | FCER1A | CST3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B Cells | 0.00 (0.01) | 0.01 (0.02) | 0.01 (0.03) | 0.02 | 0.02 | 0.08 | 0.00 | 0.00 | **8.21 (1.00)** | **6.14 (1.00)** | 0.00 | 0.00 | 0.01 |
| Erythroid | **1.48 (0.81)** | **1.70 (0.87)** | 0.06 (0.07) | 0.06 | 0.06 | 0.18 | 0.01 | 0.01 | 2.80 (1.00) | 1.18 | 0.00 | 0.01 | 0.06 |
| Lymphocytes | 0.03 | 0.07 | **0.76 (0.57)** | **0.57 (0.26)** | **0.44** | 0.26 | 0.03 | 0.10 | 2.95 | 1.25 | 0.00 | 0.02 | 0.11 |
| Monocytes | 0.03 | 0.04 | 0.05 | 0.20 | 0.06 | **3.78 (1.00)** | **1.32 (0.82)** | 0.18 | 2.65 | 1.30 | 0.01 | 0.17 | **2.31 (0.99)** |
| null | 0.03 | 0.04 | 0.05 | 0.06 | 0.06 | 0.22 | 0.04 | 0.01 | **7.42 (1.00)** | **5.42 (1.00)** | 0.00 | 0.01 | 0.12 |

QC medians per label support it: "B Cells" has `pct_counts_hb` 92.2% and 438 genes
per cell (mature red cells: almost only hemoglobin), "Erythroid" has `pct_counts_hb`
0.4% and 1600 genes (a normal lymphocyte profile).

- **"B Cells" (cluster 3) are erythroid cells**, mature red cells / reticulocytes:
  HBB, HBA1, HBD, ALAS2, AHSP in ~100% of cells, TFRC off, 438 genes, 0.3% mito.
- **"Erythroid" (cluster 2) are B cells**: MS4A1 in 81%, CD79A in 87% of cells,
  no hemoglobin beyond the ambient level every cluster shows.
- **"Lymphocytes" (cluster 0)** is correct as a lineage: T cells (CD3E/CD3D) plus
  NK cells (NKG7/GNLY/FCGR3A).
- **"Monocytes" (cluster 1)** is correct: LYZ, CD14, S100A8, CST3.
- **null (cluster 4)** is erythroid too, but earlier in maturation (erythroblasts):
  GYPA, CA1, AHSP, ALAS2, SLC4A1 and TFRC in >80% of cells, 59% hemoglobin
  counts, ~1900 genes, and a proliferating part (MKI67 in 91% of `leiden_res_0.50`
  cluster 9).
- **No platelets**: PPBP and PF4 are essentially zero everywhere.
- **No separate DC label**: FCER1A/CST3-high cells exist (`leiden_res_0.50` cluster
  3, 83 cells) but sit inside "Lymphocytes" at this resolution.

HBB is detected in 100% of cells of every cluster (mean 2.3 to 3.8 outside the
erythroid clusters): ambient hemoglobin RNA, not a label signal.

## Finding 3: the tissue is bone marrow, not PBMC

Erythroblasts (GYPA, TFRC, proliferating), pro-B / pre-B cells (CD34, DNTT, VPREB1,
IGLL1) and plasma cells (MZB1, JCHAIN) are bone-marrow populations, and the sample
ids `s1d1` / `s1d3` match the site/donor naming of the NeurIPS 2021 bone marrow
multiome benchmark. The dataset name `pbmc3k` (and its 17041 cells) is therefore
misleading. Not changed here; worth a note in the loader docstring or a rename
decision.

## Finer structure (`leiden_res_0.50`), evidence for an optional level 2

| cluster | cells | evidence | proposed level 2 |
| --- | ---: | --- | --- |
| 0 | 3945 | CD3E/CD3D 0.91 to 0.96, IL7R, no NKG7 | CD4 / naive T |
| 11 | 1302 | CD3 + CD8A, NKG7 0.88 | CD8 T |
| 7 | 459 | NKG7 1.00, GNLY 0.95, FCGR3A 0.77, CD3E low | NK |
| 4 | 3050 | weak T/NK markers, 605 genes, 27.9% mito | low-quality lymphocytes |
| 1 | 1440 | LYZ 1.00, CD14 0.85, S100A8 0.97 | CD14 monocytes |
| 15 | 71 | FCGR3A 1.00, CST3 1.00, LYZ 0.92 | FCGR3A monocytes |
| 6 | 309 | LYZ 0.73, 781 genes, 40% mito | low-quality myeloid |
| 3 | 83 | FCER1A 0.70, CST3 0.98, JCHAIN/MZB1 0.98, CD4 0.84 | dendritic cells (cDC2 + pDC mixed) |
| 2, 10 | 1917 | MS4A1 0.88 to 0.96, CD79A 0.77 to 0.99 | mature B |
| 8 | 386 | CD79A 0.87, VPREB1 0.65, IGLL1 0.46, MS4A1 0.27 | pre-B / immature B |
| 14 | 508 | CD34 0.48, DNTT 0.32, VPREB1 0.48, IGLL1 0.70, MKI67 0.39 | pro-B / progenitors |
| 13 | 70 | MZB1 0.93, JCHAIN 0.57, CD79A 0.70, MS4A1 0.13 | plasma cells |
| 5, 16 | 1743 | hemoglobin 92%, ~440 genes, TFRC off | mature erythroid / reticulocytes |
| 12, 9 | 1758 | GYPA, CA1, AHSP, ALAS2, TFRC; cluster 9 MKI67 0.91 | erythroblasts |

Counts are `leiden_res_0.50` cluster sizes. The two resolutions nest almost exactly;
the exceptions are a handful of cells (e.g. 9 null cells in cluster 14). Clusters 3,
6, 13 and 14 sit inside "Lymphocytes" at level 1, so that label also holds DCs,
low-quality myeloid cells, plasma cells and B progenitors.

## Proposed corrected mapping

Level 1, from `leiden_res_0.02`:

| `leiden_res_0.02` | current | proposed |
| --- | --- | --- |
| 0 | Lymphocytes | T/NK Cells (or keep "Lymphocytes") |
| 1 | Monocytes | Monocytes |
| 2 | Erythroid | **B Cells** |
| 3 | B Cells | **Erythroid** |
| 4 | null | **Erythroid** (erythroblasts; or "Erythroid Progenitors" to keep it separate) |

Minimal option: swap only clusters 2 and 3 and leave 4 null. This keeps every label
name and the 1767 null cells, so no test changes, and fixes the biology.

Level 2: the `leiden_res_0.50` table above, if a second level is wanted.

## Tests and docs that depend on the current labels

Names only (they keep working after the minimal swap, since all four names still
exist; the biology they show changes):

- `tests/test_layers.py:128, 135, 142` (`cluster_outlines(groups="B Cells" / ["B Cells", "Erythroid"] / ...)`)
- `tests/test_errors.py:92` (`groups=["Lymphocytes", ...]`)
- `tests/test_heatmap_audit.py:229` (`isin(["B Cells", "Monocytes"])`, shared barcodes fixture)
- `tests/test_followup_differential.py:91, 94` (`volcano(..., "B Cells", min_fraction_expressed=0.1)`)
- `tests/test_heatmap.py:382, 391` and `tests/test_followup_differential.py:217` use
  the names as plain strings, not as data.

Group structure (column `cell_type_lvl1` via the `group_key` fixture in
`tests/conftest.py`): `tests/test_heatmap.py`, `tests/test_heatmap_audit.py`
(`value_counts().index[-1]` picks the smallest group in the first 3000 cells),
`tests/test_followup_heatmap.py`, `tests/test_coverage_boost.py`, and every test
using `group_key`. These depend on four groups plus nulls existing. Merging cluster
4 into "Erythroid" removes the null cells, which the 2026-11-18 audits used to
exercise null-group handling (heatmap audit fix 4, `_filter_groups` null tests use
their own frames).

Docstring examples naming labels: `cellestial/layers/outline.py:197, 214, 231`,
`cellestial/layers/bracket.py:293, 304`, `cellestial/layers/highlight.py:78`,
`cellestial/single/differential/volcano.py:99, 249, 519, 671`. Notebooks under
`test/` and `README.md` use `cell_type_lvl1` too.

## Reproduce

```python
import numpy as np, pandas as pd
import cellestial as cl

data = cl.datasets.pbmc3k(cache_directory="data")
genes = ["MS4A1", "CD79A", "CD3E", "NKG7", "GNLY", "LYZ", "CD14", "FCGR3A",
         "HBB", "HBA1", "PPBP", "FCER1A", "CST3"]
matrix = data[:, genes].X
frame = pd.DataFrame(matrix.toarray() if hasattr(matrix, "toarray") else matrix,
                     columns=genes)
labels = data.obs["cell_type_lvl1"].astype(str).to_numpy()
print(frame.groupby(labels).mean().round(2))
print(pd.crosstab(data.obs["cell_type_lvl1"], data.obs["leiden_res_0.02"]))
```
