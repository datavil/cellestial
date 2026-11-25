# Cellestial 1.0.0

Cellestial 1.0 is the first stable release. It finishes the v1 API review, so
most code written for 0.62 needs a few renames. See the
[migration guide](https://cellestial.datavil.org/migrating_to_v1.html) and the
full list in `CHANGELOG.md`.

## Breaking changes

- **Datasets.** `cl.datasets.pbmc3k()` used to load bone marrow cells (the
  scanpy clustering tutorial data). That loader is now
  `cl.datasets.bone_marrow()`, with the same cells and labels.
  `cl.datasets.pbmc3k()` now loads the real 10x Genomics PBMC 3k (2,638 cells,
  labels in `cell_type`).
- **Stored-result keys.** `ranking_key` (was `key`) in `markers`, `volcano(s)`,
  `marker_genes(_dict)`; the heatmap family takes `markers: bool` plus
  `ranking_key`. `embedding_key` (was `use_key`) in the dimensional family.
- **Renames.** `dotplot(expression_threshold=)` (was `threshold`), columns
  `percent_expressed` and `mean_expression` (were `pct_exp`, `avg_exp`);
  `spatial(grayscale=)` (was `greyscale`); `variable_keys` removed in favor of
  `add_keys`; `save` is a thin wrapper over `ggsave`.
- **Signatures.** `groups` is keyword-only in `markers`, `marker_genes` and
  `marker_genes_dict`.
- **Errors.** Errors now come from `cellestial.errors`, and every class
  subclasses `CellestialError` and the matching builtin (`KeyNotFoundError` is
  a `KeyError`). A renamed parameter raises `UnknownParameterError` naming the
  new one.
- **Frames.** `build_frame` returns categorical metadata as `pl.Enum` in the
  source category order. The AnnData-specific frame builders and
  `DeferredLayer` are no longer public.

## New

- `highlight` works on plots filled by a key (`violin`, `boxplot`, ...).
- `annotated_heatmap` gains `variable_group_by`, `key_labels`, `group_labels`
  and `annotation_legends`.
- `outline=True` on the dimensional plots (`umap`, `tsne`, `pca`, `expression`, ...).
- `volcano(s)` take `min_fraction_expressed`.

## Improvements

- Clearer errors throughout: every error says what was wrong, where, and
  suggests a close match for typos (`'leidn'`: did you mean `'leiden'`?).
- pandas 3 is supported on Python 3.12 and later.
- The bundled datasets download from one Hugging Face repo,
  [`datavil/cellestial-datasets`](https://huggingface.co/datasets/datavil/cellestial-datasets).
- Many fixes across heatmaps, volcano plots, spatial polygons (holes are drawn
  as holes) and plural grids; see `CHANGELOG.md`.

## Install

```bash
pip install --upgrade cellestial
pip install --upgrade "cellestial[spatial]"   # for SpatialData
```
