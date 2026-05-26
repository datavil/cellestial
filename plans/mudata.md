# MuData Support

## Background

MuData is a container of AnnData modalities. For per-modality plots the user
already extracts the modality (`mdata["rna"]`) and passes the AnnData in, so no
code is needed. The only thing native support buys is the **container level**:
global `mdata.obs`, `mdata.var`, and joint embeddings in `mdata.obsm`
(WNN / MOFA / totalVI), optionally colored by a variable from another modality.

Key fact: the MuData container duck-types AnnData for `.obs`, `.var`, `.obsm`,
`.varm`, `.obs_names`, `.var_names`. The single divergence is `.X`, which does
not exist at the container level (expression lives in the modalities).

## Purpose

Let `build_frame` accept a `MuData` and build a frame from container-level
`obs` / `var` / `obsm` / `varm`, resolving `variable_keys` to the owning
modality. Stays a thin visualization layer; no analysis logic added.

## Scope of change

Because the obs/var/obsm extraction is identical (duck-typed) and the only
backend-specific op is the `.X` pull, the divergence collapses to **one new
helper plus a single dispatch point**.

1. New `mudata_variable_columns(data, column_names, keys)` in `frames/build.py`.
   - Route each key to the owning modality.
   - Bare key (`"CD14"`) must be unique across modalities; qualified key
     (`"rna:CD14"`) selects the modality explicitly.
   - Pull `data[modality][:, name].X`, reindex to `data.obs_names` so
     observations absent from a modality align (surface as NaN).
   - Raise `VariableNotFoundError` if missing in any modality; `ValueError` if a
     bare key is ambiguous across modalities.

2. Wire it into the existing frame helpers (no `mudata_observations_frame`):
   - Loosen guards in `anndata_observations_frame` and `anndata_variables_frame`
     from `isinstance(data, AnnData)` to `isinstance(data, (AnnData, MuData))`.
   - In `anndata_observations_frame` PART 4 (the only `.X` site), dispatch:
     `mudata_variable_columns` when `isinstance(data, MuData)`, else
     `anndata_variable_columns`.

3. `build_frame` dispatcher:
   - Widen the existing `if isinstance(data, AnnData):` branch to
     `isinstance(data, (AnnData, MuData))`. The tuple is correct here because
     the type-specific work now lives one level down (step 2 PART 4 check), so
     the branch body is identical for both types.
   - Widen the signature to `AnnData | SpatialData | MuData`.
   - `from mudata import MuData`; add `mudata` as a dependency.

## Decision points

- **Naming:** `anndata_observations_frame` now also accepts MuData. Leave the
  name (surgical) or rename to `observations_frame` if more backends land.
- **NaN vs null:** missing observations come out as NaN, not polars null. Append
  `.fill_nan(None)` only if null semantics are wanted; it also clobbers genuine
  NaN, so it is a real choice. See `project_polars_drop_null_gotcha`.

## Alternative: `mdata.to_anndata()` (zero-code path)

MuData exposes `to_anndata()` (note: not `to_adata`), which runs
`anndata.concat` across modalities. Verified on mudata 0.3.8:

- Works: `.X` concatenated (all features addressable), container `obs`
  preserved (global + `modality:col` pulled up), joint `obsm` embedding kept.
- Breaks: var-name **collisions are not namespaced** (duplicate `var_names`,
  `data[:, key]` becomes ambiguous); raw per-modality matrices are dumped into
  `obsm` (pollutes `include_dimensions`); whole-X materialization is wasteful
  when only a few keys are needed; conflicting `var` metadata is dropped; obs/var
  pull behavior changes in mudata 0.4 (FutureWarning).

Boundary: recommend `to_anndata()` when modalities have **disjoint feature
names** and the conversion cost is acceptable. Build the helper (above) only for
**collisions** (explicit `modality:key` namespacing) and **efficiency**
(pull only requested keys). If neither matters, document `to_anndata()` and
skip the code.

Verified on real CITE-seq (`data/minipbcite.h5mu`, 411 cells, prot+rna):
- Names are disjoint (`CD3_TotalSeqB` vs `NKG7`), so `to_anndata()` var_names
  stay unique and a cross-modal frame works with **no cellestial code**:
  `build_frame(mdata.to_anndata(), variable_keys=["CD3_TotalSeqB","NKG7"])`.
- obsm pollution is real but pre-existing in the file (`prot`/`rna` raw matrices
  sit in `obsm` alongside `X_mofa`/`X_wnn_umap`), so `include_dimensions` over a
  broad set emits junk `PROT1`/`RNA1`. Selecting a specific embedding key avoids
  it; not a `to_anndata()` artifact.

Conclusion: CITE-seq and multiome both have disjoint feature names, so
collisions are uncommon. Default recommendation is **document `to_anndata()`,
do not build the helper** unless a real collision use case appears.

## Out of scope

- Auto-reducing a single-modality MuData (rare; SpatialData-style reduction not
  worth it when `mdata[key]` already works).
- MuData-level `varm` cross-modal semantics beyond plain concatenated `var`.

## Verification

1. Per-modality round-trip unchanged: `build_frame(mdata["rna"], axis=0)`.
2. Container obs + joint embedding: `build_frame(mdata, axis=0,
   include_dimensions=2)` includes `mdata.obsm` joint embeddings.
3. Cross-modal key: `build_frame(mdata, variable_keys=["rna:CD14"])` pulls from
   the RNA modality and aligns to `mdata.obs_names`.
4. Ambiguous bare key raises `ValueError`; unknown key raises
   `VariableNotFoundError`.
5. `axis=1` builds a variables frame from container-level `var` / `varm`.
