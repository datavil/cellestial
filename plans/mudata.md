# MuData Support

> **Status: implemented** (2026-08-04). 757 tests pass, 82 of them new in
> `tests/test_mudata.py`; the 675 pre-existing tests are unchanged.
>
> **Validated against three real datasets, not one.** `minipbcite.h5mu` (411
> cells, CITE-seq, 2 modalities), `HumanTonsil_filtered.h5mu` (4194 cells,
> variable names stored already prefixed, `obsm` holding a `DataFrame`) and
> `pancan_depmap.h5mu` (778 cell lines, **12** modalities, **2521 genuinely
> duplicated variable names**, modality names containing spaces). The last two
> are Zenodo downloads and are not fixtures; the shapes they exposed are
> reproduced synthetically in the test suite so it stays hermetic. Two of the
> three found bugs listed below came from those datasets and could not have been
> found on `minipbcite` alone.
>
> Five things the plan did not anticipate, all now in the code:
>
> 1. **Circular import.** `util/utilities.py` has to import `_container` lazily
>    inside a function. `_container.py` imports `cellestial.util.errors`, which
>    executes `cellestial/util/__init__.py`, which imports `utilities` while
>    `_container` is still initialising.
> 2. **Cheap column accessors.** `observation_columns` / `variable_columns` are
>    separate from `observation_metadata` / `variable_metadata`, so the hot
>    key-classification predicates never call `to_memory()` on a backed table
>    just to read column names.
> 3. **Prefix translation for dendrograms.** A container column is `rna:leiden`
>    while the modality calls it `leiden`. `_Container.modality_column` and the
>    `_dendrogram_source` helper bridge that; without it every dendrogram on a
>    container failed.
> 4. **Dendrogram precondition.** Grouping by a container-only column such as
>    `leiden_wnn` cannot be clustered by any single modality, so `_get_dendrogram`
>    raises a clear error instead of an opaque scanpy traceback.
> 5. **`from_url` reader hook.** It hardcoded `read_h5ad` at two return sites; it
>    now takes `extension` and dispatches through `_read_dataset`.
>
> Known fixture limitation: `minipbcite.h5mu` was subset to 27 genes *after*
> `rank_genes_groups` was computed, so `markers=True` fails on it for stale gene
> names. This is not a MuData issue: `cl.heatmap(mdata["rna"], markers=True)`
> fails identically on the pure AnnData path. Tests re-rank a copy of the
> modality to get past it.
>
> **Bugs found by widening to other datasets.**
>
> - *Colons in variable names.* Reading a key only as `modality` + `name` made
>   **every variable unreachable** in a dataset whose names are stored already
>   prefixed (`rna:SAMD11`, the tonsil dataset), and would have done the same to
>   ATAC peaks (`chr1:1000-2000`). The qualified reading is still tried first,
>   then the key is matched literally. A bad qualified key keeps its precise
>   error.
> - *`DataFrame` embeddings.* An embedding store accepts a `DataFrame`, which
>   rejects the `value[:, column]` indexing the frame builder uses, raising
>   `InvalidIndexError`. Pre-existing and **not** MuData-specific: plain AnnData
>   failed identically. Fixed in the container, which is now the single place
>   embeddings are read, so both backends benefit.
>
> **Bug the fixture limitation was hiding.** Re-ranking exposed a silent wrong-data
> path. A ranking records its group column under the *modality's* name
> (`leiden`), but the plotted frame is built from the container, which in this
> fixture carries both a global `leiden` (9 joint clusters) and `rna:leiden` (14
> RNA clusters) that disagree on 356 of 411 cells. Marker genes ranked on RNA's
> clustering were therefore grouped by the joint clustering, with no error.
> `_Container.container_column` now translates the stored name back up to the
> container, preferring the qualified form, and `group_by="rna:leiden"` is
> accepted where previously it was rejected for "not matching" the stored
> `leiden`. Covered by `test_markers_group_by_the_clustering_the_ranking_used`.


Supersedes the earlier revision of this document, which recommended documenting
`mdata.to_anndata()` and building nothing. That conclusion rested on "feature
name collisions are uncommon". Collisions are uncommon but they fail *silently*
(see Verified facts, item 5), and `to_anndata()` copies the whole concatenated
matrix on every call even when the plot needs three variables. Both are now
addressed by the container below.

## Decisions

| Question | Decision |
| --- | --- |
| Backend dispatch | Accessor module: one module owns every backend-specific access, callers ask it questions |
| Variable naming | Bare name when unique across modalities, `modality:name` to disambiguate and always wins |
| Scope | Container-native wherever the container is meaningful, `modality=` where results are per-modality, skipped for spatial |
| Modality-scoped `uns` | Explicit `modality=` parameter, required when the object has more than one modality |

"No additional classes" in `plans/principles.md` sits under Simplicity, which is
about the API a user touches. `cellestial/layers/_deferred.py` already holds an
internal `DeferredLayer` class, so the rule is read here as *no user-constructed
classes*. The container below is strictly internal: see "Keeping it internal".

## Verified facts

Checked against mudata 0.3.8, anndata 0.12.16, `data/minipbcite.h5mu`
(411 cells, `prot` 29 + `rna` 27).

1. `MuData` does not subclass `AnnData` (`MuData.__mro__` is `[MuData, object]`),
   so every `isinstance(data, AnnData)` guard rejects it today.
2. The container duck-types AnnData for `.obs`, `.var`, `.obs_names`,
   `.var_names`, `.obsm`, `.varm`. The classification helpers in
   `util/utilities.py` would work on it unmodified.
3. `mdata.obs` already carries modality columns pulled up and prefixed
   (`rna:leiden`, `rna:celltype`) alongside genuinely global ones (`louvain`,
   `leiden`, `leiden_wnn`, `celltype`). Same for `mdata.var`. No merge logic is
   needed from us.
4. `mdata.X` exists but is always `None`. `mdata[:, keys]` slices correctly and
   routes across modalities, returning a `MuData`, so the values must be read
   from `.mod[name].X` afterwards.
5. Bare `mdata[:, "NKG7"]` resolves natively. Prefixed `mdata[:, "rna:NKG7"]`
   raises `KeyError`, so `modality:name` is our addition for variables (it is
   mudata's own convention only for obs/var *columns*). On a genuine collision,
   `mdata[:, "CD14"]` silently returns **both** modalities, shape `(4, 2)`, with
   no warning. That silent widening is the bug the prefix exists to fix.
6. `mdata.obsm` and `mdata.varm` carry boolean membership masks keyed by
   modality name (`obsm["rna"]` is `(411, 1)` bool) alongside real embeddings
   (`X_mofa`, `X_mofa_umap`, `X_umap`, `X_wnn_umap`). Unfiltered, these become
   junk `RNA1` / `PROT1` columns in every frame.
7. `mdata.uns` is **empty**. `rank_genes_groups`, `pca`, `leiden_colors`, `hvg`
   live only in `mdata["rna"].uns`.
8. Column names containing `:` render correctly in lets-plot for both
   categorical and continuous aesthetics. The `rna:leiden` naming is safe.
9. `mdata.axis` is 0 for this file. Only axis 0 (shared observations,
   concatenated variables) is in scope.

## Architecture

New module `cellestial/frames/_container.py`, named after
`cellestial/layers/_deferred.py`. It is the only place in the codebase that
touches a backend-specific attribute.

Because the MuData container duck-types AnnData for `.obs`, `.var`, `.obsm`,
`.varm`, `.obs_names` and `.var_names` (fact 2), the base class serves **both**
types and the MuData subclass overrides only what genuinely differs. The
override list is therefore the complete inventory of the divergence.

```python
class _Container:
    """Backend-agnostic view over a single-cell data object."""

    __slots__ = ("_data",)

    def __init__(self, data: AnnData) -> None:
        self._data = data

    # --- shared, correct for AnnData and MuData alike -------------------
    def observation_metadata(self) -> pd.DataFrame      # .obs, Dataset2D-safe
    def variable_metadata(self) -> pd.DataFrame         # .var, Dataset2D-safe
    def observation_names(self) -> pd.Index
    def variable_names(self) -> pd.Index

    # --- overridden by _MuDataContainer ---------------------------------
    def modality_names(self) -> list[str]                       # []
    def select_modality(self, modality: str | None) -> AnnData  # self._data
    def observation_embeddings(self) -> dict[str, np.ndarray]   # dict(.obsm)
    def variable_embeddings(self) -> dict[str, np.ndarray]      # dict(.varm)
    def owns_variable(self, key: str) -> bool                   # never raises
    def resolve_variable(self, key: str) -> tuple[str | None, str]  # raises
    def fetch_variable_columns(self, keys) -> list[pl.Series]


class _MuDataContainer(_Container):
    """Adds modality routing. Overrides exactly the six divergent methods."""

    __slots__ = ()


def _container(data: object) -> _Container:
    if isinstance(data, MuData):
        return _MuDataContainer(data)
    if isinstance(data, AnnData):
        return _Container(data)
    raise _unsupported_data_type(data, AnnData, MuData)
```

`__init__` only stores a reference, so constructing a container is free and
callers keep their current `data` signatures. Nothing needs threading through
the call graph, which keeps step 3 a body-only change.

**Two lookups, not one, and the distinction matters.** The `_is_*` / `_are_*`
predicates are used in boolean context and must never raise, so they call
`owns_variable`. The fetch path wants a precise error, so it calls
`resolve_variable`. Collapsing them would make
`_is_variable_key(mdata, "rna:leiden")` raise instead of returning `False`,
breaking classification of the pulled-up metadata columns.

**Column naming.** `fetch_variable_columns` names each series with the key the
user wrote, so `"rna:CD14"` stays `rna:CD14` and `"NKG7"` stays `NKG7`. Plot
functions map aesthetics to the user's own string and never have to learn what
it resolved to. Verified safe in lets-plot by fact 8.

### Keeping it internal

Stricter than `DeferredLayer`, which is un-prefixed only because it appears in
public return annotations. This one never appears in a signature at all.

- Underscore module, underscore class names, underscore factory.
- Absent from `cellestial/frames/__init__.py`'s `__all__` and never imported in
  `cellestial/__init__.py`. Matches the 0.59.0 change that stripped
  underscore-prefixed names from the package `__all__` declarations.
- Never a parameter or return type in a public signature. Users pass
  `AnnData | MuData` and get a `PlotSpec`, exactly as today.
- Never constructed, subclassed, or named in user-facing docs or docstrings.
- `tests/test_api_audit.py` gains an assertion that no container name leaks into
  the public surface.

`mudata` is already a hard dependency (`pyproject.toml`) and costs 2.2 ms to
import once `anndata` is loaded, so it is imported at module top rather than
lazily like `spatialdata`.

### Variable resolution

```
"rna:CD14"  -> split on the first ":"; unknown modality or missing name raises
"NKG7"      -> exactly one owning modality  -> that modality
"CD14"      -> several owning modalities    -> AmbiguousVariableError listing
                                               "rna:CD14", "prot:CD14"
"XYZ"       -> no owning modality           -> VariableNotFoundError
```

Metadata wins over variables. `_collect_aes_columns` already tests the metadata
pool before variable names, so `rna:leiden` keeps resolving as an obs column and
never reaches variable resolution.

### Alignment for absent observations

`obsmap[modality]` maps each global observation to a 1-based position in that
modality, with 0 meaning absent. Dense in this fixture, so a positional
assignment would pass every test here and corrupt data on any object where a
modality is missing cells.

```python
positions = data.obsmap[modality][:, 0]          # 1-based, 0 == absent
present = positions > 0
dtype = np.promote_types(matrix.dtype, np.float32)   # NOT dtype=float
values = np.full(data.n_obs, np.nan, dtype=dtype)
values[present] = matrix[positions[present] - 1]
```

Densify sparse `X` first. The fill dtype must be promoted from the source, not
hardcoded: single-cell `.X` is float32 almost universally (both modalities of
the fixture are), and `dtype=float` means float64, so every variable column
fetched through this path would be **exactly 2x** the size of the same column
fetched from an AnnData. `np.promote_types(dtype, np.float32)` keeps float32 at
float32 and still widens int32/int64 counts to float64, where float32 could not
represent them exactly above 2^24.

Missing entries surface as NaN rather than polars null, matching
`project_polars_drop_null_gotcha`.

### Memory

Audited against the fixture, since a container design invites copies.

| Site | Copy? |
| --- | --- |
| `_container(data)` | None. Stores a reference, no work in `__init__`. |
| `observation_embeddings()` | None. `dict(...)` rebinds keys, `np.shares_memory` confirms the arrays are the originals. |
| `observation_metadata()` | None. Returns `.obs` itself. The per-column `pl.Series` copy in `build.py` is pre-existing and identical for AnnData. |
| `_variable_column()` | One column of `n_obs` values per key, at source dtype once promoted correctly. Matches the AnnData path, which allocates `n_obs * n_keys` in a single slice. |
| `pl.Series(key, values)` | None for float arrays. Verified zero-copy: mutating the numpy array afterwards is visible through the series. |

Two things that are **not** ours but that users will attribute to us:

- mudata's own `pull_obs` stores modality obs columns a second time in the
  container (`rna:leiden` alongside `mdata["rna"].obs["leiden"]`, 178 KB in this
  small fixture). So `build_frame(mdata, axis=0)` with the default
  `metadata_columns=None` materialises both `leiden` and `rna:leiden`. Plot
  functions narrow via `_collect_aes_columns` and are unaffected, but a bare
  `build_frame` call is wider than the AnnData equivalent. Document it.
- `to_anndata()` holds a full second copy of every modality matrix while the
  original is still alive, plus the raw per-modality matrices again in `obsm`.
  Same total bytes as the source, so 2x peak, and it is 2x of the *entire*
  feature space even when the plot needs three variables. This is the concrete
  reason the container path exists.

## Scope

**Container-native.** MuData is accepted directly, the frame is built from
container `obs` / `var` / `obsm` / `varm`, and variables resolve per modality.

- `build_frame`
- `dimensional`, `dimensionals`, `umap`, `umaps`, `tsne`, `tsnes`, `pca`, `pcas`
- `scatter`, `xyplot`, `xyplots`
- `violin(s)`, `boxplot(s)`, `histogram(s)`, `ridge(s)`, `expression(s)`, `bar`
- `heatmap`, `dotplot`, `matrixplot`, `stacked_violin`, `annotated_heatmap`

This is where the container earns its place. `X_wnn_umap` and `X_mofa_umap`
exist *only* at the container level, so a joint embedding colored by a protein
is impossible without this work. Gene-versus-protein `scatter` and a `dotplot`
mixing both feature types on one axis grouped by `leiden_wnn` are the two plots
users actually come to MuData for.

`pca` needs no special handling: with no container `X_pca`, the existing
`_resolve_embedding_key` raises and lists what is available, which is correct.

Deferred layers (`stream`, `bracket`, `ondata_legend`, `cluster_outlines`,
`arrow_axis`) need **no work**. They read the built frame, not the data object
(`project_deferred_layers_read_frame`).

**Per-modality via `modality=`.** These read stored analysis results from `uns`,
which is empty at the container.

- `markers`, `volcano`, `volcanos`, `elbow`, `highest_expressed_genes`
- the `uns`-reading options of the heatmap family (dendrogram, marker groups)
  and `annotated_heatmap`

One meaning throughout: *which modality's stored results to use*. For the first
group that also selects the data, because the whole computation is per-modality.
Required when a MuData with more than one modality is passed, ignored entirely
for AnnData.

**Skipped, and settled rather than deferred.** `spatial`, `spatials`.

Spot coordinates live in `obsm["spatial"]` and the tissue image in
`uns["spatial"][library_id]`. mudata pulls `obs` and `var` up to the container
but **not** `obsm` and **not** `uns`, so a container has neither. Letting a
container through therefore either fails on missing coordinates, or, once a user
lifts coordinates up by hand, takes the "no image metadata" branch and returns
`image=None` with **no error**. Silently dropping the tissue image is worse than
refusing.

The one capability native support would have added, colouring spots by a
variable from another modality, already works by composition:

```python
cl.spatial(
    mdata["rna"],
    key="prot:CD3",
    frame=cl.build_frame(mdata, variable_keys=["prot:CD3"]),
)
```

Coordinates and image come from the modality, so nothing has to decide whose
registration is authoritative; values come from the container frame, so they may
belong to any modality. Nothing is lost by refusing, so this is a decision, not
a gap. Covered by `test_spatial_colours_across_modalities_via_a_container_frame`,
and the rejection message points at the composition.

## Implementation steps

**1. `cellestial/frames/_container.py`.** The module above, plus
`AmbiguousVariableError(ValueError)` in `util/errors.py`. Standalone and fully
unit-testable before anything calls it.

**2. `frames/build.py`.** Replace direct slot access with container calls in
`anndata_observations_frame` and `anndata_variables_frame`: one
`container = _container(data)` at the top, then `data.obs` becomes
`container.observation_metadata()`, `data.obsm` becomes
`container.observation_embeddings()` (which drops the masks from fact 6), and
the PART 4 `.X` pull becomes `container.fetch_variable_columns(...)`. Widen the
guards and the `build_frame` signature to `AnnData | SpatialData | MuData`, and
reject `mdata.axis != 0` with a clear message. The two frame builders keep their
names for now; renaming to `observations_frame` / `variables_frame` is a
separate, larger change.

**3. `util/utilities.py`.** Point the twelve classification helpers
(`_require_feature_key`, `_is_variable_key`, `_are_variables`,
`_is_observation_key`, `_are_observations`, `_select_variable_keys`,
`_collect_aes_columns`, `_is_observation_feature`, `_are_observation_features`,
`_is_variable_feature`, `_are_variable_features`, `_determine_axis`) and
`_resolve_embedding_key` at the container. Signatures do not change, each body
opens with `container = _container(data)`. Behavior for AnnData must be
byte-identical; the existing suite is the guard.

**4. Plot guards.** Widen `isinstance(data, AnnData)` to accept MuData across
the container-native list. Mostly one-line changes at the top of each function,
plus the `AnnData | MuData` type annotation and a docstring line.

**5. `modality=` parameter.** Add to the per-modality functions, threaded into
`util/dendrogram.py`, `util/markers.py`,
`single/differential/utilities.py`, `single/heatmap/utilities.py` via
`select_modality`. Raise listing `modality_names(data)` when it is needed and
absent. Docstrings must stay backend-agnostic per
`feedback_docs_no_anndata_internals`: describe it as selecting a modality's
stored results, not as reaching into `uns`.

**6. Test fixture.** `data/` is gitignored, so the fixture must be fetched like
`pbmc3k` is. `from_url` hardcodes `read_h5ad` at two return sites, so it needs a
reader hook before it can serve `.h5mu`. Add `cl.datasets.pbmc_cite()` pointing
at `https://github.com/gtca/h5xx-datasets/raw/main/datasets/minipbcite.h5mu`
(17 MB), and a session `mudata` fixture in `tests/conftest.py`.

**7. Tests.** New `tests/test_mudata.py`, plus the collision cases, which the
real fixture cannot exercise because `prot` uses the `_TotalSeqB` suffix. Build
a synthetic two-modality MuData with a shared `CD14` and with a modality missing
cells, so both the ambiguity error and the NaN alignment are covered.

**8. Bookkeeping.** Every generated function gets a row in `plans/audit_AI.md`
with the verification columns blank (`feedback_audit_ai_tracking`). CHANGELOG
under 0.59.0 as **Added**, not Breaking, since all of this is additive. Update
the notebook and docs. Mirror any new memory into the repo `MEMORY.md`
(`feedback_sync_repo_memory`).

Steps 1 and 2 are the "starting from dataframe construction" milestone and are
independently shippable: at their end `build_frame(mdata, ...)` works and every
plot still rejects MuData cleanly. Step 6 can be done in parallel with 1 to 5.

## Verification

Run against the real fixture, not just synthetic data (`feedback_test_code`).

1. Every existing test passes unchanged. This is the real gate on step 3.
2. `build_frame(mdata["rna"], axis=0)` is byte-identical to today.
3. `build_frame(mdata, axis=0, include_dimensions=2)` carries the global obs
   columns, the pulled-up `rna:*` columns, `X_WNN_UMAP1/2` and `X_MOFA_UMAP1/2`,
   and **no** `RNA1` / `PROT1` mask columns.
4. `build_frame(mdata, variable_keys=["NKG7", "CD3_TotalSeqB"])` pulls one
   column from each modality, aligned to `mdata.obs_names`.
5. `build_frame(mdata, variable_keys=["rna:NKG7"])` matches the bare form.
6. Synthetic collision: bare `"CD14"` raises `AmbiguousVariableError` naming
   both qualified forms; `"rna:CD14"` and `"prot:CD14"` return different columns.
7. Synthetic missing cells: absent observations are NaN and every present value
   sits at the right row.
8. `build_frame(mdata, axis=1)` builds from container `var` and `varm["LFs"]`.
9. `cl.umap(mdata, keys="rna:CD14")` and `cl.dimensional(mdata,
   dimensions="wnn_umap", keys="leiden_wnn")` render, checked through the
   existing browser-render assertions.
10. `cl.dotplot(mdata, keys=["NKG7", "CD3_TotalSeqB"], group_by="leiden_wnn")`
    puts both feature types on one axis.
11. `cl.volcano(mdata, modality="rna", ...)` matches
    `cl.volcano(mdata["rna"], ...)`; omitting `modality` raises listing
    `['prot', 'rna']`.
12. `cl.spatial(mdata)` raises a message naming `mdata["rna"]`.
13. `mdata.axis != 0` raises at `build_frame`.

## Out of scope

- MuData with `axis` 1 or -1 (concatenated observations, repeating
  `obs_names`), which breaks the observations-frame contract.
- Spatial plus MuData.
- Cross-modal `varm` semantics beyond the plain concatenated `var`.
- Auto-reducing a single-modality MuData, since `mdata[key]` already works.
- Renaming `anndata_observations_frame` / `anndata_variables_frame`.
