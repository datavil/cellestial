"""
Tests for multimodal (MuData) support.

Each section maps to an item in `plans/mudata.md`. The real fixture cannot
exercise feature-name collisions (the CITE-seq panel suffixes protein names) or
partially shared observations (every cell is in both modalities), so those two
cases are built synthetically here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec
from mudata import MuData
from scipy.sparse import csr_matrix

import cellestial as cl
from cellestial.frames._container import _Container, _container, _MuDataContainer
from cellestial.util.errors import (
    AmbiguousVariableError,
    KeyNotFoundError,
    UnsupportedDataTypeError,
    VariableNotFoundError,
)
from cellestial.util.utilities import _determine_axis

GENE = "NKG7"
PROTEIN = "CD3_TotalSeqB"
JOINT_GROUP = "leiden_wnn"


@pytest.fixture
def collided():
    """Two modalities sharing the variable name `CD14`."""
    rna = AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    rna.var_names = ["CD14", "A", "B"]
    rna.obs_names = list("wxyz")
    prot = AnnData(np.arange(8, dtype="float32").reshape(4, 2) * 10)
    prot.var_names = ["CD14", "C"]
    prot.obs_names = list("wxyz")
    return MuData({"rna": rna, "prot": prot})


@pytest.fixture
def partial():
    """A container whose `prot` modality is missing two of the four cells."""
    rna = AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    rna.var_names = ["A", "B", "C"]
    rna.obs_names = list("wxyz")
    prot = AnnData(np.array([[100.0], [200.0]], dtype="float32"))
    prot.var_names = ["P"]
    prot.obs_names = ["x", "z"]
    return MuData({"rna": rna, "prot": prot})


@pytest.fixture
def peaks():
    """10x Multiome shape: ATAC peak names contain colons of their own."""
    rna = AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    rna.var_names = ["GENE_A", "GENE_B", "GENE_C"]
    rna.obs_names = list("wxyz")
    atac = AnnData(np.arange(8, dtype="float32").reshape(4, 2) * 100)
    atac.var_names = ["chr1:1000-2000", "chr2:5000-6000"]
    atac.obs_names = list("wxyz")
    return MuData({"rna": rna, "atac": atac})


@pytest.fixture
def prefixed():
    """Some datasets store variable names already carrying the modality prefix."""
    rna = AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    rna.var_names = ["rna:SAMD11", "rna:PLEKHN1", "rna:NOC2L"]
    rna.obs_names = list("wxyz")
    prot = AnnData(np.arange(8, dtype="float32").reshape(4, 2) * 10)
    prot.var_names = ["prot:PCNA", "prot:VIM"]
    prot.obs_names = list("wxyz")
    return MuData({"rna": rna, "prot": prot})


@pytest.fixture
def many_modalities():
    """Multi-assay shape: many modalities, names containing spaces."""
    parts = {}
    for index, name in enumerate(["RNA", "Protein", "Drug response", "CRISPR scores"]):
        part = AnnData(np.full((4, 2), float(index), dtype="float32"))
        part.var_names = ["SASH3", f"{name}_only"]  # SASH3 collides in all four
        part.obs_names = list("wxyz")
        parts[name] = part
    return MuData(parts)


# --- container -------------------------------------------------------------


def test_container_dispatches_on_type(mudata, adata):
    assert type(_container(mudata)) is _MuDataContainer
    assert type(_container(adata)) is _Container


def test_container_rejects_unknown_type():
    with pytest.raises(UnsupportedDataTypeError):
        _container(object())


def test_modality_masks_are_not_embeddings(mudata):
    """`obsm`/`varm` carry a boolean mask per modality that is not an embedding."""
    container = _container(mudata)
    assert set(mudata.obsm).issuperset(set(mudata.mod))
    assert not set(container.observation_embeddings()) & set(mudata.mod)
    assert not set(container.variable_embeddings()) & set(mudata.mod)


def test_owns_variable_never_raises_on_collision(collided):
    """The `_is_*` predicates run in boolean context and must not raise."""
    assert _container(collided).owns_variable("CD14") is True


def test_owns_variable_rejects_metadata_column(mudata):
    """`rna:leiden` is a pulled-up metadata column, not a variable."""
    container = _container(mudata)
    assert container.owns_variable("rna:leiden") is False
    assert "rna:leiden" in container.observation_columns()


def test_bare_name_resolves_to_its_modality(mudata):
    assert _container(mudata).resolve_variable(GENE) == ("rna", GENE)
    assert _container(mudata).resolve_variable(PROTEIN) == ("prot", PROTEIN)


def test_qualified_name_resolves(mudata):
    assert _container(mudata).resolve_variable(f"rna:{GENE}") == ("rna", GENE)


def test_ambiguous_bare_name_raises_with_candidates(collided):
    with pytest.raises(AmbiguousVariableError, match="more than one modality") as excinfo:
        _container(collided).resolve_variable("CD14")
    assert "rna:CD14" in str(excinfo.value)
    assert "prot:CD14" in str(excinfo.value)


def test_qualified_name_disambiguates_a_collision(collided):
    container = _container(collided)
    assert container.fetch_variable_columns(["rna:CD14"])[0].to_list() == [0.0, 3.0, 6.0, 9.0]
    assert container.fetch_variable_columns(["prot:CD14"])[0].to_list() == [0.0, 20.0, 40.0, 60.0]


def test_unknown_variable_raises(mudata):
    with pytest.raises(VariableNotFoundError):
        _container(mudata).resolve_variable("NOT_A_GENE")


def test_unknown_modality_in_qualified_name_raises(mudata):
    with pytest.raises(VariableNotFoundError):
        _container(mudata).resolve_variable("rna:NOT_A_GENE")


def test_peak_names_containing_colons_resolve(peaks):
    """An ATAC peak name is not a `modality:variable` reference."""
    container = _container(peaks)
    assert container.owns_variable("chr1:1000-2000") is True
    assert container.resolve_variable("chr1:1000-2000") == ("atac", "chr1:1000-2000")
    # and it can still be qualified, splitting only on the first colon
    assert container.resolve_variable("atac:chr1:1000-2000") == ("atac", "chr1:1000-2000")


def test_peak_column_matches_its_source(peaks):
    column = _container(peaks).fetch_variable_columns(["chr1:1000-2000"])[0]
    direct = np.asarray(peaks.mod["atac"][:, "chr1:1000-2000"].X).reshape(-1)
    assert column.to_list() == list(direct)


def test_source_prefixed_names_resolve(prefixed):
    """
    A variable literally named `rna:SAMD11` must be reachable.

    Reading it only as `modality` + `name` makes every variable in such a
    dataset unreachable, since the modality holds no bare `SAMD11`.
    """
    container = _container(prefixed)
    assert container.owns_variable("rna:SAMD11") is True
    assert container.resolve_variable("rna:SAMD11") == ("rna", "rna:SAMD11")
    assert container.resolve_variable("prot:PCNA") == ("prot", "prot:PCNA")


def test_source_prefixed_names_build_a_frame(prefixed):
    frame = cl.build_frame(prefixed, variable_keys=["rna:SAMD11", "prot:PCNA"])
    assert {"rna:SAMD11", "prot:PCNA"} <= set(frame.columns)


def test_unknown_name_under_a_real_modality_still_reports_precisely(prefixed):
    """The fallback must not swallow the specific error for a bad qualified key."""
    with pytest.raises(VariableNotFoundError, match="modality `rna`"):
        _container(prefixed).resolve_variable("rna:NOPE")


def test_collision_across_many_modalities(many_modalities):
    with pytest.raises(AmbiguousVariableError) as excinfo:
        _container(many_modalities).resolve_variable("SASH3")
    assert "CRISPR scores:SASH3" in str(excinfo.value)


def test_modality_names_may_contain_spaces(many_modalities):
    container = _container(many_modalities)
    assert container.resolve_variable("Drug response:SASH3") == ("Drug response", "SASH3")
    assert container.resolve_variable("CRISPR scores:SASH3") == ("CRISPR scores", "SASH3")


def test_qualified_keys_select_distinct_data(many_modalities):
    """Disambiguating must actually reach different modalities, not just parse."""
    container = _container(many_modalities)
    values = [
        container.fetch_variable_columns([f"{name}:SASH3"])[0].to_list()[0]
        for name in many_modalities.mod
    ]
    assert len(set(values)) == len(values)


def test_dataframe_embeddings_are_indexable(mudata):
    """An embedding store accepts a DataFrame, which rejects positional indexing."""
    container = mudata.copy()
    container.obsm["coords"] = pd.DataFrame(
        {"x": np.arange(container.n_obs, dtype="float64"), "y": np.zeros(container.n_obs)},
        index=container.obs_names,
    )
    frame = cl.build_frame(container, axis=0, include_dimensions=2, dimension_keys=["coords"])
    assert {"COORDS1", "COORDS2"} <= set(frame.columns)
    assert frame["COORDS1"].to_list() == list(range(container.n_obs))


def test_dataframe_embeddings_are_indexable_for_anndata(adata):
    """The same store accepts a DataFrame on a plain AnnData."""
    data = AnnData(np.ones((3, 2), dtype="float32"))
    data.obsm["coords"] = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]}, index=data.obs_names
    )
    frame = cl.build_frame(data, axis=0, include_dimensions=2)
    assert frame["COORDS1"].to_list() == [1.0, 2.0, 3.0]


def test_absent_observations_align_as_nan(partial):
    """Values must land on the right rows, with absent cells left as NaN."""
    column = _container(partial).fetch_variable_columns(["P"])[0]
    values = column.to_numpy()
    assert np.isnan(values[0])  # cell "w" is not in prot
    assert values[1] == 100.0  # cell "x"
    assert np.isnan(values[2])  # cell "y" is not in prot
    assert values[3] == 200.0  # cell "z"


def test_variable_columns_keep_source_dtype(mudata):
    """A float64 fill would double the memory of every fetched column."""
    columns = _container(mudata).fetch_variable_columns([GENE, PROTEIN])
    assert all(str(column.dtype) == "Float32" for column in columns)


def test_variable_column_matches_direct_read(mudata):
    column = _container(mudata).fetch_variable_columns([GENE])[0]
    direct = np.asarray(mudata.mod["rna"][:, GENE].X).reshape(-1)
    assert np.allclose(column.to_numpy(), direct)


def test_integer_counts_widen_for_nan(partial):
    """Integer matrices cannot hold NaN, so the fill dtype must widen."""
    partial.mod["prot"].X = partial.mod["prot"].X.astype("int32")
    column = _container(partial).fetch_variable_columns(["P"])[0]
    assert column.dtype.is_float()
    assert np.isnan(column.to_numpy()[0])


# --- frames ----------------------------------------------------------------


def test_frame_uses_container_metadata(mudata):
    frame = cl.build_frame(mudata, axis=0)
    assert JOINT_GROUP in frame.columns  # container-level column
    assert "rna:leiden" in frame.columns  # pulled up from the modality


def test_frame_carries_joint_embeddings(mudata):
    frame = cl.build_frame(mudata, axis=0, include_dimensions=2)
    assert {"X_WNN_UMAP1", "X_WNN_UMAP2", "X_MOFA_UMAP1", "X_MOFA_UMAP2"} <= set(frame.columns)


def test_frame_omits_modality_masks(mudata):
    """The `obsm` masks must not surface as `RNA1` / `PROT1` columns."""
    frame = cl.build_frame(mudata, axis=0, include_dimensions=True)
    assert not [column for column in frame.columns if column.upper() in {"RNA1", "PROT1"}]


def test_frame_pulls_across_modalities(mudata):
    frame = cl.build_frame(mudata, variable_keys=[GENE, PROTEIN])
    assert {GENE, PROTEIN} <= set(frame.columns)
    assert frame.height == mudata.n_obs


def test_qualified_and_bare_keys_agree(mudata):
    bare = cl.build_frame(mudata, variable_keys=[GENE])[GENE].to_list()
    qualified = cl.build_frame(mudata, variable_keys=[f"rna:{GENE}"])[f"rna:{GENE}"].to_list()
    assert bare == qualified


def test_variables_frame_from_container(mudata):
    frame = cl.build_frame(mudata, axis=1)
    assert frame.height == mudata.n_vars
    assert "highly_variable" in frame.columns


def test_non_joint_axis_is_rejected():
    rna = AnnData(np.ones((3, 2), dtype="float32"))
    rna.obs_names = list("abc")
    prot = AnnData(np.ones((3, 2), dtype="float32"))
    prot.obs_names = list("abc")
    concatenated = MuData({"rna": rna, "prot": prot}, axis=1)
    with pytest.raises(ValueError, match="axis=0"):
        cl.build_frame(concatenated, axis=0)


def test_anndata_frame_is_unchanged(mudata):
    """Passing a modality directly must behave exactly as an AnnData always did."""
    frame = cl.build_frame(mudata.mod["rna"], axis=0, variable_keys=[GENE])
    assert GENE in frame.columns
    assert frame.height == mudata.mod["rna"].n_obs


# --- plots -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("umap_categorical", lambda data: cl.umap(data, key=JOINT_GROUP)),
        ("umap_gene", lambda data: cl.umap(data, key=GENE)),
        ("umap_protein", lambda data: cl.umap(data, key=PROTEIN)),
        ("umap_qualified", lambda data: cl.umap(data, key=f"rna:{GENE}")),
        (
            "dimensional_wnn",
            lambda data: cl.dimensional(data, dimensions="wnn_umap", key="celltype"),
        ),
        ("dimensional_mofa", lambda data: cl.dimensional(data, dimensions="mofa", key="celltype")),
        ("xyplot", lambda data: cl.xyplot(data, x=GENE, y=PROTEIN, color="celltype")),
        ("violin", lambda data: cl.violin(data, key=GENE, group_by=JOINT_GROUP)),
        ("boxplot", lambda data: cl.boxplot(data, key=PROTEIN, group_by="celltype")),
        ("histogram", lambda data: cl.histogram(data, key=GENE)),
        ("expression", lambda data: cl.expression(data, key=GENE, group_by=JOINT_GROUP)),
        ("ridge", lambda data: cl.ridge(data, key=GENE, group_by="celltype")),
        ("heatmap", lambda data: cl.heatmap(data, keys=[GENE, PROTEIN], group_by=JOINT_GROUP)),
        ("dotplot", lambda data: cl.dotplot(data, keys=[GENE, PROTEIN], group_by=JOINT_GROUP)),
        (
            "matrixplot",
            lambda data: cl.matrixplot(data, keys=[GENE, PROTEIN], group_by=JOINT_GROUP),
        ),
        (
            "stacked_violin",
            lambda data: cl.stacked_violin(data, keys=[GENE, PROTEIN], group_by=JOINT_GROUP),
        ),
    ],
)
def test_single_plots_accept_a_container(mudata, name, call):
    assert isinstance(call(mudata), PlotSpec), name


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("umaps", lambda data: cl.umaps(data, keys=[GENE, PROTEIN])),
        (
            "dimensionals",
            lambda data: cl.dimensionals(data, dimensions="wnn_umap", keys=[GENE, PROTEIN]),
        ),
        ("violins", lambda data: cl.violins(data, keys=[GENE, PROTEIN], group_by=JOINT_GROUP)),
        ("boxplots", lambda data: cl.boxplots(data, keys=[GENE, PROTEIN], group_by="celltype")),
        ("histograms", lambda data: cl.histograms(data, keys=[GENE, PROTEIN])),
        ("ridges", lambda data: cl.ridges(data, keys=[GENE, PROTEIN], group_by="celltype")),
        (
            "expressions",
            lambda data: cl.expressions(data, keys=[GENE, PROTEIN], group_by=JOINT_GROUP),
        ),
        ("xyplots", lambda data: cl.xyplots(data, x=GENE, y=PROTEIN, colors=["celltype"])),
        (
            "annotated_heatmap",
            lambda data: cl.annotated_heatmap(data, keys=[GENE, PROTEIN], group_by=JOINT_GROUP),
        ),
    ],
)
def test_grid_plots_accept_a_container(mudata, name, call):
    assert isinstance(call(mudata), SupPlotsSpec), name


def test_mixed_modality_keys_share_one_axis(mudata):
    """The point of container support: genes and proteins on a single axis."""
    plot = cl.dotplot(mudata, keys=[GENE, PROTEIN], group_by=JOINT_GROUP)
    html = plot.to_html()
    assert GENE in html
    assert PROTEIN in html


def test_ambiguous_key_raises_from_a_plot(collided):
    with pytest.raises(AmbiguousVariableError):
        cl.build_frame(collided, variable_keys=["CD14"])


# --- modality selection ----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("volcano", lambda data, **kw: cl.volcano(data, group="0", **kw)),
        ("volcanos", lambda data, **kw: cl.volcanos(data, groups=["0", "1"], **kw)),
        ("markers", lambda data, **kw: cl.markers(data, **kw)),
        ("elbow", lambda data, **kw: cl.elbow(data, **kw)),
        ("highest", lambda data, **kw: cl.highest_expressed_genes(data, n=5, **kw)),
        ("marker_genes", lambda data, **kw: cl.marker_genes(data, n_genes=2, **kw)),
        ("marker_genes_dict", lambda data, **kw: cl.marker_genes_dict(data, n_genes=2, **kw)),
    ],
)
def test_modality_selects_stored_results(mudata, name, call):
    assert call(mudata, modality="rna") is not None, name


def test_marker_genes_match_the_modality(mudata):
    """Selecting a modality must give exactly what that modality alone gives."""
    assert cl.marker_genes(mudata, modality="rna", n_genes=2) == cl.marker_genes(
        mudata.mod["rna"], n_genes=2
    )


def test_ambiguous_key_uses_the_grouping_column_to_pick_an_axis(mudata):
    """
    `total_counts` is stored per observation *and* per variable.

    On its own the key cannot decide the axis, but grouping by an
    observation-only column settles it. Picking the variables axis here built a
    variables frame, so the grouping column was missing from it entirely.
    """
    plot = cl.violin(mudata, key="rna:total_counts", fill="leiden")
    frame = cl.retrieve(plot)
    assert "leiden" in frame.columns
    assert "rna:total_counts" in frame.columns


def test_ambiguous_key_axis_is_unchanged_without_a_hint(mudata, adata):
    """With nothing to break the tie the variables axis still wins, as before."""
    assert _determine_axis(mudata, ["rna:total_counts"]) == 1
    assert _determine_axis(adata, ["total_counts"]) == 1


def test_variable_axis_companion_still_selects_variables(mudata):
    assert _determine_axis(mudata, ["rna:total_counts"], companions=["highly_variable"]) == 1


def test_ambiguous_key_disambiguation_reaches_anndata(adata):
    """The ambiguity was never multimodal-specific."""
    assert isinstance(cl.violin(adata, key="total_counts", fill="leiden"), PlotSpec)


def test_bare_metadata_name_suggests_the_qualified_form(mudata):
    """
    A modality's columns are prefixed on the container, so the bare name fails.

    Reaching for `celltype` when only `rna:celltype` exists is the natural
    mistake, and previously surfaced as a raw polars `ColumnNotFoundError`
    listing the narrowed frame.
    """
    with pytest.raises(KeyNotFoundError, match="Did you mean") as excinfo:
        cl.violin(mudata, key=GENE, group_by="n_genes_by_counts")
    assert "rna:n_genes_by_counts" in str(excinfo.value)


def test_unknown_metadata_name_does_not_invent_a_suggestion(mudata):
    with pytest.raises(KeyNotFoundError) as excinfo:
        cl.violin(mudata, key=GENE, group_by="not_a_column")
    assert "Did you mean" not in str(excinfo.value)


def test_group_by_validation_reaches_anndata_too(mudata):
    """The raw polars error was not multimodal-specific."""
    with pytest.raises(KeyNotFoundError, match="not a column in the data"):
        cl.violin(mudata.mod["rna"], key=GENE, group_by="not_a_column")


def test_modality_is_required_when_ambiguous(mudata):
    with pytest.raises(KeyNotFoundError, match="modality"):
        cl.volcano(mudata, group="0")


def test_unknown_modality_raises_with_candidates(mudata):
    with pytest.raises(KeyNotFoundError, match="Unknown modality") as excinfo:
        cl.volcano(mudata, group="0", modality="nope")
    assert "rna" in str(excinfo.value)


def test_modality_is_rejected_for_anndata(mudata):
    with pytest.raises(KeyNotFoundError, match="no modalities"):
        cl.volcano(mudata.mod["rna"], group="0", modality="rna")


@pytest.fixture
def ranked(mudata):
    """
    Container whose `rna` ranking matches the genes it actually holds.

    The shipped fixture was subset after `rank_genes_groups` ran, so its stored
    ranking names genes the modality no longer has.
    """
    import scanpy as sc

    # copy: `mudata` is session-scoped, and re-ranking it would leak into other tests
    container = mudata.copy()
    sc.tl.rank_genes_groups(
        container.mod["rna"], groupby="leiden", key_added="rank_genes_groups", use_raw=False
    )
    return container


@pytest.mark.parametrize("plot", [cl.heatmap, cl.dotplot, cl.stacked_violin])
def test_markers_group_by_the_clustering_the_ranking_used(ranked, plot):
    """
    The derived `group_by` must be read as the container names it.

    A ranking stores its group column under the modality's own name (`leiden`).
    The container carries both a global `leiden` (9 joint clusters) and
    `rna:leiden` (14 RNA clusters), which disagree on most cells, so taking the
    stored name at face value silently groups by the wrong clustering.
    """
    frame = cl.retrieve(plot(ranked, markers=True, n_genes=2, modality="rna"))
    column = next(name for name in frame.columns if "leiden" in name.lower())
    assert column == "rna:leiden"
    assert frame[column].n_unique() == ranked.obs["rna:leiden"].nunique()


def test_markers_accept_the_container_group_column(ranked):
    """`rna:leiden` and the stored `leiden` name the same column."""
    assert isinstance(
        cl.heatmap(ranked, markers=True, n_genes=2, modality="rna", group_by="rna:leiden"),
        PlotSpec,
    )


def test_markers_still_reject_a_mismatched_group_column(ranked):
    with pytest.raises(ValueError, match="does not match"):
        cl.heatmap(ranked, markers=True, n_genes=2, modality="rna", group_by="celltype")


def test_markers_and_dendrogram_round_trip(ranked):
    """Container name down to the modality for clustering, back up for the frame."""
    assert isinstance(
        cl.dotplot(ranked, markers=True, n_genes=2, modality="rna", dendrogram=True),
        PlotSpec,
    )


def test_dendrogram_translates_a_prefixed_group_column(mudata):
    """`rna:leiden` at the container is `leiden` inside the modality."""
    plot = cl.dotplot(mudata, keys=[GENE], group_by="rna:leiden", dendrogram=True, modality="rna")
    assert isinstance(plot, PlotSpec)


def test_dendrogram_rejects_a_container_only_group_column(mudata):
    """`leiden_wnn` exists only at the container, so no modality can cluster on it."""
    with pytest.raises(KeyNotFoundError, match="dendrogram"):
        cl.dotplot(mudata, keys=[GENE], group_by=JOINT_GROUP, dendrogram=True, modality="rna")


# --- edge cases ------------------------------------------------------------


@pytest.fixture
def sparse():
    """Sparse matrices, which real single-cell data almost always uses."""
    rna = AnnData(csr_matrix(np.arange(12, dtype="float32").reshape(4, 3)))
    rna.var_names = ["A", "B", "C"]
    rna.obs_names = list("wxyz")
    prot = AnnData(csr_matrix(np.array([[7.0], [9.0]], dtype="float32")))
    prot.var_names = ["P"]
    prot.obs_names = ["x", "z"]
    return MuData({"rna": rna, "prot": prot})


def test_sparse_matrices_are_densified(sparse):
    assert _container(sparse).fetch_variable_columns(["A"])[0].to_list() == [0.0, 3.0, 6.0, 9.0]


def test_sparse_and_absent_observations_together(sparse):
    """Both the sparse branch and the alignment branch on one column."""
    values = _container(sparse).fetch_variable_columns(["P"])[0].to_numpy()
    assert np.isnan(values[0])  # cell "w" is not in prot
    assert values[1] == 7.0  # cell "x"
    assert np.isnan(values[2])  # cell "y" is not in prot
    assert values[3] == 9.0  # cell "z"


def test_single_modality_needs_no_modality_argument(sparse):
    """With one modality there is nothing to disambiguate."""
    solo = MuData({"rna": sparse.mod["rna"]})
    assert isinstance(_container(solo).select_modality(None), AnnData)


def test_duplicate_variable_name_inside_a_modality_raises(collided):
    """
    A name matching two columns has no single set of values.

    Slicing on it returned an interleaving of both columns, silently.
    """
    duplicated = AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    duplicated.var_names = ["A", "A", "B"]
    duplicated.obs_names = list("wxyz")
    container = MuData({"rna": duplicated, "prot": collided.mod["prot"]})
    with pytest.raises(AmbiguousVariableError, match="more than one variable"):
        _container(container).fetch_variable_columns(["A"])


def test_duplicate_variable_name_raises_for_anndata_too(adata):
    """The same guard, where the failure used to be an opaque pandas error."""
    duplicated = AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    duplicated.var_names = ["A", "A", "B"]
    duplicated.obs_names = list("wxyz")
    with pytest.raises(AmbiguousVariableError, match="more than one variable"):
        cl.build_frame(duplicated, variable_keys=["A"])


def test_unique_names_alongside_a_duplicate_still_resolve(collided):
    duplicated = AnnData(np.arange(12, dtype="float32").reshape(4, 3))
    duplicated.var_names = ["A", "A", "B"]
    duplicated.obs_names = list("wxyz")
    container = _container(MuData({"rna": duplicated, "prot": collided.mod["prot"]}))
    assert container.fetch_variable_columns(["B"])[0].to_list() == [2.0, 5.0, 8.0, 11.0]


def test_modality_with_no_observations_yields_all_nan(sparse):
    ghost = AnnData(np.zeros((0, 1), dtype="float32"))
    ghost.var_names = ["G"]
    ghost.obs_names = []
    container = _container(MuData({"rna": sparse.mod["rna"], "ghost": ghost}))
    assert all(np.isnan(v) for v in container.fetch_variable_columns(["G"])[0].to_numpy())


def test_integer_counts_widen_to_a_float_that_holds_them(sparse):
    """float32 cannot represent int64 counts exactly beyond 2**24."""
    counts = AnnData(np.arange(8, dtype="int64").reshape(4, 2))
    counts.var_names = ["Q1", "Q2"]
    counts.obs_names = list("wxyz")
    container = _container(MuData({"rna": sparse.mod["rna"], "cnt": counts}))
    assert str(container.fetch_variable_columns(["Q1"])[0].dtype) == "Float64"


def test_variable_axis_dataframe_embeddings(mudata):
    """`varm` takes a DataFrame just as `obsm` does."""
    container = mudata.copy()
    container.varm["loadings"] = pd.DataFrame(
        {"a": np.arange(container.n_vars, dtype="float64"), "b": np.zeros(container.n_vars)},
        index=container.var_names,
    )
    frame = cl.build_frame(container, axis=1, include_dimensions=2, dimension_keys=["loadings"])
    assert {"LOADINGS1", "LOADINGS2"} <= set(frame.columns)


@pytest.mark.parametrize("key", ["", "rna:", ":NKG7", "rna::NKG7"])
def test_malformed_keys_are_rejected_without_crashing(mudata, key):
    container = _container(mudata)
    assert container.owns_variable(key) is False
    with pytest.raises(VariableNotFoundError):
        container.resolve_variable(key)


def test_variable_named_like_a_modality(sparse):
    """A variable may share a modality's name; the literal match still wins."""
    rna = AnnData(np.arange(8, dtype="float32").reshape(4, 2))
    rna.var_names = ["prot", "X"]  # `prot` is also a modality name
    rna.obs_names = list("wxyz")
    container = _container(MuData({"rna": rna, "prot": sparse.mod["prot"]}))
    assert container.resolve_variable("prot") == ("rna", "prot")


def test_non_joint_axis_minus_one_is_rejected(sparse):
    """`axis=-1` shares both axes, which is not the supported layout either."""
    rna = sparse.mod["rna"]
    other = AnnData(np.ones((4, 2), dtype="float32"))
    other.var_names = ["Z1", "Z2"]
    other.obs_names = list("wxyz")
    with pytest.raises(ValueError, match="axis=0"):
        cl.build_frame(MuData({"rna": rna, "other": other}, axis=-1), axis=0)


def test_empty_variable_keys_short_circuits(mudata):
    """An empty request must not trigger a full slice."""
    frame = cl.build_frame(mudata, axis=0, variable_keys=[], metadata_columns=[])
    assert frame.height == mudata.n_obs


def test_metadata_and_variable_sharing_a_name_prefers_metadata(mudata):
    """`_collect_aes_columns` tests metadata first, so a column wins over a variable."""
    container = mudata.copy()
    container.obs[GENE] = np.arange(container.n_obs, dtype="float64")
    frame = cl.build_frame(container, axis=0, variable_keys=[GENE])
    assert frame[GENE].to_list() == list(range(container.n_obs))


# --- out of scope ----------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda data: cl.spatial(data),
        lambda data: cl.spatials(data, keys=["celltype"]),
    ],
)
def test_spatial_rejects_a_container(mudata, call):
    with pytest.raises(UnsupportedDataTypeError, match="single modality"):
        call(mudata)


def test_spatial_colours_across_modalities_via_a_container_frame():
    """
    The supported route for the one thing native support would have added.

    Coordinates and the tissue image come from the modality, so there is no
    question of whose registration wins, while the values come from a container
    frame and so may belong to any modality.
    """
    rna = AnnData(np.arange(20, dtype="float32").reshape(5, 4))
    rna.var_names = ["G1", "G2", "G3", "G4"]
    rna.obs_names = list("vwxyz")
    rna.obsm["spatial"] = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    rna.uns["spatial"] = {
        "lib1": {
            "images": {"lowres": np.zeros((8, 8, 3), dtype="uint8")},
            "scalefactors": {"tissue_lowres_scalef": 1.0, "spot_diameter_fullres": 1.0},
        }
    }
    prot = AnnData(np.array([[1.0], [5.0], [9.0], [13.0], [17.0]], dtype="float32"))
    prot.var_names = ["CD3"]
    prot.obs_names = list("vwxyz")
    container = MuData({"rna": rna, "prot": prot})

    plot = cl.spatial(
        container.mod["rna"],
        key="prot:CD3",
        frame=cl.build_frame(container, axis=0, variable_keys=["prot:CD3"]),
    )
    frame = cl.retrieve(plot)
    assert frame["prot:CD3"].to_list() == [1.0, 5.0, 9.0, 13.0, 17.0]
    assert {"spatial_x", "spatial_y"} <= set(frame.columns)


# --- public surface --------------------------------------------------------


def test_container_is_not_public():
    """The container is internal and must never reach the package surface."""
    assert not hasattr(cl, "_container")
    assert not hasattr(cl.frames, "_Container")
    leaked = [name for name in cl.frames.__all__ if "ontainer" in name]
    assert leaked == []
