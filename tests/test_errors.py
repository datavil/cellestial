"""Error-path tests for public misuse: singular/plural, missing keys, wrong types."""

import numpy as np
import pytest

import cellestial as cl
from cellestial.util.errors import (
    EmptySelectionError,
    InvalidArgumentTypeError,
    InvalidPlotSpecError,
    KeyNotFoundError,
    MissingAestheticError,
    NonCategoricalColumnError,
    UnsupportedDataTypeError,
)
from tests.conftest import GROUP_KEY

# ---- Category 1: singular functions guide users to the plural ----


@pytest.mark.parametrize(
    ("fn", "plural"),
    [
        (cl.umap, "umaps"),
        (cl.tsne, "tsnes"),
        (cl.pca, "pcas"),
        (cl.dimensional, "dimensionals"),
        (cl.expression, "expressions"),
    ],
)
def test_singular_sequence_key_points_to_plural(adata, fn, plural):
    with pytest.raises(TypeError, match=plural):
        fn(adata, ["CD14", "MS4A1"])


def test_ridge_sequence_key_points_to_plural(adata, group_key):
    with pytest.raises(TypeError, match="ridges"):
        cl.ridge(adata, ["CD14", "MS4A1"], group_key)


# ---- Category 2: missing key says where Cellestial looked ----


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne, cl.dimensional])
def test_missing_key_reports_where_it_looked(adata, fn):
    with pytest.raises(KeyNotFoundError, match="observation metadata and variable"):
        fn(adata, "NOT_A_REAL_KEY")


# ---- Category 3: wrong data type says expected and received ----


@pytest.mark.parametrize("fn", [cl.umap, cl.dimensional, cl.heatmap, cl.elbow])
def test_wrong_data_type_reports_expected_and_received(fn):
    with pytest.raises(UnsupportedDataTypeError, match="Expected AnnData.*received `str`"):
        fn("not anndata")


def test_volcano_wrong_data_type():
    with pytest.raises(UnsupportedDataTypeError, match="Expected AnnData.*received `str`"):
        cl.volcano("not anndata", "B Cells")


def test_build_frame_wrong_type_mentions_spatialdata():
    with pytest.raises(UnsupportedDataTypeError, match="AnnData or SpatialData"):
        cl.build_frame("not anndata", axis=0)


# ---- Category 4: unknown `groups` / `drop` names name what is available ----


@pytest.mark.parametrize(
    ("fn", "kwargs"),
    [
        (cl.umap, {"key": GROUP_KEY}),
        (cl.pca, {"key": GROUP_KEY}),
        (cl.tsne, {"key": GROUP_KEY}),
        (cl.dimensional, {"key": GROUP_KEY}),
        (cl.violin, {"key": "CD3D", "group_by": GROUP_KEY}),
        (cl.boxplot, {"key": "CD3D", "group_by": GROUP_KEY}),
        (cl.histogram, {"key": "CD3D", "group_by": GROUP_KEY}),
        (cl.ridge, {"key": "CD3D", "group_by": GROUP_KEY}),
    ],
)
def test_unknown_group_reports_available(adata, fn, kwargs):
    with pytest.raises(KeyNotFoundError, match=r"`groups` \['Nope'\].*Available:"):
        fn(adata, groups=["Nope"], **kwargs)


def test_unknown_group_caught_alongside_a_valid_one(adata, group_key):
    """A partial typo must not quietly produce a plot missing that condition."""
    with pytest.raises(KeyNotFoundError, match=r"\['NoSuchGroup'\]"):
        cl.umap(adata, key=group_key, groups=["Lymphocytes", "NoSuchGroup"])


def test_unknown_drop_reports_available(adata, group_key):
    with pytest.raises(KeyNotFoundError, match=r"`drop` \['Lymphcytes'\].*Available:"):
        cl.violin(adata, key="CD3D", group_by=group_key, drop=["Lymphcytes"])


def test_group_removed_by_nonfinite_coordinates_is_not_unknown(adata, group_key):
    """A group whose cells all lack coordinates still exists, so naming it must not raise."""
    missing = sorted(adata.obs[group_key].dropna().unique())[0]
    adata = adata.copy()
    umap = adata.obsm["X_umap"].copy()
    umap[(adata.obs[group_key] == missing).to_numpy()] = np.nan
    adata.obsm["X_umap"] = umap
    plot = cl.umap(adata, key=group_key, groups=[missing])
    assert cl.retrieve(plot).height == 0


def test_highlight_unknown_group_reports_available(adata, cluster_key):
    with pytest.raises(
        KeyNotFoundError, match=r"`groups` \['leidn'\] not found in color `leiden`.*Available:"
    ):
        cl.umap(adata, key=cluster_key) + cl.highlight(["0", "leidn"])


def test_highlight_without_color_key_says_how_to_fix(adata):
    with pytest.raises(MissingAestheticError, match=r"no color mapping.*key='leiden'"):
        cl.umap(adata) + cl.highlight("0")


def test_highlight_continuous_color_says_how_to_fix(adata):
    with pytest.raises(
        NonCategoricalColumnError, match=r"`CD14` holds numeric values.*categorical key"
    ):
        cl.umap(adata, key="CD14") + cl.highlight("0")


def test_highlight_empty_groups_fails_early():
    with pytest.raises(EmptySelectionError, match=r"`groups` is empty"):
        cl.highlight([])


def test_highlight_on_a_grid_points_to_singular(adata, cluster_key):
    with pytest.raises(InvalidPlotSpecError, match=r"adds to a single plot.*SupPlotsSpec"):
        cl.umaps(adata, [cluster_key]) + cl.highlight("0")


# ---- Renamed embedding selector: `use_key` and `basis` point to `embedding_key` ----

_EMBEDDING_CALLS = [
    (cl.umap, ("leiden",)),
    (cl.pca, ("leiden",)),
    (cl.tsne, ("leiden",)),
    (cl.dimensional, ("leiden",)),
    (cl.expression, ("CD14",)),
    (cl.umaps, (["leiden"],)),
    (cl.pcas, (["leiden"],)),
    (cl.tsnes, (["leiden"],)),
    (cl.dimensionals, (["leiden"],)),
    (cl.expressions, (["CD14"],)),
    (cl.embedding_density, ()),
    (cl.blend, (["CD14", "MS4A1"],)),
]


@pytest.mark.parametrize(("fn", "args"), _EMBEDDING_CALLS)
def test_use_key_points_to_embedding_key(adata, fn, args):
    with pytest.raises(
        InvalidArgumentTypeError, match="`use_key` was renamed to `embedding_key`"
    ) as info:
        fn(adata, *args, use_key="X_umap")
    assert "embedding_key='X_umap'" in str(info.value)
    assert isinstance(info.value, TypeError)


@pytest.mark.parametrize(("fn", "args"), _EMBEDDING_CALLS)
def test_basis_points_to_embedding_key(adata, fn, args):
    with pytest.raises(InvalidArgumentTypeError, match="to choose a stored embedding") as info:
        fn(adata, *args, basis="X_umap")
    assert "embedding_key='X_umap'" in str(info.value)


# ---- `add_keys`: unknown names raise instead of vanishing ----

_ADD_KEYS_CALLS = [
    (cl.umap, ("leiden",)),
    (cl.pca, ("leiden",)),
    (cl.tsne, ("leiden",)),
    (cl.dimensional, ("leiden",)),
    (cl.expression, ("CD14",)),
    (cl.umaps, (["leiden"],)),
    (cl.pcas, (["leiden"],)),
    (cl.tsnes, (["leiden"],)),
    (cl.dimensionals, (["leiden"],)),
    (cl.expressions, (["CD14"],)),
]


@pytest.mark.parametrize(("fn", "args"), _ADD_KEYS_CALLS)
def test_add_keys_unknown_name_raises(adata, fn, args):
    with pytest.raises(KeyNotFoundError, match="`add_keys` names not found") as info:
        fn(adata, *args, add_keys=["CD14", "LYZZ"])
    message = str(info.value)
    assert "LYZZ" in message
    assert "'LYZ'" in message
    assert "CD14" not in message.split("\n")[0]


def test_add_keys_accepts_metadata_and_genes(adata):
    frame = cl.retrieve(cl.umap(adata, "leiden", add_keys=["CD14", "n_genes_by_counts"]))
    assert {"CD14", "n_genes_by_counts"} <= set(frame.columns)


def test_add_keys_accepts_column_of_passed_frame(adata):
    frame = cl.build_frame(adata, variable_keys=["CD14"], include_dimensions=2).with_columns(
        extra=1.0
    )
    plot = cl.umap(adata, "CD14", frame=frame, add_keys=["extra"])
    assert "extra" in cl.retrieve(plot).columns


def test_add_keys_unknown_name_raises_mudata(mudata):
    with pytest.raises(KeyNotFoundError, match="prot:CD3_TotalSeqBB"):
        cl.umap(mudata, "leiden", add_keys=["prot:CD3_TotalSeqBB"])


def _spatial_anndata():
    import pandas as pd
    from anndata import AnnData

    rng = np.random.default_rng(0)
    data = AnnData(
        X=rng.random((3, 2)).astype("float32"),
        obs=pd.DataFrame({"score": [1.0, 2.0, 3.0]}, index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["GENE_A", "GENE_B"]),
    )
    data.obsm["spatial"] = rng.random((3, 2)).astype("float32")
    return data


@pytest.mark.parametrize("plural", [False, True])
def test_add_keys_unknown_name_raises_spatial(plural):
    data = _spatial_anndata()
    with pytest.raises(KeyNotFoundError, match="GENE_Z") as info:
        if plural:
            cl.spatials(data, ["score"], image=False, add_keys=["GENE_Z"])
        else:
            cl.spatial(data, "score", image=False, add_keys=["GENE_Z"])
    assert "Did you mean" in str(info.value)
