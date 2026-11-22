"""Error-path tests for public misuse: singular/plural, missing keys, wrong types."""

import numpy as np
import pytest

import cellestial as cl
from cellestial.util.errors import (
    EmptySelectionError,
    InvalidArgumentError,
    InvalidArgumentTypeError,
    InvalidPlotSpecError,
    KeyNotFoundError,
    MissingAestheticError,
    NonCategoricalColumnError,
    UnknownParameterError,
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


@pytest.mark.parametrize(
    "call",
    [
        lambda data, scale: cl.ridge(data, "CD14", GROUP_KEY, scale=scale),
        lambda data, scale: cl.ridges(data, ["CD14", "MS4A1"], GROUP_KEY, scale=scale),
    ],
    ids=["ridge", "ridges"],
)
@pytest.mark.parametrize(
    ("scale", "error", "parts"),
    [
        ("width", InvalidArgumentTypeError, ["'width'", "str", "overlap", "not a width method"]),
        (True, InvalidArgumentTypeError, ["True", "bool", "positive number", "2.0"]),
        (None, InvalidArgumentTypeError, ["None", "positive number"]),
        (0, InvalidArgumentError, ["got 0", "above 0"]),
        (-1.5, InvalidArgumentError, ["got -1.5", "above 0"]),
        (float("nan"), InvalidArgumentError, ["got nan", "positive number"]),
        (float("inf"), InvalidArgumentError, ["got inf", "positive number"]),
        (10**400, InvalidArgumentError, ["got 1000", "positive number"]),
    ],
)
def test_ridge_scale_rejects_non_positive_numbers(adata, call, scale, error, parts):
    with pytest.raises(error) as info:
        call(adata, scale)
    message = str(info.value)
    assert "`scale`" in message
    for part in parts:
        assert part in message


@pytest.mark.parametrize("scale", [1, 0.5, np.float64(3.0), np.int64(2)])
def test_ridge_scale_accepts_positive_numbers(adata, group_key, scale):
    cl.ridge(adata, "CD14", group_key, scale=scale)
    cl.ridges(adata, ["CD14", "MS4A1"], group_key, scale=scale)



@pytest.mark.parametrize(
    ("sort_by", "parts"),
    [
        ("avg_exp", ["'avg_exp'", "percent_expressed", "mean_expression", "`sort_by`"]),
        (["mean_expresion"], ["Did you mean 'mean_expression'?", "`sort_by`"]),
    ],
)
def test_dotplot_unknown_sort_by_column(adata, group_key, sort_by, parts):
    with pytest.raises(KeyNotFoundError) as info:
        cl.dotplot(adata, keys=["CD14", "MS4A1"], group_by=group_key, sort_by=sort_by)
    message = str(info.value)
    for part in parts:
        assert part in message


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


# ---- stored ranking: `ranking_key` replaces `key` and string `markers` ----


@pytest.fixture(scope="module")
def ranked_pbmc(adata):
    """pbmc3k with a ranking on `leiden_res_0.02` stored under a custom key."""
    import scanpy as sc

    data = adata.copy()
    sc.tl.rank_genes_groups(data, groupby="leiden_res_0.02", key_added="rank_leiden")
    return data


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("markers", lambda data: cl.markers(data, key="rank_leiden")),
        ("volcano", lambda data: cl.volcano(data, "0", key="rank_leiden")),
        ("volcanos", lambda data: cl.volcanos(data, ["0", "1"], key="rank_leiden")),
    ],
)
def test_old_ranking_key_parameter_is_rejected(ranked_pbmc, name, call):
    """`key=` is not swallowed by the geom kwargs; it points to `ranking_key`."""
    with pytest.raises(UnknownParameterError) as error:
        call(ranked_pbmc)
    message = str(error.value)
    assert f"`key` is not a parameter of `{name}`" in message
    assert "ranking_key='rank_leiden'" in message


@pytest.mark.parametrize("fn", [cl.marker_genes, cl.marker_genes_dict])
def test_marker_genes_old_key_parameter_is_rejected(ranked_pbmc, fn):
    """The marker-name helpers take no extra keywords, so `key=` fails loudly."""
    with pytest.raises(TypeError, match="key"):
        fn(ranked_pbmc, key="rank_leiden")


@pytest.mark.parametrize(
    "fn", [cl.heatmap, cl.matrixplot, cl.dotplot, cl.stacked_violin, cl.tracksplot]
)
def test_string_markers_points_to_ranking_key(ranked_pbmc, fn):
    """`markers='x'` raises and says to pass `markers=True, ranking_key='x'`."""
    with pytest.raises(InvalidArgumentTypeError) as error:
        fn(ranked_pbmc, markers="rank_leiden")
    message = str(error.value)
    assert "`markers` must be True or False" in message
    assert "markers=True, ranking_key='rank_leiden'" in message


@pytest.mark.parametrize(
    "call",
    [
        lambda data: cl.dotplot(data, markers=True, ranking_key="rank_leidn"),
        lambda data: cl.markers(data, ranking_key="rank_leidn"),
        lambda data: cl.marker_genes(data, ranking_key="rank_leidn"),
    ],
)
def test_unknown_ranking_key_suggests_a_close_match(ranked_pbmc, call):
    """A misspelled `ranking_key` names the parameter and the closest stored key."""
    with pytest.raises(KeyNotFoundError) as error:
        call(ranked_pbmc)
    message = str(error.value)
    assert "ranking_key='rank_leidn'" in message
    assert "rank_leiden" in message
    assert "adata" not in message
