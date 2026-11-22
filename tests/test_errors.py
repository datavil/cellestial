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


# ---- Renamed spatial flag: `greyscale` points to `grayscale` ----


@pytest.mark.parametrize(("fn", "key"), [(cl.spatial, "leiden"), (cl.spatials, ["leiden"])])
def test_greyscale_points_to_grayscale(adata, fn, key):
    with pytest.raises(
        UnknownParameterError, match="`greyscale` was renamed to `grayscale`"
    ) as info:
        fn(adata, key, greyscale=True)
    assert "pass `grayscale=True`" in str(info.value)
    assert isinstance(info.value, TypeError)


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
        UnknownParameterError, match="`use_key` was renamed to `embedding_key`"
    ) as info:
        fn(adata, *args, use_key="X_umap")
    assert "embedding_key='X_umap'" in str(info.value)
    assert isinstance(info.value, TypeError)


@pytest.mark.parametrize(("fn", "args"), _EMBEDDING_CALLS)
def test_basis_points_to_embedding_key(adata, fn, args):
    with pytest.raises(UnknownParameterError, match="to choose a stored embedding") as info:
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


def test_add_keys_accepts_frame_columns(adata):
    plot = cl.umap(adata, "leiden", add_keys=["X_UMAP1", "Barcode"])
    assert {"X_UMAP1", "Barcode"} <= set(cl.retrieve(plot).columns)


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


# ---- Removed `variable_keys`: points to `add_keys` instead of reaching the geom ----


@pytest.mark.parametrize(("fn", "args"), _ADD_KEYS_CALLS)
def test_variable_keys_points_to_add_keys(adata, fn, args):
    with pytest.raises(
        UnknownParameterError,
        match=rf"`variable_keys` was removed from `{fn.__name__}`; pass the same names as `add_keys=`",
    ) as info:
        fn(adata, *args, variable_keys=["CD14"])
    assert isinstance(info.value, TypeError)


@pytest.mark.parametrize("plural", [False, True])
def test_variable_keys_points_to_add_keys_spatial(plural):
    data = _spatial_anndata()
    name = "spatials" if plural else "spatial"
    with pytest.raises(UnknownParameterError, match=rf"removed from `{name}`.*`add_keys=`"):
        if plural:
            cl.spatials(data, ["score"], image=False, variable_keys=["GENE_A"])
        else:
            cl.spatial(data, "score", image=False, variable_keys=["GENE_A"])


def test_variable_keys_on_blend_names_blend(adata):
    with pytest.raises(UnknownParameterError, match=r"removed from `blend`"):
        cl.blend(adata, ["CD14", "MS4A1"], variable_keys=["CD3D"])


# ---- Former bare builtins: cellestial classes that still subclass the builtin ----


def _multipolygon_spatialdata():
    import geopandas as gpd
    import pandas as pd
    from anndata import AnnData
    from shapely.geometry import MultiPolygon, Point
    from spatialdata import SpatialData
    from spatialdata.models import ShapesModel, TableModel

    table = AnnData(
        X=np.ones((2, 1), dtype="float32"),
        obs=pd.DataFrame(
            {"instance_id": [0, 1], "region": ["mp", "mp"]}, index=["c0", "c1"]
        ),
        var=pd.DataFrame(index=["G1"]),
    )
    table = TableModel.parse(table, region="mp", region_key="region", instance_key="instance_id")
    geometries = [
        MultiPolygon([Point(i, i).buffer(1.0), Point(i + 5, i + 5).buffer(1.0)]) for i in range(2)
    ]
    shapes = ShapesModel.parse(gpd.GeoDataFrame(geometry=geometries, index=np.arange(2)))
    return SpatialData(shapes={"mp": shapes}, tables={"table": table})


FORMER_BUILTIN_CASES = [
    pytest.param(
        lambda data: cl.umaps(data, "leiden"),
        InvalidArgumentTypeError,
        TypeError,
        ["`keys` must be a sequence, got the string 'leiden'", "use `umap` for one key"],
        id="umaps",
    ),
    pytest.param(
        lambda data: cl.dimensionals(data, 3),
        InvalidArgumentTypeError,
        TypeError,
        ["got `int`", "use `dimensional` for one key"],
        id="dimensionals",
    ),
    pytest.param(
        lambda data: cl.violins(data, "CD14"),
        InvalidArgumentTypeError,
        TypeError,
        ["got the string 'CD14'", "use `violin` for one key"],
        id="violins",
    ),
    pytest.param(
        lambda data: cl.ridges(data, "CD14", GROUP_KEY),
        InvalidArgumentTypeError,
        TypeError,
        ["use `ridge` for one key"],
        id="ridges",
    ),
    pytest.param(
        lambda data: cl.spatials(data, "CD14"),
        InvalidArgumentTypeError,
        TypeError,
        ["use `spatial` for one key"],
        id="spatials",
    ),
    pytest.param(
        lambda data: cl.volcanos(data, "0", ranking_key="rank_leiden"),
        InvalidArgumentTypeError,
        TypeError,
        ["`groups` must be a sequence, got the string '0'", "use `volcano` for one group"],
        id="volcanos-groups",
    ),
    pytest.param(
        lambda data: cl.markers(data, groups="0", ranking_key="rank_leiden"),
        InvalidArgumentTypeError,
        TypeError,
        ["`groups`", "got the string '0'", "None for all groups"],
        id="markers-groups",
    ),
    pytest.param(
        lambda data: cl.heatmap(data, keys={"monocyte": "CD14"}, group_by=GROUP_KEY),
        InvalidArgumentTypeError,
        TypeError,
        ["group 'monocyte'", "Wrap it in a list"],
        id="heatmap-group-string",
    ),
    pytest.param(
        lambda data: cl.heatmap(data),
        InvalidArgumentError,
        ValueError,
        ["missing ['keys', 'group_by']", "markers=True"],
        id="heatmap-no-keys",
    ),
    pytest.param(
        lambda data: cl.dotplot(data, keys=["CD14"]),
        InvalidArgumentError,
        ValueError,
        ["missing ['group_by']"],
        id="dotplot-no-group-by",
    ),
    pytest.param(
        lambda data: cl.stacked_violin(data, group_by=GROUP_KEY),
        InvalidArgumentError,
        ValueError,
        ["missing ['keys']"],
        id="stacked-violin-no-keys",
    ),
    pytest.param(
        lambda data: cl.annotated_heatmap(data, ["CD14"], dendrogram=True),
        InvalidArgumentError,
        ValueError,
        ["`group_by` is required when `dendrogram=True`", "dendrogram=False"],
        id="annotated-heatmap-dendrogram",
    ),
    pytest.param(
        lambda data: cl.dotplot(data, markers=True, ranking_key="rank_leiden", n_genes=0),
        InvalidArgumentError,
        ValueError,
        ["`n_genes` must be >= 1, got 0"],
        id="dotplot-n-genes-zero",
    ),
    pytest.param(
        lambda data: cl.dotplot(data, markers=True, ranking_key="rank_leiden", n_genes=10**6),
        InvalidArgumentError,
        ValueError,
        ["exceeds", "ranking `rank_leiden`", "Pass `n_genes` <="],
        id="dotplot-n-genes-too-many",
    ),
    pytest.param(
        lambda data: cl.markers(data, ranking_key="rank_leiden", n_genes=10**6),
        InvalidArgumentError,
        ValueError,
        ["exceeds", "ranking `rank_leiden`", "Pass `n_genes` <="],
        id="markers-n-genes-too-many",
    ),
    pytest.param(
        lambda data: cl.marker_genes(data, ranking_key="rank_leiden", n_genes=10**6),
        InvalidArgumentError,
        ValueError,
        ["exceeds", "ranking `rank_leiden`"],
        id="marker-genes-n-genes-too-many",
    ),
    pytest.param(
        lambda data: cl.heatmap(data, markers=True, ranking_key="rank_leiden", groups=["nope"]),
        KeyNotFoundError,
        KeyError,
        ["Groups ['nope'] not found in the ranking `rank_leiden`", "Available:"],
        id="heatmap-unknown-group",
    ),
    pytest.param(
        lambda data: cl.markers(data, ranking_key="rank_leiden", groups=["nope"]),
        KeyNotFoundError,
        KeyError,
        ["Groups ['nope'] not found in the ranking `rank_leiden`"],
        id="markers-unknown-group",
    ),
    pytest.param(
        lambda data: cl.dotplot(data, keys=["CD14"], markers=True, ranking_key="rank_leiden"),
        InvalidArgumentError,
        ValueError,
        ["`keys` must be omitted when `markers=True`", "Remove `keys`"],
        id="dotplot-keys-with-markers",
    ),
    pytest.param(
        lambda data: cl.dotplot(
            data, markers=True, ranking_key="rank_leiden", group_by="cell_type_lvl1"
        ),
        InvalidArgumentError,
        ValueError,
        ["does not match", "Omit `group_by`"],
        id="dotplot-group-by-mismatch",
    ),
    pytest.param(
        lambda data: cl.volcano(data, "0", ranking_key="rank_leiden", tooltips=["nope"]),
        InvalidArgumentError,
        ValueError,
        ["`tooltips` names columns that are not in the volcano data: ['nope']", "Available:"],
        id="volcano-tooltips",
    ),
    pytest.param(
        lambda data: cl.volcano(data, "0", ranking_key="rank_leidn"),
        KeyNotFoundError,
        KeyError,
        ["No stored ranking named 'rank_leidn'; did you mean 'rank_leiden'?", "`ranking_key`"],
        id="volcano-missing-ranking",
    ),
]


@pytest.mark.parametrize(("call", "error", "builtin", "parts"), FORMER_BUILTIN_CASES)
def test_former_builtin_raises_use_cellestial_classes(ranked_pbmc, call, error, builtin, parts):
    """Each raise is the cellestial class, still the old builtin, and says how to fix it."""
    with pytest.raises(error) as info:
        call(ranked_pbmc)
    assert isinstance(info.value, builtin)
    message = str(info.value).strip("'\"")
    for part in parts:
        assert part in message


@pytest.mark.parametrize(("call", "error", "builtin", "parts"), FORMER_BUILTIN_CASES)
def test_former_builtin_messages_avoid_backend_internals(ranked_pbmc, call, error, builtin, parts):
    """No message names storage slots or scanpy functions."""
    with pytest.raises(error) as info:
        call(ranked_pbmc)
    message = str(info.value)
    for internal in ("adata.", ".uns[", "data.uns", "scanpy.tl", "key_added"):
        assert internal not in message


def test_unsupported_geometry_is_not_implemented_error():
    """A shapes element of MultiPolygons raises `UnsupportedGeometryError`."""
    from cellestial.errors import CellestialError, UnsupportedGeometryError

    with pytest.raises(UnsupportedGeometryError) as info:
        cl.spatial(_multipolygon_spatialdata(), polygon=True)
    assert isinstance(info.value, NotImplementedError)
    assert isinstance(info.value, CellestialError)
    assert "MultiPolygon" in str(info.value)
    assert "`shapes_name`" in str(info.value)
