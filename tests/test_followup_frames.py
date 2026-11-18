"""Follow-up fixes to the 2026-11-18 audit of `cellestial/frames`, `cellestial/util` and datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.frames._container import _container
from cellestial.frames.operations import _pca_variance_frame
from cellestial.util.dendrogram import _dendrogram_info
from cellestial.util.errors import (
    AmbiguousColumnError,
    ConflictingLengthError,
    EmptySelectionError,
    GridIndexError,
    InvalidArgumentError,
    InvalidArgumentTypeError,
    InvalidPlotSpecError,
    KeyNotFoundError,
    UnsupportedDataTypeError,
    VariableNotFoundError,
    _did_you_mean,
)
from cellestial.util.utilities import (
    _determine_axis,
    _require_feature_key,
    _resolve_embedding_key,
    _resolve_tooltips,
    _validate_tooltips,
)

# --- errors ------------------------------------------------------------------


@pytest.mark.parametrize("error_class", [KeyNotFoundError, VariableNotFoundError])
def test_key_errors_subclass_key_error_and_print_unquoted(error_class):
    """Key errors are `KeyError`s whose message prints without quotes."""
    error = error_class("key 'x' not found")
    assert isinstance(error, KeyError)
    assert str(error) == "key 'x' not found"


def test_did_you_mean_names_close_matches():
    """The hint names a close match and is empty when nothing is close."""
    assert _did_you_mean("leidn", ["leiden", "louvain"]) == "Did you mean 'leiden'?"
    assert _did_you_mean("zzz", ["leiden"]) == ""


def test_require_feature_key_suggests_close_match(adata):
    """An unknown color key names the closest metadata column."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        _require_feature_key(adata, "leidn")


def test_unknown_variable_suggests_close_match(adata):
    """An unknown variable names the closest variable."""
    with pytest.raises(VariableNotFoundError, match="CD14"):
        cl.build_frame(adata, axis=0, variable_keys=["CD1A4"])


def test_unknown_metadata_column_is_key_not_found_with_hint(adata):
    """An unknown metadata column raises `KeyNotFoundError` with a hint."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        cl.build_frame(adata, axis=0, metadata_columns=["leidn"])


def test_unknown_modality_suggests_close_match(mudata):
    """An unknown modality names the closest modality."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'rna'"):
        _container(mudata).select_modality("rnaa")


# --- bare builtin raises become errors.py classes ------------------------------


def _small_plot():
    from ggrust import aes, geom_point, ggplot

    return ggplot({"x": [1], "y": [1]}, aes("x", "y")) + geom_point()


def test_layout_accepts_numpy_widths():
    """A flat or nested numpy `widths` works like the matching list."""
    plot = _small_plot()
    flat = cl.layout([[plot, plot]], widths=np.array([1, 2]))
    nested = cl.layout([[plot, plot], [plot]], widths=[np.array([1, 2]), np.array([1])])
    matrix = cl.layout([[plot, plot], [plot, plot]], widths=np.array([[1, 2], [2, 1]]))
    assert flat.as_dict() == cl.layout([[plot, plot]], widths=[1, 2]).as_dict()
    assert nested.as_dict() == cl.layout([[plot, plot], [plot]], widths=[[1, 2], [1]]).as_dict()
    assert matrix.as_dict() == cl.layout([[plot, plot]] * 2, widths=[[1, 2], [2, 1]]).as_dict()


@pytest.mark.parametrize(
    ("call", "error_class", "message"),
    [
        (lambda plot: cl.layout(plot), InvalidArgumentTypeError, "plots"),
        (lambda plot: cl.layout([]), InvalidArgumentError, "at least one row"),
        (lambda plot: cl.layout([[]]), InvalidArgumentError, "row 0 is empty"),
        (lambda plot: cl.layout([[plot]], heights=[1, 2]), ConflictingLengthError, "heights"),
        (lambda plot: cl.layout([[plot]], widths="1"), InvalidArgumentTypeError, "widths"),
        (lambda plot: cl.layout([[plot]], widths=[1, [1]]), InvalidArgumentError, "mixes"),
        (lambda plot: cl.layout([[plot]], widths=[1, 2]), ConflictingLengthError, "broadcast"),
        (lambda plot: cl.layout([[plot]], widths=[[1], [1]]), ConflictingLengthError, "rows"),
        (lambda plot: cl.layout([[plot]], widths=[[1, 2]]), ConflictingLengthError, "widths"),
    ],
)
def test_layout_raises_specific_errors(call, error_class, message):
    """Invalid `layout` arguments raise the matching `errors.py` class."""
    with pytest.raises(error_class, match=message):
        call(_small_plot())


def test_grid_lookups_raise_specific_errors():
    """`get_figure` and `get_figures` raise specific classes for bad input."""
    grid = cl.layout([[_small_plot(), _small_plot()]])
    with pytest.raises(InvalidPlotSpecError):
        cl.get_figure(_small_plot(), 0)
    with pytest.raises(InvalidArgumentTypeError, match="index"):
        cl.get_figure(grid, "0")
    with pytest.raises(GridIndexError, match="out of range"):
        cl.get_figure(grid, 5)
    with pytest.raises(GridIndexError, match="out of range"):
        cl.get_figures(grid, [0, 5])
    with pytest.raises(InvalidPlotSpecError):
        cl.retrieve("not a plot")


@pytest.mark.parametrize(
    ("kwargs", "error_class"),
    [
        ({"axis": 0, "include_dimensions": -1}, InvalidArgumentError),
        ({"axis": 0, "include_dimensions": 1.5}, InvalidArgumentTypeError),
        ({}, InvalidArgumentError),
    ],
)
def test_build_frame_invalid_arguments(adata, kwargs, error_class):
    """Invalid `build_frame` arguments raise the matching `errors.py` class."""
    with pytest.raises(error_class):
        cl.build_frame(adata, **kwargs)


def test_determine_axis_unknown_key_suggests_close_match(adata):
    """A key on neither axis raises `KeyNotFoundError` with a hint."""
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        _determine_axis(adata, ["leidn"])


def test_unknown_tooltip_field_raises_key_not_found(adata):
    """A tooltip naming no column raises `KeyNotFoundError`."""
    frame = pl.DataFrame({"leiden": ["0"]})
    with pytest.raises(KeyNotFoundError, match="Did you mean 'leiden'"):
        _validate_tooltips(["leidn"], frame)


def test_missing_pca_variance_raises_key_not_found(adata):
    """Plotting variance without a stored PCA names the missing key."""
    with pytest.raises(KeyNotFoundError, match="PCA variance ratio"):
        _pca_variance_frame(adata, use_key="missing_pca")


def test_too_many_marker_genes_is_invalid_argument(mudata):
    """Asking for more marker genes than ranked raises `InvalidArgumentError`."""
    with pytest.raises(InvalidArgumentError, match="n_genes"):
        cl.marker_genes(mudata, modality="rna", n_genes=10**6)


def test_dendrogram_rejects_unsupported_data():
    """The dendrogram helper rejects a non-AnnData object with the shared class."""
    with pytest.raises(UnsupportedDataTypeError):
        _dendrogram_info("not data", "leiden")


def test_from_cellxgene_rejects_source_without_uuid():
    """A source holding no dataset UUID raises `InvalidArgumentError`."""
    with pytest.raises(InvalidArgumentError, match="UUID"):
        cl.datasets.from_cellxgene("not-a-uuid")


# --- category order ----------------------------------------------------------


def test_build_frame_keeps_source_category_order(adata):
    """Categorical metadata keeps its source category order, not first appearance."""
    source = adata.obs["leiden"].cat.categories.astype(str).to_list()
    frame = cl.build_frame(adata, axis=0, metadata_columns=["leiden"])
    assert frame["leiden"].dtype == pl.Enum(source)
    assert frame["leiden"].to_list() == adata.obs["leiden"].astype(str).to_list()


def test_build_frame_keeps_numeric_category_order():
    """Numeric categories keep their order and missing values stay null."""
    data = AnnData(X=np.zeros((4, 1), dtype=np.float32))
    data.obs["cluster"] = pd.Categorical([10, 2, None, 1], categories=[1, 2, 10])
    frame = cl.build_frame(data, axis=0, metadata_columns=["cluster"])
    assert frame["cluster"].dtype == pl.Enum(["1", "2", "10"])
    assert frame["cluster"].to_list() == ["10", "2", None, "1"]


def test_umap_legend_follows_category_order(adata):
    """The plotted factor levels follow the source categories."""
    spec = cl.umap(adata, key="leiden").as_dict()
    annotations = spec["data_meta"]["series_annotations"]
    levels = next(item["factor_levels"] for item in annotations if item["column"] == "leiden")
    assert levels == adata.obs["leiden"].cat.categories.astype(str).to_list()


# --- frames report O1 to O5 ----------------------------------------------------


def test_id_column_colliding_with_metadata_names_the_parameter(adata):
    """O1: an id column named like a metadata column raises, naming the parameter."""
    data = adata.copy()
    data.obs["Barcode"] = data.obs_names.to_numpy()
    data.var["Variable"] = data.var_names.to_numpy()
    with pytest.raises(InvalidArgumentError, match="observation_id_column"):
        cl.build_frame(data, axis=0)
    with pytest.raises(InvalidArgumentError, match="variable_id_column"):
        cl.build_frame(data, axis=1)
    frame = cl.build_frame(data, axis=0, observation_id_column="cell")
    assert frame["cell"].to_list() == frame["Barcode"].to_list()


def test_single_string_metadata_and_dimension_keys_are_one_key(adata):
    """O2: a string `metadata_columns` / `dimension_keys` is one key, like `variable_keys`."""
    frame = cl.build_frame(
        adata,
        axis=0,
        metadata_columns="leiden",
        include_dimensions=2,
        dimension_keys="X_umap",
        observation_id_column=None,
    )
    assert frame.columns == ["leiden", "X_UMAP1", "X_UMAP2"]


def test_variables_without_matrix_raise_clear_error():
    """O3: reading variables from an object without a main matrix raises a clear error."""
    data = AnnData(
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1", "g2"]),
        layers={"counts": np.ones((2, 2), dtype=np.float32)},
    )
    with pytest.raises(KeyNotFoundError, match="no main expression matrix"):
        cl.build_frame(data, variable_keys=["g1"])


@pytest.mark.parametrize("key", [{"CD14"}, np.array(["CD14", "LYZ"]), pd.Index(["CD14"])])
def test_singular_rejects_sets_and_arrays(adata, key):
    """O4: sets and arrays where one key is expected point to the plural."""
    with pytest.raises(InvalidArgumentTypeError, match="umaps"):
        cl.umap(adata, key=key)


def test_resolve_tooltips_rejects_strings_other_than_none(adata):
    """O5: a string other than 'none' for `tooltips` raises with a fix."""
    with pytest.raises(InvalidArgumentError, match=r"tooltips=\['CD14'\]"):
        _resolve_tooltips("CD14", data=adata, variable_keys=[], defaults=[])
    assert _resolve_tooltips("none", data=adata, variable_keys=[], defaults=[]) == "none"


# --- MuData: plotted and dendrogram group columns ------------------------------


@pytest.mark.parametrize(
    "plotter",
    [cl.dotplot, cl.matrixplot, cl.stacked_violin, cl.heatmap, cl.tracksplot],
)
def test_bare_group_on_container_and_modality_is_ambiguous(mudata, plotter):
    """A bare `group_by` on both the container and the modality must be qualified."""
    gene = str(mudata["rna"].var_names[0])
    with pytest.raises(AmbiguousColumnError, match="rna:leiden"):
        plotter(mudata, [gene], group_by="leiden", modality="rna", dendrogram=True)


@pytest.mark.parametrize("plotter", [cl.dotplot, cl.matrixplot, cl.stacked_violin])
def test_qualified_group_plots_the_dendrogram_groups(mudata, plotter):
    """With `rna:leiden`, the plotted groups are the groups the dendrogram clusters."""
    gene = str(mudata["rna"].var_names[0])
    plot = plotter(mudata, [gene], group_by="rna:leiden", modality="rna", dendrogram=True)
    frame = cl.retrieve(plot) if isinstance(plot, PlotSpec) else cl.retrieve(plot, 0)
    plotted = set(pl.Series(frame["rna:leiden"]).cast(pl.String).drop_nulls().to_list())
    clustered = set(mudata["rna"].obs["leiden"].astype(str))
    container_groups = set(mudata.obs["leiden"].astype(str))
    assert plotted <= clustered
    assert not plotted <= container_groups  # the modality clustering has more groups


# --- low-priority robustness ---------------------------------------------------


def test_embedding_fallback_warning_lists_rest_sorted(adata):
    """The fallback warning lists the other embeddings in sorted order."""
    data = adata.copy()
    embedding = data.obsm["X_umap"]
    del data.obsm["X_umap"]
    data.obsm["X_umap_harmony"] = embedding
    data.obsm["X_zeta"] = embedding
    data.obsm["X_alpha"] = embedding
    with pytest.warns(UserWarning, match=r"Rest: \['X_alpha', .*'X_zeta'\]"):
        _resolve_embedding_key(data, "umap", None, (1, 2))


def test_determine_axis_rejects_empty_keys(adata):
    """An empty key list has no axis to determine."""
    with pytest.raises(EmptySelectionError, match="keys"):
        _determine_axis(adata, [])


@pytest.mark.parametrize("axis", [True, False])
def test_build_frame_rejects_boolean_axis(adata, axis):
    """`axis=True` is not accepted as axis 1."""
    with pytest.raises(InvalidArgumentError, match="axis"):
        cl.build_frame(adata, axis=axis)


def test_container_column_without_modality_names_the_problem(mudata):
    """A missing modality on a multimodal object is reported, not printed as `None`."""
    with pytest.raises(KeyNotFoundError, match="`modality` is required") as error:
        _container(mudata).container_column(None, "missing")
    assert "None" not in str(error.value)
