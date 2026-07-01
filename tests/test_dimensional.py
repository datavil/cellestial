import inspect

import pytest
from lets_plot import aes
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.util import retrieve

# ---- singular: dimensional / umap / pca / tsne ----


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_singular_default_no_key(adata, fn):
    plot = fn(adata)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_singular_with_obs_key(adata, fn, group_key):
    plot = fn(adata, group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_singular_with_gene_key(adata, fn):
    plot = fn(adata, "CD14")
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize(
    "dims",
    ["umap", "pca", "tsne"],
)
def test_dimensional_dimensions_param(adata, dims):
    plot = cl.dimensional(adata, "leiden", dimensions=dims)
    assert isinstance(plot, PlotSpec)


def test_dimensional_xy_swap(adata):
    # Non-default embedding components
    plot = cl.pca(adata, "CD14", xy=(2, 3))
    assert isinstance(plot, PlotSpec)


def test_dimensional_mapping_extra_aesthetic(adata, group_key):
    # Providing a key + an extra aesthetic via mapping (shape)
    plot = cl.umap(adata, group_key, mapping=aes(shape=group_key))
    assert isinstance(plot, PlotSpec)


def test_dimensional_with_variable_keys(adata):
    plot = cl.umap(adata, "CD14", variable_keys=["MS4A1", "NKG7"])
    assert isinstance(plot, PlotSpec)


def test_dimensional_tooltips_none(adata, group_key):
    plot = cl.umap(adata, group_key, tooltips="none")
    assert isinstance(plot, PlotSpec)


def test_dimensional_tooltips_sequence(adata, group_key):
    plot = cl.umap(adata, group_key, tooltips=[group_key, "n_genes_by_counts"])
    assert isinstance(plot, PlotSpec)


def test_dimensional_invalid_key_raises(adata):
    from cellestial.util.errors import KeyNotFoundError

    with pytest.raises(KeyNotFoundError, match="not found"):
        cl.umap(adata, "NOT_A_REAL_KEY_xyz")


# ---- plural: dimensionals / umaps / pcas / tsnes ----


@pytest.mark.parametrize("fn", [cl.umaps, cl.pcas, cl.tsnes])
def test_plural_returns_subplots(adata, fn, group_key):
    plot = fn(adata, [group_key, "leiden"])
    assert isinstance(plot, SupPlotsSpec)


def test_plural_mixed_gene_and_obs(adata, group_key):
    plot = cl.umaps(adata, ["CD14", group_key, "MS4A1"])
    assert isinstance(plot, SupPlotsSpec)


def test_plural_single_entry_list(adata):
    plot = cl.umaps(adata, ["CD14"])
    assert isinstance(plot, SupPlotsSpec)


def test_dimensionals_dimensions(adata, group_key):
    plot = cl.dimensionals(adata, [group_key, "CD14"], dimensions="pca")
    assert isinstance(plot, SupPlotsSpec)


def test_dimensionals_ncol(adata, group_key):
    plot = cl.umaps(adata, [group_key, "CD14", "MS4A1"], ncol=2)
    assert isinstance(plot, SupPlotsSpec)


def test_plural_invalid_key_raises(adata):
    with pytest.raises(Exception):
        cl.umaps(adata, ["NOT_A_KEY_xyz"])


# ---- drop ----


def test_dimensional_drop_removes_group(adata, group_key):
    full = set(cl.umap(adata, group_key).as_dict()["data"][group_key].to_list())
    dropped = sorted(value for value in full if value is not None)[0]
    remaining = set(cl.umap(adata, group_key, drop=dropped).as_dict()["data"][group_key].to_list())
    assert remaining == full - {dropped}


def test_dimensional_drop_non_categorical_warns(adata):
    with pytest.warns(cl.util.errors.CellestialWarning, match="`drop` filter ignored"):
        cl.umap(adata, "CD14", drop="x")


def test_dimensional_groups_keeps_only_selected(adata, group_key):
    full = sorted(
        value
        for value in set(cl.umap(adata, group_key).as_dict()["data"][group_key].to_list())
        if value is not None
    )
    kept = full[:2]
    remaining = set(cl.umap(adata, group_key, groups=kept).as_dict()["data"][group_key].to_list())
    assert remaining == set(kept)


def test_dimensional_groups_then_drop(adata, group_key):
    full = sorted(
        value
        for value in set(cl.umap(adata, group_key).as_dict()["data"][group_key].to_list())
        if value is not None
    )
    kept = full[:3]
    remaining = set(
        cl.umap(adata, group_key, groups=kept, drop=kept[-1])
        .as_dict()["data"][group_key]
        .to_list()
    )
    assert remaining == set(kept[:-1])


def test_dimensionals_drop_propagates(adata, group_key):
    full = set(cl.umap(adata, group_key).as_dict()["data"][group_key].to_list())
    dropped = sorted(value for value in full if value is not None)[0]
    plot = cl.umaps(adata, [group_key], drop=dropped)
    assert isinstance(plot, SupPlotsSpec)
    panel = plot.as_dict()["figures"][0]
    assert dropped not in set(panel["data"][group_key].to_list())


# ---- frame narrowing: custom metadata tooltips and add_keys ----


def test_dimensionals_custom_metadata_tooltip(adata, group_key):
    # Regression: the shared frame built for the grid must contain custom
    # tooltip columns, even though they are not the colour key or in `mapping`.
    obs_column = "n_genes_by_counts"
    plot = cl.dimensionals(adata, keys=[group_key, "CD3D"], tooltips=[obs_column])
    assert isinstance(plot, SupPlotsSpec)
    for panel in plot.as_dict()["figures"]:
        assert obs_column in panel["data"]


def test_umap_add_keys_materializes_extra_columns(adata, group_key):
    # `add_keys` forces obs metadata and gene columns into the frame so an
    # added layer can read columns the plot itself does not reference.
    frame = retrieve(cl.umap(adata, group_key, add_keys=["n_genes_by_counts", "MS4A1"]))
    assert "n_genes_by_counts" in frame.columns
    assert "MS4A1" in frame.columns


def test_umap_custom_metadata_tooltip_single(adata, group_key):
    # Single-plot path: a custom obs tooltip must survive into the frame.
    frame = retrieve(cl.umap(adata, group_key, tooltips=["n_genes_by_counts"]))
    assert "n_genes_by_counts" in frame.columns


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_dimensional_wrappers_add_keys_materialize_extra_columns(adata, fn, group_key):
    frame = retrieve(
        fn(
            adata,
            group_key,
            add_keys=["n_genes_by_counts", "MS4A1"],
            tooltips="none",
        )
    )
    assert "n_genes_by_counts" in frame.columns
    assert "MS4A1" in frame.columns


@pytest.mark.parametrize("fn", [cl.dimensionals, cl.umaps, cl.pcas, cl.tsnes])
def test_plural_dimensional_wrappers_add_keys_shared_frame(adata, fn, group_key):
    plot = fn(
        adata,
        [group_key, "CD14"],
        add_keys=["n_genes_by_counts", "MS4A1"],
        tooltips="none",
    )
    assert isinstance(plot, SupPlotsSpec)
    for panel in plot.as_dict()["figures"]:
        panel_columns = panel["data"].columns
        assert "n_genes_by_counts" in panel_columns
        assert "MS4A1" in panel_columns


def test_expression_add_keys_materializes_extra_columns(adata):
    frame = retrieve(
        cl.expression(
            adata,
            "CD14",
            add_keys=["n_genes_by_counts", "MS4A1"],
            tooltips="none",
        )
    )
    assert "n_genes_by_counts" in frame.columns
    assert "MS4A1" in frame.columns


def test_expressions_add_keys_materializes_shared_frame_columns(adata):
    plot = cl.expressions(
        adata,
        ["CD14", "MS4A1"],
        add_keys="n_genes_by_counts",
        tooltips="none",
    )
    assert isinstance(plot, SupPlotsSpec)
    for panel in plot.as_dict()["figures"]:
        assert "n_genes_by_counts" in panel["data"].columns


# ---- on-data legend halo forwarding ----


@pytest.mark.parametrize(
    "fn",
    [
        cl.dimensional,
        cl.umap,
        cl.pca,
        cl.tsne,
        cl.expression,
        cl.dimensionals,
        cl.umaps,
        cl.pcas,
        cl.tsnes,
        cl.expressions,
    ],
)
def test_wrappers_expose_halo_params(fn):
    params = inspect.signature(fn).parameters
    assert "halo_width" in params
    assert "halo_color" in params


@pytest.mark.parametrize("fn", [cl.umap, cl.pca, cl.tsne])
def test_singular_legend_ondata_halo_renders(adata, group_key, fn):
    plot = fn(adata, group_key, legend_ondata=True, halo_width=2.0, halo_color="black")
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.umaps, cl.pcas, cl.tsnes])
def test_plural_legend_ondata_halo_renders(adata, group_key, fn):
    plot = fn(adata, [group_key], legend_ondata=True, halo_width=2.0, halo_color="black")
    assert isinstance(plot, SupPlotsSpec)
