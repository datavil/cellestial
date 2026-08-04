import pytest
from lets_plot import aes, geom_vline
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.util import retrieve
from cellestial.util.errors import KeyNotFoundError

# ---- violin / boxplot (singular) ----


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_single_key_gene(adata, fn, group_key):
    plot = fn(adata, "CD14", fill=group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_single_key_obs(adata, fn, group_key):
    plot = fn(adata, "n_genes_by_counts", fill=group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_sequence_of_keys_returns_single_plot(adata, fn, group_key):
    # Singular violin/boxplot accepts a list of keys and still returns a single PlotSpec
    # (keys become facets / panels within one plot).
    plot = fn(
        adata,
        ["pct_counts_in_top_200_genes", "n_genes_by_counts"],
        fill=group_key,
    )
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_threshold(adata, fn, group_key):
    plot = fn(adata, "CD14", fill=group_key, threshold=0.1)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_show_points_false(adata, fn, group_key):
    plot = fn(adata, "CD14", fill=group_key, show_points=False)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_group_by(adata, fn, group_key):
    plot = fn(adata, "CD14", group_by=group_key)
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_add_keys(adata, fn, group_key):
    plot = fn(adata, "CD14", group_by=group_key, add_keys=["MS4A1", "NKG7"])
    assert isinstance(plot, PlotSpec)


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_invalid_key(adata, fn):
    with pytest.raises(Exception):
        fn(adata, "NOT_A_REAL_KEY_xyz")


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot, cl.histogram])
@pytest.mark.parametrize("aesthetic", ["fill", "color"])
def test_literal_color_suggests_geom_variant(adata, fn, aesthetic):
    # `fill`/`color` map a column; a literal color is a common mix-up coming from
    # lets-plot, so the error points at `geom_fill`/`geom_color`.
    with pytest.raises(KeyNotFoundError, match=f"use `geom_{aesthetic}='red'`"):
        fn(adata, "CD14", **{aesthetic: "red"})


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots, cl.histograms])
def test_plural_literal_color_suggests_geom_variant(adata, fn):
    with pytest.raises(KeyNotFoundError, match="use `geom_fill='red'`"):
        fn(adata, ["CD14", "MS4A1"], fill="red")


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_invalid_point_geom_raises_value_error(adata, fn, group_key):
    with pytest.raises(ValueError, match="point_geom must be one of"):
        fn(adata, "CD14", fill=group_key, point_geom="bad")


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_point_kwargs_not_mutated(adata, fn, group_key):
    point_kwargs = {"stroke": 0, "color": "red"}
    expected = dict(point_kwargs)

    plot = fn(
        adata,
        "CD14",
        fill=group_key,
        point_mapping=aes(color=group_key),
        point_kwargs=point_kwargs,
    )

    assert isinstance(plot, PlotSpec)
    assert point_kwargs == expected


# ---- violins / boxplots (plural) ----


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_grid(adata, fn, group_key):
    plot = fn(
        adata,
        ["n_genes_by_counts", "HLA-DRA", "pct_counts_hb"],
        fill=group_key,
        ncol=2,
    )
    assert isinstance(plot, SupPlotsSpec)


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_single_key_list(adata, fn, group_key):
    plot = fn(adata, ["CD14"], fill=group_key)
    assert isinstance(plot, SupPlotsSpec)


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_group_by(adata, fn, group_key):
    plot = fn(adata, ["CD14", "MS4A1"], group_by=group_key)
    assert isinstance(plot, SupPlotsSpec)


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots, cl.histograms])
def test_plural_threshold_filters_each_panel(adata, fn):
    keys = ["CD14", "MS4A1"]
    plot = fn(adata, keys, threshold=0.1)

    for panel, key in zip(plot.as_dict()["figures"], keys, strict=True):
        assert panel["data"][key].min() >= 0.1


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_point_mapping_materializes_its_column(adata, fn):
    plot = fn(
        adata,
        ["CD14", "MS4A1"],
        point_mapping=aes(color="n_genes_by_counts"),
    )

    for panel in plot.as_dict()["figures"]:
        assert "n_genes_by_counts" in panel["data"].columns
        assert any(
            layer.get("mapping", {}).get("color") == "n_genes_by_counts"
            for layer in panel["layers"]
        )


# ---- ridge / ridges ----


def test_ridge_basic(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key)
    assert isinstance(plot, PlotSpec)


def test_ridge_threshold_and_scale(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key, threshold=0.1, scale=1.5)
    assert isinstance(plot, PlotSpec)


def test_ridge_add_keys(adata, group_key):
    plot = cl.ridge(adata, key="CD14", group_by=group_key, add_keys=["MS4A1"])
    assert isinstance(plot, PlotSpec)


def test_ridges_multiple(adata, group_key):
    plot = cl.ridges(adata, keys=["CD14", "MS4A1", "NKG7"], group_by=group_key)
    assert isinstance(plot, SupPlotsSpec)


def test_ridges_ncol(adata, group_key):
    plot = cl.ridges(adata, keys=["CD14", "MS4A1", "NKG7", "CST3"], group_by=group_key, ncol=2)
    assert isinstance(plot, SupPlotsSpec)


# ---- mapping override ----


def test_violin_mapping_extra_aes(adata, group_key):
    plot = cl.violin(adata, "CD14", fill=group_key, mapping=aes(color=group_key))
    assert isinstance(plot, PlotSpec)


# ---- drop ----


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_distribution_drop_removes_group(adata, fn, group_key):
    full = set(fn(adata, "CD14", group_by=group_key).as_dict()["data"][group_key].to_list())
    dropped = sorted(full)[0]
    remaining = set(
        fn(adata, "CD14", group_by=group_key, drop=dropped).as_dict()["data"][group_key].to_list()
    )
    assert remaining == full - {dropped}


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_distribution_groups_keeps_only_selected(adata, fn, group_key):
    full = sorted(
        set(fn(adata, "CD14", group_by=group_key).as_dict()["data"][group_key].to_list())
    )
    kept = full[:2]
    remaining = set(
        fn(adata, "CD14", group_by=group_key, groups=kept).as_dict()["data"][group_key].to_list()
    )
    assert remaining == set(kept)


def test_distribution_drop_non_categorical_warns(adata):
    with pytest.warns(cl.util.errors.CellestialWarning, match="`drop` filter ignored"):
        cl.violin(adata, "CD14", group_by="n_genes_by_counts", drop="x")


def test_ridge_drop_removes_group(adata, group_key):
    full = set(
        cl.ridge(adata, key="CD14", group_by=group_key).as_dict()["data"][group_key].to_list()
    )
    dropped = sorted(value for value in full if value is not None)[0]
    remaining = set(
        cl.ridge(adata, key="CD14", group_by=group_key, drop=dropped)
        .as_dict()["data"][group_key]
        .to_list()
    )
    assert remaining == full - {dropped}


def test_ridge_custom_metadata_tooltip(adata, group_key):
    # Regression: tooltips are resolved before the frame is built, so a custom
    # obs tooltip column is materialised rather than failing validation.
    plot = cl.ridge(adata, key="CD14", group_by=group_key, tooltips=["n_genes_by_counts"])
    assert "n_genes_by_counts" in retrieve(plot).columns


def test_ridges_custom_metadata_tooltip(adata, group_key):
    # Regression: the shared ridges frame must contain the custom tooltip column.
    plot = cl.ridges(
        adata, keys=["CD14", "MS4A1"], group_by=group_key, tooltips=["n_genes_by_counts"]
    )
    assert isinstance(plot, SupPlotsSpec)
    for panel in plot.as_dict()["figures"]:
        assert "n_genes_by_counts" in panel["data"]


@pytest.mark.parametrize("fn", [cl.violin, cl.boxplot])
def test_distribution_custom_metadata_tooltip(adata, fn, group_key):
    # Regression mirror of ridge: custom tooltip columns must be included before
    # the narrowed frame is validated.
    plot = fn(adata, "CD14", fill=group_key, tooltips=["n_genes_by_counts"])
    assert "n_genes_by_counts" in retrieve(plot).columns


@pytest.mark.parametrize("fn", [cl.violins, cl.boxplots])
def test_plural_distribution_custom_metadata_tooltip(adata, fn, group_key):
    # Regression mirror of ridges: shared frames must contain tooltip-only columns.
    plot = fn(
        adata,
        ["CD14", "MS4A1"],
        fill=group_key,
        tooltips=["n_genes_by_counts"],
    )
    assert isinstance(plot, SupPlotsSpec)
    for panel in plot.as_dict()["figures"]:
        assert "n_genes_by_counts" in panel["data"].columns


# ---- histogram / histograms ----


def test_histogram_builds_expected_geom_and_filters_threshold(adata, group_key):
    plot = cl.histogram(
        adata,
        "n_genes_by_counts",
        fill=group_key,
        bins=8,
        binwidth=5,
        threshold=100,
        tooltips=["n_genes_by_counts"],
    )
    spec = plot.as_dict()
    layer = spec["layers"][0]

    assert layer["geom"] == "histogram"
    assert layer["mapping"] == {"x": "n_genes_by_counts", "fill": group_key}
    assert layer["bins"] == 8
    assert layer["binwidth"] == 5
    assert spec["data"]["n_genes_by_counts"].min() >= 100


def test_histogram_group_filters_and_variable_axis(adata, group_key):
    groups = sorted(adata.obs[group_key].dropna().unique())
    kept = groups[:2]
    dropped = groups[0]

    grouped = cl.histogram(adata, "CD14", group_by=group_key, groups=kept)
    without_group = cl.histogram(adata, "CD14", group_by=group_key, drop=dropped)
    variable_axis = cl.histogram(adata, "means", axis=1)

    assert set(grouped.as_dict()["data"][group_key].to_list()) == set(kept)
    assert dropped not in without_group.as_dict()["data"][group_key].to_list()
    assert variable_axis.as_dict()["data"].height == adata.n_vars


def test_histograms_materialize_shared_columns_and_grid_options(adata, group_key):
    grid = cl.histograms(
        adata,
        ["CD14", "MS4A1"],
        fill=group_key,
        bins=12,
        add_keys="n_genes_by_counts",
        tooltips=["n_genes_by_counts"],
        layers=geom_vline(xintercept=0),
        share_ticks=True,
        share_axis=True,
        interactive=True,
        ncol=1,
    )
    spec = grid.as_dict()

    assert spec["layout"]["ncol"] == 1
    assert len(spec["figures"]) == 2
    for panel, key in zip(spec["figures"], ["CD14", "MS4A1"], strict=True):
        assert {group_key, "n_genes_by_counts", key} <= set(panel["data"].columns)
        assert panel["layers"][0]["geom"] == "histogram"
        assert panel["layers"][0]["bins"] == 12
        assert any(layer["geom"] == "vline" for layer in panel["layers"])


def test_histogram_validation_paths(adata):
    with pytest.raises(TypeError):
        cl.histogram("not anndata", "CD14")
    with pytest.raises(ValueError, match="axis"):
        cl.histogram(adata, "not_a_key")
    with pytest.raises(ValueError, match="empty"):
        cl.histograms(adata, [])
    with pytest.raises(ValueError):
        cl.histograms(adata, ["CD14", 1])
