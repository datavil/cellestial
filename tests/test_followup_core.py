"""Regression tests for the follow-up fixes to the 2026-11-18 core audit open issues."""

import inspect

import pytest
from ggrust import aes

import cellestial as cl
from cellestial.util.errors import (
    EmptySelectionError,
    InvalidArgumentError,
    KeyNotFoundError,
    VariableNotFoundError,
)

# ---- histogram / histograms: one key, string keys ----


def test_histogram_rejects_a_list_of_keys(adata):
    """Histogram rejects a list of keys."""
    with pytest.raises(TypeError, match="histograms"):
        cl.histogram(adata, ["n_genes_by_counts", "total_counts"])


def test_histograms_reject_string_keys(adata):
    """Histograms reject string keys."""
    with pytest.raises(TypeError, match="Sequence"):
        cl.histograms(adata, "CD14")


# ---- ridge: no hidden fill ----


def test_ridge_does_not_map_fill(adata):
    """Ridge does not map fill."""
    plot = cl.ridge(adata, "CD14", "leiden")
    assert "fill" not in plot.as_dict()["layers"][0]["mapping"]


def test_ridge_user_fill_mapping(adata):
    """Ridge user fill mapping."""
    plot = cl.ridge(adata, "CD14", "leiden", mapping=aes(fill="leiden"))
    assert plot.as_dict()["layers"][0]["mapping"]["fill"] == "leiden"


def test_ridge_fill_shortcut_maps_the_column(adata):
    """Ridge fill shortcut maps the column."""
    plot = cl.ridge(adata, "CD14", "leiden", fill="leiden")
    assert plot.as_dict()["layers"][0]["mapping"]["fill"] == "leiden"


def test_ridge_geom_fill_sets_a_constant(adata):
    """Ridge geom_fill sets a constant fill, not a mapping."""
    layer = cl.ridge(adata, "CD14", "leiden", geom_fill="#8da0cb").as_dict()["layers"][0]
    assert layer["fill"] == "#8da0cb"
    assert "fill" not in layer["mapping"]


@pytest.mark.parametrize("function", ["ridge", "ridges"])
def test_ridge_fill_shortcut_and_mapping_fill_raise(adata, function):
    """Ridge fill shortcut and mapping fill raise."""
    keys = "CD14" if function == "ridge" else ["CD14"]
    with pytest.raises(InvalidArgumentError, match="`fill`"):
        getattr(cl, function)(adata, keys, "leiden", fill="leiden", mapping=aes(fill="leiden"))


def test_ridges_fill_shortcut_reaches_every_panel(adata):
    """Ridges fill shortcut reaches every panel."""
    figures = cl.ridges(adata, ["CD14", "B2M"], "leiden", fill="leiden").as_dict()["figures"]
    assert all(figure["layers"][0]["mapping"]["fill"] == "leiden" for figure in figures)


def test_ridge_literal_color_points_to_geom_color(adata):
    """Ridge literal color points to geom_color."""
    with pytest.raises(KeyNotFoundError, match="geom_color"):
        cl.ridge(adata, "CD14", "leiden", color="#1f1f1f")


def test_highest_expressed_genes_does_not_map_fill(adata):
    """Highest expressed genes does not map fill unless asked."""
    assert "fill" not in cl.highest_expressed_genes(adata, 5).as_dict()["layers"][0]["mapping"]
    plot = cl.highest_expressed_genes(adata, 5, mapping=aes(fill="variable"))
    assert plot.as_dict()["layers"][0]["mapping"]["fill"] == "variable"


@pytest.mark.parametrize("function", ["elbow", "highest_expressed_genes"])
def test_geom_color_sets_the_constant_color(adata, function):
    """Geom_color sets the constant color."""
    layers = getattr(cl, function)(adata, geom_color="#0f4f8f").as_dict()["layers"]
    assert layers[0]["color"] == "#0f4f8f"


@pytest.mark.parametrize("function", ["elbow", "highest_expressed_genes"])
def test_old_color_constant_points_to_geom_color(adata, function):
    """Old color constant points to geom_color."""
    with pytest.raises(InvalidArgumentError, match="geom_color"):
        getattr(cl, function)(adata, color="#0f4f8f")


# ---- mapping clash ----


@pytest.mark.parametrize(
    ("call", "aesthetic"),
    [
        (lambda data: cl.umap(data, "leiden", mapping=aes(color="cell_type_lvl1")), "color"),
        (lambda data: cl.dimensional(data, "leiden", mapping=aes(x="total_counts")), "x"),
        (lambda data: cl.pca(data, "leiden", mapping=aes(y="total_counts")), "y"),
        (lambda data: cl.expression(data, "CD14", mapping=aes(color="leiden")), "color"),
        (lambda data: cl.violin(data, "CD14", group_by="leiden", mapping=aes(x="leiden")), "x"),
        (lambda data: cl.boxplot(data, "CD14", group_by="leiden", mapping=aes(y="CD3D")), "y"),
        (lambda data: cl.histogram(data, "CD14", mapping=aes(x="CD3D")), "x"),
        (
            lambda data: cl.violin(data, "CD14", group_by="leiden", point_mapping=aes(x="leiden")),
            "x",
        ),
        (lambda data: cl.ridge(data, "CD14", "leiden", mapping=aes(y="leiden")), "y"),
        (lambda data: cl.umaps(data, ["leiden"], mapping=aes(color="cell_type_lvl1")), "color"),
        (lambda data: cl.violins(data, ["CD14"], group_by="leiden", mapping=aes(x="leiden")), "x"),
    ],
)
def test_mapping_clash_raises_invalid_argument(adata, call, aesthetic):
    """Mapping clash raises invalid argument."""
    with pytest.raises(InvalidArgumentError, match=f"`{aesthetic}`"):
        call(adata)


def test_dimensional_without_key_accepts_color_mapping(adata):
    """Dimensional without key accepts color mapping."""
    plot = cl.umap(adata, mapping=aes(color="leiden"))
    assert plot.as_dict()["layers"][0]["mapping"]["color"] == "leiden"


@pytest.mark.parametrize(
    ("layer", "aesthetic"),
    [
        (lambda: cl.cluster_outlines(groups="0", mapping=aes(x="X_UMAP2")), "x"),
        (lambda: cl.cluster_outlines(groups="0", mapping=aes(group="leiden")), "group"),
        (lambda: cl.stream(mapping=aes(y="X_UMAP1")), "y"),
    ],
)
def test_layer_mapping_clash_raises_invalid_argument(adata, layer, aesthetic):
    """Layer mapping clash raises invalid argument."""
    plot = cl.umap(adata, "leiden")
    with pytest.raises(InvalidArgumentError, match=f"`{aesthetic}`"):
        plot + layer()


# ---- violin: key equal to group_by, non-numeric key ----


def test_violin_key_equal_to_group_by_raises(adata):
    """Violin key equal to group by raises."""
    with pytest.raises(InvalidArgumentError, match="group_by"):
        cl.violin(adata, "leiden", group_by="leiden")


def test_violin_non_numeric_key_raises(adata):
    """Violin non numeric key raises."""
    with pytest.raises(InvalidArgumentError, match="numeric"):
        cl.violin(adata, "cell_type_lvl1", group_by="leiden")


def test_violin_boolean_key_still_plots(adata):
    """Violin boolean key still plots."""
    data = adata.copy()
    data.obs["flag"] = data.obs["leiden"].astype(str) == "0"
    cl.violin(data, "flag", group_by="leiden")


# ---- highest_expressed_genes: unused parameter removed ----


def test_highest_expressed_genes_has_no_observation_id_column():
    """Highest expressed genes has no observation id column."""
    assert "observation_id_column" not in inspect.signature(cl.highest_expressed_genes).parameters


# ---- pca labels ----


def test_pca_axis_labels_read_pc(adata):
    """Pca axis labels read pc."""
    guides = cl.pca(adata, "leiden", xy=(2, 3)).as_dict()["guides"]
    assert guides["x"]["title"] == "PC2"
    assert guides["y"]["title"] == "PC3"
    figure = cl.pcas(adata, ["leiden", "CD14"]).as_dict()["figures"][0]
    assert figure["guides"]["x"]["title"] == "PC1"


# ---- bare builtin raises ----


def test_point_geom_invalid_raises(adata):
    """Point geom invalid raises."""
    with pytest.raises(InvalidArgumentError, match="point_geom"):
        cl.violin(adata, "CD14", group_by="leiden", point_geom="dots")


def test_axis_type_invalid_raises(adata):
    """Axis type invalid raises."""
    with pytest.raises(InvalidArgumentError, match="axis_type"):
        cl.umap(adata, "leiden", axis_type="arrows")


def test_xy_length_raises_invalid_argument(adata):
    """Xy length raises invalid argument."""
    with pytest.raises(InvalidArgumentError, match="xy"):
        cl.umap(adata, "leiden", xy=(1, 2, 3))


def test_stream_missing_velocity_raises_key_not_found(adata):
    """Stream missing velocity raises key not found."""
    with pytest.raises(KeyNotFoundError, match="velocity"):
        cl.umap(adata, "leiden") + cl.stream()


def test_stream_unparsable_dimension_raises_invalid_argument(adata):
    """Stream unparsable dimension raises invalid argument."""
    plot = cl.xyplot(adata, "total_counts", "n_genes_by_counts")
    with pytest.raises(InvalidArgumentError, match="velocity_key"):
        plot + cl.stream(velocity_key="velocity_umap")


def test_cluster_outlines_unknown_group_raises_key_not_found(adata):
    """Cluster outlines unknown group raises key not found."""
    with pytest.raises(KeyNotFoundError, match=r"(?i)did you mean"):
        cl.umap(adata, "cell_type_lvl1") + cl.cluster_outlines(groups="B Cell")


def test_cluster_outlines_too_few_points_raises_empty_selection(adata):
    """Cluster outlines too few points raises empty selection."""
    subset = adata[:40].copy()
    group = str(subset.obs["leiden"].value_counts().index[-1])
    plot = cl.umap(subset[subset.obs["leiden"] == group].copy()[:4], "leiden")
    with pytest.raises(EmptySelectionError, match="fewer than 5"):
        plot + cl.cluster_outlines(groups=group)


def test_expression_unknown_gene_suggests_close_match(adata):
    """Expression unknown gene suggests close match."""
    with pytest.raises(VariableNotFoundError, match=r"(?i)did you mean"):
        cl.expression(adata, "CD1")


# ---- xyplot: backend helper keeps behavior ----


def test_xyplot_embedding_with_metadata_on_observations(adata):
    """Xyplot embedding with metadata on observations."""
    plot = cl.xyplot(adata, "X_UMAP1", "total_counts")
    assert "X_UMAP1" in cl.retrieve(plot).columns


# ---- dimensionals: groups across keys ----


def test_dimensionals_groups_missing_in_one_key_names_it(adata):
    """Dimensionals groups missing in one key names it."""
    with pytest.raises(KeyNotFoundError, match="cell_type_lvl1") as error:
        cl.dimensionals(adata, ["leiden", "cell_type_lvl1"], groups=["0"])
    assert "every categorical key" in str(error.value)


def test_dimensionals_groups_skip_numeric_keys(adata):
    """Dimensionals groups skip numeric keys."""
    plots = cl.dimensionals(adata, ["leiden", "CD14"], groups=["0"])
    assert plots is not None
