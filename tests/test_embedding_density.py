"""Embedding density plots with `cl.embedding_density` (issue #74)."""

import re

import numpy as np
import polars as pl
import pytest
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.util.errors import (
    KeyNotFoundError,
    NonCategoricalColumnError,
    UnsupportedDataTypeError,
)


def _layer_frame(plot: PlotSpec, index: int) -> pl.DataFrame:
    return pl.DataFrame(plot.layer_data(index))


def _strip_labels(plot: PlotSpec, labels: list[str]) -> list[str]:
    """Facet labels in the order they are drawn."""
    texts = re.findall(r"<text[^>]*>([^<]*)</text>", plot.to_svg())
    return [text.strip() for text in texts if text.strip() in labels]


def test_unfaceted_density_covers_every_cell(adata):
    """Without `group_by`, one layer holds every cell with densities scaled to 1."""
    plot = cl.embedding_density(adata)
    assert isinstance(plot, PlotSpec)
    assert len(plot.layers) == 1
    density = _layer_frame(plot, 0)
    assert density.height == adata.n_obs
    assert density["ndensity"].max() == pytest.approx(1.0)
    assert density["ndensity"].min() > 0
    assert "UMAP1" in plot.to_svg()


def test_dense_cells_score_higher_than_sparse_cells(adata):
    """Cells near the embedding's densest point score higher than isolated ones."""
    density = _layer_frame(cl.embedding_density(adata, method="neighbors"), 0)
    densest = density.sort("density", descending=True).row(0, named=True)
    distance = ((pl.col("x") - densest["x"]) ** 2 + (pl.col("y") - densest["y"]) ** 2).sqrt()
    near = density.filter(distance < 0.3)["ndensity"].mean()
    far = density.filter(distance > 5)["ndensity"].mean()
    assert near > far


def test_group_density_uses_only_that_groups_cells(adata, group_key):
    """Each panel's density matches the density of its group plotted alone."""
    plot = cl.embedding_density(adata, group_by=group_key)
    faceted = _layer_frame(plot, 1)
    sizes = adata.obs[group_key].value_counts()
    categories = list(adata.obs[group_key].cat.categories)
    # Cells without a group are left out.
    assert faceted.height == int(adata.obs[group_key].notna().sum()) < adata.n_obs
    counts = faceted.group_by("panel").len().sort("panel")["len"].to_list()
    assert counts == [int(sizes[category]) for category in categories]
    # Every panel's own densest cell reaches 1.
    assert faceted.group_by("panel").agg(pl.col("ndensity").max())["ndensity"].min() == (
        pytest.approx(1.0)
    )

    group = categories[0]
    alone = _layer_frame(cl.embedding_density(adata, group_by=group_key, groups=group), 1)
    in_panel = faceted.filter(pl.col("panel") == 0)
    np.testing.assert_allclose(
        np.sort(in_panel["density"].to_numpy()), np.sort(alone["density"].to_numpy())
    )
    pooled = _layer_frame(cl.embedding_density(adata), 0)
    assert pooled["density"].max() > in_panel["density"].max()


def test_background_is_drawn_in_every_panel(adata, group_key):
    """Every cell is drawn in grey behind each group's panel."""
    n_groups = adata.obs[group_key].nunique()
    background = _layer_frame(cl.embedding_density(adata, group_by=group_key), 0)
    assert background.height == adata.n_obs * n_groups
    assert background.group_by("panel").len()["len"].to_list() == [adata.n_obs] * n_groups

    without = cl.embedding_density(adata, group_by=group_key, background_color=None)
    assert len(without.layers) == 1


def test_panels_follow_category_order(adata, cluster_key):
    """Panels are ordered like the categories (2 before 10), not alphabetically."""
    groups = ["10", "2", "0"]
    plot = cl.embedding_density(adata, group_by=cluster_key, groups=groups, method="kde2d")
    assert _strip_labels(plot, groups) == ["0", "2", "10"]


def test_drop_removes_groups(adata, group_key):
    """`drop` removes groups from the panels."""
    dropped = adata.obs[group_key].cat.categories[0]
    plot = cl.embedding_density(adata, group_by=group_key, drop=dropped, background_color=None)
    density = _layer_frame(plot, 0)
    labeled = adata.obs[group_key].notna()
    assert density.height == int((labeled & (adata.obs[group_key] != dropped)).sum())


def test_tsne_labels_and_arrow_axis(adata):
    """t-SNE axes are labeled `tSNE` and arrow axes are supported."""
    svg = cl.embedding_density(adata, dimensions="tsne", axis_type="arrow").to_svg()
    assert "tSNE1" in svg
    assert "tSNE2" in svg


def test_mudata_container(mudata):
    """MuData input is plotted through the same path."""
    plot = cl.embedding_density(mudata, dimensions="wnn_umap")
    assert _layer_frame(plot, 0).height == mudata.n_obs


def test_missing_group_by_suggests_close_match(adata):
    """A misspelled `group_by` names the close match."""
    with pytest.raises(KeyNotFoundError, match=r"`group_by` key `leidn` not found.*'leiden'"):
        cl.embedding_density(adata, group_by="leidn")


def test_numeric_group_by_is_rejected(adata):
    """A numeric `group_by` cannot be split into panels."""
    with pytest.raises(NonCategoricalColumnError, match=r"`group_by`.*`total_counts`.*numeric"):
        cl.embedding_density(adata, group_by="total_counts")


def test_unknown_group_is_rejected(adata, group_key):
    """An unknown group in `groups` is named."""
    with pytest.raises(KeyNotFoundError, match=r"`groups` \['T Cells'\] not found"):
        cl.embedding_density(adata, group_by=group_key, groups="T Cells")


def test_wrong_data_type_is_rejected():
    """Only single-cell data objects are accepted."""
    with pytest.raises(UnsupportedDataTypeError):
        cl.embedding_density(np.zeros((3, 2)))


def test_xy_must_hold_two_dimensions(adata):
    """`xy` needs exactly two dimensions."""
    with pytest.raises(KeyError, match="exactly two dimensions"):
        cl.embedding_density(adata, xy=(1, 2, 3))
