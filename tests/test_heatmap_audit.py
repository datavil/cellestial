"""Regression and edge-case tests from the 2026-11-18 audit of the heatmap family."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scanpy as sc

import cellestial as cl
from cellestial.util.errors import DendrogramMismatchError, DuplicateKeysError

KEYS = ["CD3D", "MS4A1", "LYZ", "NKG7"]
GROUP_PLOTS = ["heatmap", "matrixplot", "dotplot", "stacked_violin"]
ALL_PLOTS = [*GROUP_PLOTS, "annotated_heatmap"]


def _expression(adata, keys: list[str]) -> np.ndarray:
    matrix = adata[:, keys].X
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


@pytest.fixture(scope="module")
def integer_groups(adata):
    """Data grouped by a categorical of integers (not strings)."""
    local = adata[:2000].copy()
    local.obs["integer_cluster"] = pd.Categorical(local.obs["leiden"].astype(int))
    return local


@pytest.mark.parametrize("function", ALL_PLOTS)
def test_dendrogram_with_integer_categories_renders(integer_groups, function):
    plot = getattr(cl, function)(integer_groups, KEYS, "integer_cluster", dendrogram=True)
    assert plot.to_svg() is not None


@pytest.mark.parametrize("function", ALL_PLOTS)
def test_duplicate_keys_raise_duplicate_keys_error(adata, function):
    with pytest.raises(DuplicateKeysError, match=r"duplicates: \['CD3D'\]"):
        getattr(cl, function)(adata, ["CD3D", "LYZ", "CD3D"], "cell_type_lvl1")


def test_duplicate_key_within_one_key_group_names_the_group(adata):
    with pytest.raises(DuplicateKeysError, match="more than once in group 'T'"):
        cl.dotplot(adata, {"T": ["CD3D", "CD3D"]}, "cell_type_lvl1")


@pytest.fixture(scope="module")
def stale_dendrogram(adata):
    """A dendrogram computed before one group was removed from the data."""
    local = adata[:3000].copy()
    group_key = "cell_type_lvl1"
    sc.tl.dendrogram(local, groupby=group_key)
    removed = local.obs[group_key].value_counts().index[-1]
    subset = local[local.obs[group_key] != removed].copy()
    return subset, removed


@pytest.mark.parametrize(
    ("function", "options"),
    [("heatmap", {"aggregate": False}), ("annotated_heatmap", {})],
)
def test_dendrogram_with_groups_absent_from_data_raises(stale_dendrogram, function, options):
    subset, removed = stale_dendrogram
    with pytest.raises(DendrogramMismatchError, match=rf"no cells in the data: \['{removed}'\]"):
        getattr(cl, function)(subset, KEYS, "cell_type_lvl1", dendrogram=True, **options)


@pytest.fixture(scope="module")
def incomplete_dendrogram(adata):
    """A dendrogram that lacks one of the groups present in the data."""
    local = adata[:3000].copy()
    group_key = "cell_type_lvl1"
    missing = local.obs[group_key].value_counts().index[-1]
    partial = local[local.obs[group_key] != missing].copy()
    sc.tl.dendrogram(partial, groupby=group_key)
    local.uns[f"dendrogram_{group_key}"] = partial.uns[f"dendrogram_{group_key}"]
    return local, missing


@pytest.mark.parametrize("function", ALL_PLOTS)
def test_dendrogram_missing_data_groups_raises(incomplete_dendrogram, function):
    local, missing = incomplete_dendrogram
    with pytest.raises(
        DendrogramMismatchError, match=rf"lacks groups in the data: \['{missing}'\]"
    ):
        getattr(cl, function)(local, KEYS, "cell_type_lvl1", dendrogram=True)


@pytest.mark.parametrize("function", ALL_PLOTS)
def test_dendrogram_with_unused_category_raises_clear_error(adata, function):
    local = adata[:2000].copy()
    local.obs["with_unused"] = local.obs["cell_type_lvl1"].cat.add_categories(["Unused"])
    with pytest.raises(DendrogramMismatchError, match=r"no cells: \['Unused'\]"):
        getattr(cl, function)(local, KEYS, "with_unused", dendrogram=True)


def test_annotated_heatmap_drops_null_group_rows_with_and_without_binning(adata):
    local = adata[:600].copy()
    local.obs["partial"] = local.obs["cell_type_lvl1"].astype(str).astype("category")
    local.obs.loc[local.obs.index[:40], "partial"] = np.nan
    n_kept = int(local.obs["partial"].notna().sum())
    unbinned = cl.retrieve(cl.annotated_heatmap(local, KEYS, "partial", max_rows=None))
    assert unbinned["position_y"].n_unique() == n_kept
    binned = cl.retrieve(cl.annotated_heatmap(local, KEYS, "partial", max_rows=100))
    # Every binned row comes from a non-null group: the bins cover all 100 rows.
    assert binned["position_y"].n_unique() == 100


def test_matrixplot_values_match_numpy_group_means(adata):
    group_key = "cell_type_lvl1"
    frame = cl.retrieve(cl.matrixplot(adata, KEYS, group_key))
    expression = _expression(adata, KEYS)
    labels = adata.obs[group_key].astype(str).to_numpy()
    for row in frame.iter_rows(named=True):
        mask = labels == str(row[group_key])
        expected = expression[mask, KEYS.index(row["variable"])].mean()
        assert row["value"] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_dotplot_percentages_match_numpy(adata):
    group_key = "leiden"
    threshold = 0.5
    frame = cl.retrieve(cl.dotplot(adata, KEYS, group_key, expression_threshold=threshold))
    expression = _expression(adata, KEYS)
    labels = adata.obs[group_key].astype(str).to_numpy()
    for row in frame.iter_rows(named=True):
        values = expression[labels == str(row[group_key]), KEYS.index(row["variable"])]
        assert row["pct_exp"] == pytest.approx((values > threshold).mean() * 100)
        assert row["avg_exp"] == pytest.approx(values.mean(), rel=1e-5, abs=1e-6)


def test_heatmap_bins_average_their_cells(adata):
    local = adata[:1500].copy()
    group_key = "cell_type_lvl1"
    local = local[local.obs[group_key].notna()].copy()
    max_rows = 50
    frame = cl.retrieve(cl.heatmap(local, KEYS, group_key, max_rows=max_rows))
    assert frame["position_y"].n_unique() <= max_rows + local.obs[group_key].nunique()
    expression = _expression(local, KEYS)
    labels = local.obs[group_key].astype(str).to_numpy()
    barcodes = local.obs_names.to_numpy()
    total = len(barcodes)
    expected_rows = []
    for group in np.unique(labels):
        indices = np.flatnonzero(labels == group)
        indices = indices[np.argsort(barcodes[indices])]
        n_bins = max(1, round(len(indices) * max_rows / total))
        bins = np.arange(len(indices)) * n_bins // len(indices)
        expected_rows.extend(expression[indices[bins == b]].mean(axis=0) for b in range(n_bins))
    wide = frame.pivot(on="variable", index="position_y", values="value").select(KEYS)
    expected = np.sort(np.round(np.array(expected_rows), 5), axis=0)
    observed = np.sort(np.round(wide.to_numpy(), 5), axis=0)
    np.testing.assert_allclose(observed, expected, atol=1e-5)


@pytest.mark.parametrize("function", ALL_PLOTS)
def test_single_gene_and_single_group_render(adata, function):
    local = adata[:500].copy()
    local.obs["single"] = pd.Categorical(["only"] * local.n_obs)
    assert getattr(cl, function)(local, ["LYZ"], "cell_type_lvl1").to_svg() is not None
    assert getattr(cl, function)(local, KEYS, "single").to_svg() is not None


def test_group_order_is_stable_across_calls(adata):
    first = cl.retrieve(cl.dotplot(adata, KEYS, "cell_type_lvl1"))
    second = cl.retrieve(cl.dotplot(adata, KEYS, "cell_type_lvl1"))
    assert first.select("cell_type_lvl1", "position_y").equals(
        second.select("cell_type_lvl1", "position_y")
    )
    assert pl.Series(first["position_y"]).is_not_null().all()


def test_histogram_names_the_plotted_value_on_the_x_axis(adata):
    svg = cl.histogram(adata, "total_counts", bins=20).to_svg()
    assert ">total_counts<" in svg
    grid = cl.histograms(adata, ["total_counts", "pct_counts_mt"], bins=20).to_svg()
    assert ">total_counts<" in grid
    assert ">pct_counts_mt<" in grid


@pytest.mark.parametrize(
    ("plots", "widths", "error", "message"),
    [
        ([["a", "a"], ["a"]], [1, 2], ValueError, "cannot broadcast to rows \\[1\\]"),
        ([["a", "a"], ["a"]], [[1, 2]], ValueError, "has 1 rows but `plots` has 2 rows"),
        ([["a", "a"], ["a"]], [[1, 2], [1, 2]], ValueError, "`widths\\[1\\]` has 2 values"),
        ([["a", "a"], ["a"]], [[1, 2], 1], ValueError, "mixes numbers and sequences"),
        ([["a"]], "1", TypeError, "flat or nested sequence"),
    ],
)
def test_normalize_widths_rejects_mismatched_widths(plots, widths, error, message):
    from cellestial.util.operations import _normalize_widths

    with pytest.raises(error, match=message):
        _normalize_widths(plots, widths)


def test_normalize_widths_broadcasts_flat_widths():
    from cellestial.util.operations import _normalize_widths

    assert _normalize_widths([["a", "a"], ["a", "a"]], (1, 2)) == [[1, 2], [1, 2]]
    assert _normalize_widths([["a"], ["a"]], None) == [None, None]


def test_layout_rejects_empty_rows_and_mismatched_heights():
    from ggrust import ggplot

    plot = ggplot()
    with pytest.raises(ValueError, match="at least one row"):
        cl.layout([])
    with pytest.raises(ValueError, match="row 1 is empty"):
        cl.layout([[plot], []])
    with pytest.raises(ValueError, match="`heights` has 1 values"):
        cl.layout([[plot], [plot]], heights=[1])
    with pytest.raises(TypeError, match="nested sequence of rows"):
        cl.layout(plot)


def test_matrixplot_passes_modality_for_multimodal_dendrogram():
    mudata = cl.datasets.pbmc_cite()
    gene = str(mudata["rna"].var_names[0])
    plot = cl.matrixplot(mudata, [gene], "rna:leiden", dendrogram=True, modality="rna")
    assert plot.to_svg() is not None


@pytest.fixture(scope="module")
def shared_barcodes(adata):
    """Two groups' cells reuse the same barcodes, as after concatenating samples."""
    local = adata[adata.obs["cell_type_lvl1"].isin(["B Cells", "Monocytes"])][:400].copy()
    local.obs["batch"] = pd.Categorical(["first"] * 200 + ["second"] * 200)
    names = local.obs_names.to_numpy().copy()
    names[200:] = names[:200]
    local.obs_names = names
    return local


def test_heatmap_keeps_cells_with_shared_barcodes_apart(shared_barcodes):
    keys = ["CD3D", "LYZ"]
    frame = cl.retrieve(cl.heatmap(shared_barcodes, keys, "batch", max_rows=None))
    assert frame.height == shared_barcodes.n_obs * len(keys)
    assert frame["position_y"].n_unique() <= shared_barcodes.n_obs
    assert frame.group_by("position_y").agg(pl.col("batch").n_unique())["batch"].max() == 1


def test_heatmap_bins_never_mix_groups_with_shared_barcodes(shared_barcodes):
    keys = ["CD3D", "LYZ"]
    frame = cl.retrieve(cl.heatmap(shared_barcodes, keys, "batch", max_rows=20))
    assert frame["position_y"].n_unique() == 20
    assert frame.group_by("position_y").agg(pl.col("batch").n_unique())["batch"].max() == 1
    expression = _expression(shared_barcodes, keys)
    batch = shared_barcodes.obs["batch"].astype(str).to_numpy()
    for name in ("first", "second"):
        bins = frame.filter(pl.col("batch").cast(pl.String) == name)
        # Equal-sized bins, so the bin means average to the group mean.
        observed = bins.group_by("variable").agg(pl.col("value").mean()).sort("variable")
        expected = expression[batch == name].mean(axis=0)
        for row in observed.iter_rows(named=True):
            assert row["value"] == pytest.approx(expected[keys.index(row["variable"])], rel=1e-5)
