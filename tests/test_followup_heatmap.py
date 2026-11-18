"""Follow-up fixes for the improvement suggestions of the 2026-11-18 heatmap audit."""

import inspect

import numpy as np
import polars as pl
import pytest
import scanpy as sc

import cellestial as cl
from cellestial.complex.annotated_heatmap import _bin_observations
from cellestial.single.heatmap.utilities import _bin_within_groups, _compute_violin_polygons
from cellestial.util.errors import DendrogramMismatchError, InvalidArgumentError

KEYS = ["CD3D", "MS4A1", "LYZ", "NKG7"]


# ---- S1: bin names keep barcode order ----


def test_bin_names_sort_in_bin_order():
    """Bin names sort as text in the same order as the bins, so rows keep barcode order."""
    frame = pl.DataFrame(
        {
            "Barcode": [f"cell_{index:03d}" for index in range(300)],
            "group": ["A"] * 300,
            "CD3D": np.arange(300, dtype=np.float64),
        }
    )
    binned = _bin_within_groups(
        frame,
        observation_id_column="Barcode",
        group_by="group",
        keys=["CD3D"],
        y_order_groups=["A"],
        max_rows=30,
    ).sort("Barcode")

    assert binned.height == 30
    # values rise with the barcode, so the bin means must rise with the bin name
    assert binned["CD3D"].is_sorted()


# ---- S2: matrixplot names `modality` ----


def test_matrixplot_declares_modality():
    """`modality` is part of the signature, not a `**geom_kwargs` passthrough."""
    assert "modality" in inspect.signature(cl.matrixplot).parameters


# ---- S3: stacked_violin validation and flat violins ----


@pytest.mark.parametrize(("option", "value"), [("fill_statistic", "max"), ("scale", "height")])
def test_stacked_violin_rejects_unknown_options(adata, option, value):
    """Unknown `fill_statistic` or `scale` values raise before any work is done."""
    with pytest.raises(InvalidArgumentError, match=f"`{option}`"):
        cl.stacked_violin(adata, KEYS, "cell_type_lvl1", **{option: value})


def _violin_frame(values: dict[str, list[float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "variable": [key for key, group_values in values.items() for _ in group_values],
            "group": [group for key in values for group in ("A", "B") for _ in range(3)],
            "value": [value for group_values in values.values() for value in group_values],
        }
    )


def _polygons(frame: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return _compute_violin_polygons(
        frame,
        variable_column="variable",
        value_column="value",
        group_by="group",
        x_keys=keys,
        y_order_groups=["A", "B"],
        n_points=20,
        scale="width",
        width_scale=0.8,
        height_scale=0.8,
        aggregate="median",
        aggregate_column="expression",
    )


def test_constant_group_draws_a_flat_violin():
    """A group whose values are all equal gets a flat line, not an empty cell."""
    frame = _violin_frame({"gene": [0.0, 0.0, 0.0, 1.0, 2.0, 4.0]})
    polygons = _polygons(frame, ["gene"])
    flat = polygons.filter(pl.col("group") == "A")

    assert flat.height > 0
    assert flat["y"].n_unique() == 1
    assert flat["x"].min() < 0 < flat["x"].max()


def test_constant_variable_draws_flat_violins_in_every_group():
    """A variable equal everywhere gets a flat line in each group."""
    frame = _violin_frame({"gene": [0.0] * 6})
    polygons = _polygons(frame, ["gene"])

    assert sorted(polygons["group"].unique()) == ["A", "B"]


# ---- S4: deterministic mode ----


def test_binned_annotation_ties_pick_the_first_value_in_sort_order():
    """A tie for the most frequent label is settled the same way on every run."""
    frame = pl.DataFrame({"label": ["y", "x", "x", "y"] * 2, "value": [1.0] * 8})
    for _ in range(20):
        shuffled = frame.sample(fraction=1.0, shuffle=True)
        binned = _bin_observations(shuffled, group_by=None, max_rows=1)

        assert binned["label"].to_list() == ["x"]


# ---- S5: color mapping on a numeric track ----


def test_annotated_heatmap_rejects_a_color_mapping_for_a_numeric_track(adata):
    """A category -> color mapping cannot color a numeric track."""
    local = adata[:500].copy()
    with pytest.raises(InvalidArgumentError, match=r"'total_counts'.*numeric"):
        cl.annotated_heatmap(
            local,
            KEYS,
            "cell_type_lvl1",
            row_annotations="total_counts",
            annotation_colors={"total_counts": {"low": "#ffffff"}},
        )


# ---- S8: tracksplot checks the dendrogram like the other plots ----


def test_tracksplot_rejects_a_dendrogram_lacking_data_groups(adata):
    """A data group the dendrogram does not cover raises, instead of being dropped."""
    local = adata[:3000].copy()
    group_key = "cell_type_lvl1"
    missing = local.obs[group_key].value_counts().index[-1]
    partial = local[local.obs[group_key] != missing].copy()
    sc.tl.dendrogram(partial, groupby=group_key)
    local.uns[f"dendrogram_{group_key}"] = partial.uns[f"dendrogram_{group_key}"]

    with pytest.raises(
        DendrogramMismatchError, match=rf"lacks groups in the data: \['{missing}'\]"
    ):
        cl.tracksplot(local, KEYS, group_key, dendrogram=True)


# ---- DendrogramMismatchError hint ----


def test_dendrogram_mismatch_hint_names_the_column_mixup(adata):
    """The hint does not claim subsetting is the only cause."""
    local = adata[:3000].copy()
    group_key = "cell_type_lvl1"
    sc.tl.dendrogram(local, groupby=group_key)
    removed = local.obs[group_key].value_counts().index[-1]
    subset = local[local.obs[group_key] != removed].copy()

    with pytest.raises(DendrogramMismatchError, match="same-named column"):
        cl.heatmap(subset, KEYS, group_key, dendrogram=True, aggregate=False)
