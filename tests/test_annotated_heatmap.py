import polars as pl
import pytest
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.complex.annotated_heatmap import _bin_observations
from cellestial.util.errors import UnsupportedDataTypeError


def test_annotated_heatmap_default(adata, markers, group_key):
    plot = cl.annotated_heatmap(adata, keys=markers, group_by=group_key)
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_both_axes(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations=["highly_variable"],
        row_annotations=[cluster_key, "pct_counts_mt"],
        scale_axis=0,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_numeric_column_annotation(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, column_annotations=["means"]
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_string_annotations(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations="highly_variable",
        row_annotations=cluster_key,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_no_legend(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, row_annotations=[cluster_key], legend=False
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_no_group_by(adata, markers):
    plot = cl.annotated_heatmap(adata, keys=markers, row_annotations="pct_counts_mt")
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_max_rows(adata, markers, group_key, cluster_key):
    plot = cl.annotated_heatmap(
        adata, keys=markers, group_by=group_key, row_annotations=[cluster_key], max_rows=200
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_annotation_colors(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        column_annotations="highly_variable",
        row_annotations=["pct_counts_mt"],
        annotation_colors={
            "highly_variable": {"true": "#000000", "false": "#dddddd"},
            "pct_counts_mt": ["#ffffcc", "#800026"],
        },
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_dendrogram(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        row_annotations=[group_key],
        dendrogram=True,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_dendrogram_requires_group_by(adata, markers):
    with pytest.raises(ValueError, match="group_by"):
        cl.annotated_heatmap(adata, keys=markers, dendrogram=True)


def test_annotated_heatmap_passthrough_params(adata, markers, group_key):
    plot = cl.annotated_heatmap(
        adata,
        keys=markers,
        group_by=group_key,
        group_lines_kwargs={"linetype": "dashed"},
        interactive=True,
        alpha=0.9,
    )
    assert isinstance(plot, SupPlotsSpec)


def test_annotated_heatmap_unsupported_data_type(markers, group_key):
    with pytest.raises(UnsupportedDataTypeError):
        cl.annotated_heatmap("not anndata", keys=markers, group_by=group_key)


def test_bin_observations_caps_rows():
    frame = pl.DataFrame(
        {
            "group": ["a"] * 50 + ["b"] * 50,
            "value": list(range(100)),
            "label": ["x"] * 50 + ["y"] * 50,
        }
    )
    binned = _bin_observations(frame, group_by="group", max_rows=10)
    # proportional allocation: 5 bins per equal-sized group, total == max_rows
    assert binned.height == 10
    # group column and a representative categorical label are preserved
    assert binned["group"].to_list() == ["a"] * 5 + ["b"] * 5
    assert set(binned["label"].to_list()) <= {"x", "y"}


def test_bin_observations_noop_when_small():
    frame = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
    assert _bin_observations(frame, group_by=None, max_rows=1000) is frame
