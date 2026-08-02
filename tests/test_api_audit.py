"""
Tests for the v1.0 API-audit changes.

Each section maps to an item in `plans/api_audit_v1.md`. The signature-level
checks guard the renames / default / annotation decisions against regressions;
the build checks confirm the plots still assemble under the new API.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from anndata import AnnData
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl
import cellestial.frames as frames
import cellestial.layers as layers
import cellestial.themes as themes
import cellestial.util as util
from cellestial.frames.build import anndata_observations_frame, anndata_variables_frame


def _param(fn, name):
    return inspect.signature(fn).parameters[name]


@pytest.fixture
def pca_adata():
    data = AnnData(X=np.ones((3, 3)))
    data.uns["pca"] = {"variance_ratio": np.array([0.5, 0.3, 0.2])}
    return data


@pytest.fixture
def ranked_adata():
    data = AnnData(X=np.ones((4, 4)))
    data.obs["cluster"] = ["A", "A", "B", "B"]
    data.uns["rank_genes_groups"] = {
        "names": np.rec.fromarrays(
            [["gene_a", "shared", "gene_c", "gene_d"], ["gene_b", "shared", "gene_e", "gene_f"]],
            names=["A", "B"],
        ),
        "scores": np.rec.fromarrays(
            [[4.0, 3.0, 2.0, 1.0], [5.0, 2.5, 1.5, 0.5]],
            names=["A", "B"],
        ),
        "logfoldchanges": np.rec.fromarrays(
            [[2.0, -2.0, 0.2, 1.5], [-2.5, 1.5, 0.0, 0.5]],
            names=["A", "B"],
        ),
        "pvals": np.rec.fromarrays(
            [[0.001, 0.002, 0.5, 0.001], [0.001, 0.003, 0.6, 0.7]],
            names=["A", "B"],
        ),
        "pvals_adj": np.rec.fromarrays(
            [[0.01, 0.02, 0.5, 0.04], [0.01, 0.02, 0.6, 0.7]],
            names=["A", "B"],
        ),
    }
    return data


# ---- #6: elbow `line_type` renamed to `linetype` ----


def test_elbow_linetype_default():
    assert _param(cl.elbow, "linetype").default == "dashed"


def test_elbow_dropped_old_line_type():
    assert "line_type" not in inspect.signature(cl.elbow).parameters


def test_elbow_accepts_linetype(pca_adata):
    plot = cl.elbow(pca_adata, n_pcs=3, linetype="dotted")
    assert plot is not None


# ---- #8: expression `axis_type` default is now None (was "axis") ----


def test_expression_axis_type_default_is_none():
    assert _param(cl.expression, "axis_type").default is None


def test_expression_axis_type_matches_plural_and_dimensional():
    default = _param(cl.expression, "axis_type").default
    assert default == _param(cl.expressions, "axis_type").default
    assert default == _param(cl.umap, "axis_type").default
    assert default == _param(cl.dimensional, "axis_type").default


@pytest.mark.parametrize("axis_type", [None, "axis", "arrow"])
def test_expression_builds_with_each_axis_type(adata, axis_type):
    plot = cl.expression(adata, "CD14", axis_type=axis_type)
    assert isinstance(plot, PlotSpec)


# ---- #9: distribution `share_ticks` defaults to False ----


@pytest.mark.parametrize("fn", [cl.histograms, cl.violins, cl.boxplots])
def test_share_ticks_defaults_false(fn):
    assert _param(fn, "share_ticks").default is False


# ---- #13: grid `widths`/`heights` annotated `list[float] | None` ----


@pytest.mark.parametrize("fn", [cl.histograms, cl.violins, cl.boxplots, cl.ridges])
@pytest.mark.parametrize("param", ["widths", "heights"])
def test_grid_size_params_annotated_list_float(fn, param):
    assert _param(fn, param).annotation == "list[float] | None"


# ---- #14: frame builders default to capitalized Barcode/Variable ----


def test_frame_builder_default_param_values():
    assert _param(cl.build_frame, "observations_name").default == "Barcode"
    assert _param(cl.build_frame, "variables_name").default == "Variable"
    assert _param(anndata_observations_frame, "observations_name").default == "Barcode"
    assert _param(anndata_variables_frame, "variables_name").default == "Variable"


def test_build_frame_obs_axis_column_is_capitalized(adata):
    frame = cl.build_frame(adata, axis=0)
    assert "Barcode" in frame.columns
    assert "barcode" not in frame.columns


def test_build_frame_var_axis_column_is_capitalized(adata):
    frame = cl.build_frame(adata, axis=1)
    assert "Variable" in frame.columns


def test_build_frame_output_feeds_plot_without_renaming(adata):
    # The footgun #14 fixes: a frame built with defaults now carries "Barcode",
    # which is exactly what the plot's observations_name default expects, so it
    # drops straight into `frame=` with no renaming.
    frame = cl.build_frame(adata, axis=0, variable_keys="CD14", include_dimensions=2)
    assert _param(cl.umap, "observations_name").default in frame.columns
    plot = cl.umap(adata, "CD14", frame=frame)
    assert isinstance(plot, PlotSpec)


# ---- batch-1 annotation fixes (#2-#4) committed in 54dab4b ----


def test_stream_cutoff_percentile_annotation():
    assert _param(cl.stream, "cutoff_percentile").annotation == "float | None"


def test_highest_expressed_genes_size_annotation():
    assert _param(cl.highest_expressed_genes, "size").annotation == "float"


def test_volcanos_return_annotation():
    assert inspect.signature(cl.volcanos).return_annotation == "SupPlotsSpec"


def test_volcanos_returns_supplots(ranked_adata):
    plot = cl.volcanos(ranked_adata, ["A", "B"])
    assert isinstance(plot, SupPlotsSpec)


# ---- final v1 API fixes: `midpoint` spelling and public subpackage exports ----


@pytest.mark.parametrize(
    "fn",
    [
        cl.annotated_heatmap,
        cl.dimensional,
        cl.dimensionals,
        cl.dotplot,
        cl.expression,
        cl.expressions,
        cl.heatmap,
        cl.matrixplot,
        cl.pca,
        cl.pcas,
        cl.spatial,
        cl.spatials,
        cl.stacked_violin,
        cl.tsne,
        cl.tsnes,
        cl.umap,
        cl.umaps,
    ],
)
def test_gradient_parameter_is_midpoint(fn):
    parameters = inspect.signature(fn).parameters
    assert "midpoint" in parameters
    assert "mid_point" not in parameters


@pytest.mark.parametrize("module", [frames, layers, themes, util])
def test_subpackage_all_excludes_private_names(module):
    assert all(not name.startswith("_") for name in module.__all__)
