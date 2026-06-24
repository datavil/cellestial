import numpy as np
import pytest
from anndata import AnnData
from lets_plot import aes, geom_point, layer_tooltips
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.single.differential.utilities import _build_markers_frame, _build_volcano_frame
from cellestial.single.heatmap.utilities import (
    _extract_rank_genes_groups,
    _resolve_rank_genes_groups_args,
    _resolve_rank_genes_groups_key,
)


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
            [[2.0, -2.0, 0.2, np.inf], [-2.5, 1.5, 0.0, 0.5]],
            names=["A", "B"],
        ),
        "pvals": np.rec.fromarrays(
            [[0.001, 0.002, 0.5, 0.001], [0.001, 0.003, 0.6, 0.7]],
            names=["A", "B"],
        ),
        "pvals_adj": np.rec.fromarrays(
            [[0.01, 0.02, 0.5, 0.0], [0.01, 0.02, 0.6, 0.7]],
            names=["A", "B"],
        ),
        "params": {"groupby": "cluster"},
    }
    return data


def test_resolve_rank_genes_groups_key_default_and_custom():
    assert _resolve_rank_genes_groups_key(True) == "rank_genes_groups"
    assert _resolve_rank_genes_groups_key("custom") == "custom"


def test_resolve_rank_genes_groups_key_invalid():
    with pytest.raises(TypeError, match="must be True or a string"):
        _resolve_rank_genes_groups_key(False)


def test_extract_rank_genes_groups_deduplicates_with_warning(ranked_adata):
    with pytest.warns(cl.util.errors.CellestialWarning, match="Some genes"):
        keys, group_by = _extract_rank_genes_groups(
            ranked_adata,
            rank_genes_groups=True,
            n_genes=2,
        )

    assert keys == {"A": ["gene_a", "shared"], "B": ["gene_b"]}
    assert group_by == "cluster"


def test_extract_rank_genes_groups_subset(ranked_adata):
    keys, group_by = _extract_rank_genes_groups(
        ranked_adata,
        rank_genes_groups=True,
        n_genes=1,
        groups=["B"],
    )

    assert keys == {"B": ["gene_b"]}
    assert group_by == "cluster"


@pytest.mark.parametrize("n_genes", [0, 10])
def test_extract_rank_genes_groups_invalid_n_genes(ranked_adata, n_genes):
    with pytest.raises(ValueError):
        _extract_rank_genes_groups(
            ranked_adata,
            rank_genes_groups=True,
            n_genes=n_genes,
        )


def test_extract_rank_genes_groups_missing_key(ranked_adata):
    with pytest.raises(Exception, match="not found"):
        _extract_rank_genes_groups(
            ranked_adata,
            rank_genes_groups="missing",
            n_genes=1,
        )


def test_extract_rank_genes_groups_missing_group(ranked_adata):
    with pytest.raises(Exception, match="Groups"):
        _extract_rank_genes_groups(
            ranked_adata,
            rank_genes_groups=True,
            n_genes=1,
            groups=["C"],
        )


def test_extract_rank_genes_groups_missing_params_groupby(ranked_adata):
    del ranked_adata.uns["rank_genes_groups"]["params"]["groupby"]

    with pytest.raises(Exception, match="missing 'groupby'"):
        _extract_rank_genes_groups(
            ranked_adata,
            rank_genes_groups=True,
            n_genes=1,
        )


def test_extract_rank_genes_groups_rejects_wrong_data_type():
    with pytest.raises(Exception, match="Expected"):
        _extract_rank_genes_groups(
            "not adata",
            rank_genes_groups=True,
            n_genes=1,
        )


def test_resolve_rank_genes_groups_args_rejects_keys(ranked_adata):
    with pytest.raises(ValueError, match="keys"):
        _resolve_rank_genes_groups_args(
            ranked_adata,
            rank_genes_groups=True,
            n_genes=1,
            groups=None,
            keys=["gene_a"],
            group_by=None,
        )


def test_resolve_rank_genes_groups_args_rejects_group_by_mismatch(ranked_adata):
    with pytest.raises(ValueError, match="does not match"):
        _resolve_rank_genes_groups_args(
            ranked_adata,
            rank_genes_groups=True,
            n_genes=1,
            groups=None,
            keys=None,
            group_by="other",
        )


def test_build_markers_frame(ranked_adata):
    frame, groups, group_by = _build_markers_frame(
        ranked_adata,
        key=True,
        n_genes=2,
        groups=["A"],
        variable_column="gene",
        score_column="score",
        rank_column="rank",
        group_column="group",
    )

    assert frame.shape == (2, 4)
    assert groups == ["A"]
    assert group_by == "cluster"
    assert frame["gene"].to_list() == ["gene_a", "shared"]


def test_build_markers_frame_missing_scores(ranked_adata):
    del ranked_adata.uns["rank_genes_groups"]["scores"]

    with pytest.raises(Exception, match="missing"):
        _build_markers_frame(
            ranked_adata,
            key=True,
            n_genes=1,
            groups=None,
            variable_column="gene",
            score_column="score",
            rank_column="rank",
            group_column="group",
        )


def test_markers_plot_options(ranked_adata):
    plot = cl.markers(
        ranked_adata,
        groups=["A", "B"],
        n_genes=2,
        mapping=aes(color="rank"),
        rank_color=True,
        line=True,
        line_kwargs={"alpha": 0.5},
        share_labels=True,
        share_axis=True,
        layers=geom_point(aes(x="rank", y="score"), size=0.1),
        interactive=True,
        ncol=1,
    )

    assert isinstance(plot, SupPlotsSpec)


def test_markers_rejects_invalid_groups(ranked_adata):
    with pytest.raises(TypeError, match="groups"):
        cl.markers(ranked_adata, groups="A")


def test_markers_rejects_wrong_data_type():
    with pytest.raises(Exception, match="Expected"):
        cl.markers("not adata")


def test_build_volcano_frame_uses_raw_pvalues(ranked_adata):
    frame = _build_volcano_frame(
        ranked_adata,
        "A",
        use_adjusted_pvalue=False,
        logfoldchange_threshold=1,
        pvalue_threshold=0.05,
    )

    assert frame["significance"].cast(str).to_list()[:3] == ["up", "down", "ns"]
    assert frame["neg_log_pvalue"].to_list()[0] == 3.0


def test_build_volcano_frame_errors(ranked_adata):
    with pytest.raises(KeyError, match="Group"):
        _build_volcano_frame(ranked_adata, "C")

    with pytest.raises(KeyError, match="not found"):
        _build_volcano_frame(ranked_adata, "A", key="missing")

    with pytest.raises(Exception, match="Expected"):
        _build_volcano_frame("not adata", "A")


def test_volcano_plot_options(ranked_adata):
    plot = cl.volcano(
        ranked_adata,
        "A",
        use_adjusted_pvalue=True,
        mapping=aes(shape="significance"),
        show_threshold_lines=True,
        threshold_kwargs={"alpha": 0.4},
        top_n=1,
        label_kwargs={"color": "blue"},
        nonsignificant_subsample=1,
        tooltips=layer_tooltips(["variable"]),
        interactive=True,
        size=1,
    )

    assert isinstance(plot, PlotSpec)


def test_volcano_tooltip_validation(ranked_adata):
    with pytest.raises(ValueError, match="tooltip columns"):
        cl.volcano(ranked_adata, "A", tooltips=["missing"])


def test_volcano_without_thresholds_or_labels(ranked_adata):
    plot = cl.volcano(
        ranked_adata,
        "A",
        show_threshold_lines=False,
        top_n=0,
        tooltips="none",
        nonsignificant_subsample=None,
    )

    assert isinstance(plot, PlotSpec)


def test_volcanos_plot_options(ranked_adata):
    plot = cl.volcanos(
        ranked_adata,
        ["A", "B"],
        show_threshold_lines=False,
        top_n=None,
        share_labels=True,
        share_axis=True,
        layers=geom_point(size=0.1),
        interactive=True,
        ncol=1,
    )

    assert isinstance(plot, SupPlotsSpec)


def test_volcanos_rejects_invalid_groups(ranked_adata):
    with pytest.raises(TypeError, match="groups"):
        cl.volcanos(ranked_adata, "A")
