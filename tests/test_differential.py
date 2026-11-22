import numpy as np
import pytest
from anndata import AnnData
from ggrust import aes, geom_point, layer_tooltips
from ggrust.plot.core import PlotSpec
from ggrust.plot.subplots import SupPlotsSpec

import cellestial as cl
from cellestial.single.differential.utilities import _build_markers_frame, _build_volcano_frame
from cellestial.single.heatmap.utilities import (
    _extract_rank_genes_groups,
    _check_markers_flag,
    _resolve_rank_genes_groups_args,
)
from cellestial.util.errors import KeyNotFoundError, UnsupportedDataTypeError


@pytest.fixture
def ranked_adata():
    """A two-group ranking with one shared gene and a non-finite fold change."""
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


def test_check_markers_flag_accepts_bools():
    """`True` and `False` pass the `markers` check."""
    _check_markers_flag(True)
    _check_markers_flag(False)
    _check_markers_flag(np.True_)


def test_volcano_frame_drops_nonfinite_values_but_keeps_zero_pvalue(ranked_adata):
    """Raw non-finite values should be dropped while zero p-values remain."""
    ranked_adata.uns["rank_genes_groups"]["pvals_adj"]["A"] = [
        0.0,
        np.nan,
        -0.1,
        0.01,
    ]

    frame = _build_volcano_frame(ranked_adata, "A")

    assert frame["variable"].to_list() == ["gene_a", "gene_c"]
    assert np.isinf(frame["negative_log_pvalue"][0])


def test_check_markers_flag_rejects_non_bool():
    """A non-bool `markers` value raises."""
    with pytest.raises(TypeError, match="must be True or False"):
        _check_markers_flag(1)


def test_extract_rank_genes_groups_deduplicates_with_warning(ranked_adata):
    """Genes shared across groups are kept once, with a warning."""
    with pytest.warns(cl.util.errors.CellestialWarning, match="Some genes"):
        keys, group_by = _extract_rank_genes_groups(
            ranked_adata,
            ranking_key="rank_genes_groups",
            n_genes=2,
        )

    assert keys == {"A": ["gene_a", "shared"], "B": ["gene_b"]}
    assert group_by == "cluster"


def test_extract_rank_genes_groups_subset(ranked_adata):
    """A subset of groups keeps only their genes, in order."""
    keys, group_by = _extract_rank_genes_groups(
        ranked_adata,
        ranking_key="rank_genes_groups",
        n_genes=1,
        groups=["B"],
    )

    assert keys == {"B": ["gene_b"]}
    assert group_by == "cluster"


@pytest.mark.parametrize("n_genes", [0, 10])
def test_extract_rank_genes_groups_invalid_n_genes(ranked_adata, n_genes):
    """`n_genes` out of range raises."""
    with pytest.raises(ValueError):
        _extract_rank_genes_groups(
            ranked_adata,
            ranking_key="rank_genes_groups",
            n_genes=n_genes,
        )


def test_extract_rank_genes_groups_missing_key(ranked_adata):
    """A missing ranking key raises."""
    with pytest.raises(KeyNotFoundError, match="No precomputed ranking found"):
        _extract_rank_genes_groups(
            ranked_adata,
            ranking_key="missing",
            n_genes=1,
        )


def test_extract_rank_genes_groups_missing_group(ranked_adata):
    """A missing group raises."""
    with pytest.raises(Exception, match="Groups"):
        _extract_rank_genes_groups(
            ranked_adata,
            ranking_key="rank_genes_groups",
            n_genes=1,
            groups=["C"],
        )


def test_extract_rank_genes_groups_missing_params_groupby(ranked_adata):
    """A ranking without a stored grouping raises."""
    del ranked_adata.uns["rank_genes_groups"]["params"]["groupby"]

    with pytest.raises(Exception, match="missing 'groupby'"):
        _extract_rank_genes_groups(
            ranked_adata,
            ranking_key="rank_genes_groups",
            n_genes=1,
        )


def test_extract_rank_genes_groups_rejects_wrong_data_type():
    """An unsupported data type raises."""
    with pytest.raises(Exception, match="Expected"):
        _extract_rank_genes_groups(
            "not adata",
            ranking_key="rank_genes_groups",
            n_genes=1,
        )


def test_resolve_rank_genes_groups_args_rejects_keys(ranked_adata):
    """Passing keys together with a ranking raises."""
    with pytest.raises(ValueError, match="keys"):
        _resolve_rank_genes_groups_args(
            ranked_adata,
            ranking_key="rank_genes_groups",
            n_genes=1,
            groups=None,
            keys=["gene_a"],
            group_by=None,
        )


def test_resolve_rank_genes_groups_args_rejects_group_by_mismatch(ranked_adata):
    """A `group_by` differing from the ranking's grouping raises."""
    with pytest.raises(ValueError, match="does not match"):
        _resolve_rank_genes_groups_args(
            ranked_adata,
            ranking_key="rank_genes_groups",
            n_genes=1,
            groups=None,
            keys=None,
            group_by="other",
        )


def test_build_markers_frame(ranked_adata):
    """The markers frame holds the top genes, scores and ranks per group."""
    frame, groups, group_by = _build_markers_frame(
        ranked_adata,
        ranking_key="rank_genes_groups",
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
    """A ranking without scores raises."""
    del ranked_adata.uns["rank_genes_groups"]["scores"]

    with pytest.raises(Exception, match="missing"):
        _build_markers_frame(
            ranked_adata,
            ranking_key="rank_genes_groups",
            n_genes=1,
            groups=None,
            variable_column="gene",
            score_column="score",
            rank_column="rank",
            group_column="group",
        )


def test_marker_gene_helpers_preserve_group_order_and_deduplicate(ranked_adata):
    """Marker helpers keep group order and drop repeated genes."""
    assert cl.marker_genes(ranked_adata, n_genes=2) == [
        "gene_a",
        "shared",
        "gene_b",
    ]
    assert cl.marker_genes(ranked_adata, groups=["B", "A"], n_genes=2) == [
        "gene_b",
        "shared",
        "gene_a",
    ]
    assert cl.marker_genes_dict(ranked_adata, groups=["B", "A"], n_genes=2) == {
        "B": ["gene_b", "shared"],
        "A": ["gene_a", "shared"],
    }


@pytest.mark.parametrize("helper", [cl.marker_genes, cl.marker_genes_dict])
def test_marker_gene_helpers_accept_custom_ranking_key(ranked_adata, helper):
    """Marker helpers read a custom ranking key."""
    ranked_adata.uns["custom_ranking"] = ranked_adata.uns.pop("rank_genes_groups")

    result = helper(ranked_adata, groups=["A"], ranking_key="custom_ranking", n_genes=1)

    expected = ["gene_a"] if helper is cl.marker_genes else {"A": ["gene_a"]}
    assert result == expected


@pytest.mark.parametrize("helper", [cl.marker_genes, cl.marker_genes_dict])
def test_marker_gene_helpers_validate_inputs(ranked_adata, helper):
    """Marker helpers reject invalid inputs."""
    with pytest.raises(TypeError, match="groups"):
        helper(ranked_adata, groups="A")
    with pytest.raises(ValueError, match=">= 1"):
        helper(ranked_adata, n_genes=0)
    with pytest.raises(ValueError, match="exceeds"):
        helper(ranked_adata, n_genes=5)
    with pytest.raises(TypeError):
        helper(ranked_adata, n_genes=1.5)
    with pytest.raises(KeyNotFoundError, match="Groups"):
        helper(ranked_adata, groups=["missing"], n_genes=1)
    with pytest.raises(UnsupportedDataTypeError, match="Expected"):
        helper("not adata", n_genes=1)


@pytest.mark.parametrize("helper", [cl.marker_genes, cl.marker_genes_dict])
def test_marker_gene_helpers_validate_ranking_record(ranked_adata, helper):
    """Marker helpers reject a malformed ranking record."""
    with pytest.raises(KeyNotFoundError, match="No precomputed ranking found"):
        helper(ranked_adata, ranking_key="missing", n_genes=1)

    del ranked_adata.uns["rank_genes_groups"]["names"]
    with pytest.raises(KeyNotFoundError, match="missing 'names'"):
        helper(ranked_adata, n_genes=1)


def test_build_markers_frame_validation_paths(ranked_adata):
    """The markers frame builder rejects invalid inputs."""
    kwargs = {
        "ranking_key": "rank_genes_groups",
        "n_genes": 1,
        "groups": None,
        "variable_column": "gene",
        "score_column": "score",
        "rank_column": "rank",
        "group_column": "group",
    }

    with pytest.raises(ValueError, match=">= 1"):
        _build_markers_frame(ranked_adata, **(kwargs | {"n_genes": 0}))
    with pytest.raises(ValueError, match="exceeds"):
        _build_markers_frame(ranked_adata, **(kwargs | {"n_genes": 5}))
    with pytest.raises(KeyNotFoundError, match="Groups"):
        _build_markers_frame(ranked_adata, **(kwargs | {"groups": ["missing"]}))

    del ranked_adata.uns["rank_genes_groups"]["params"]["groupby"]
    with pytest.raises(KeyNotFoundError, match="missing 'groupby'"):
        _build_markers_frame(ranked_adata, **kwargs)

    with pytest.raises(UnsupportedDataTypeError, match="Expected"):
        _build_markers_frame("not adata", **kwargs)


def test_markers_plot_options(ranked_adata):
    """`markers` accepts its styling options."""
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
    """`markers` rejects unknown groups."""
    with pytest.raises(TypeError, match="groups"):
        cl.markers(ranked_adata, groups="A")


def test_markers_rejects_wrong_data_type():
    """`markers` rejects an unsupported data type."""
    with pytest.raises(Exception, match="Expected"):
        cl.markers("not adata")


def test_build_volcano_frame_uses_raw_pvalues(ranked_adata):
    """`use_adjusted_pvalue=False` reads the raw p-values."""
    frame = _build_volcano_frame(
        ranked_adata,
        "A",
        use_adjusted_pvalue=False,
        logfoldchange_threshold=1,
        pvalue_threshold=0.05,
    )

    assert frame["significance"].cast(str).to_list()[:3] == ["up", "down", "ns"]
    assert frame["negative_log_pvalue"].to_list()[0] == 3.0


def test_build_volcano_frame_errors(ranked_adata):
    """A missing group or ranking key raises `KeyNotFoundError`."""
    with pytest.raises(KeyNotFoundError, match="Group"):
        _build_volcano_frame(ranked_adata, "C")

    with pytest.raises(KeyNotFoundError, match="not found"):
        _build_volcano_frame(ranked_adata, "A", ranking_key="missing")

    with pytest.raises(Exception, match="Expected"):
        _build_volcano_frame("not adata", "A")


def test_build_volcano_frame_recomputes_missing_or_stale_ranking(ranked_adata, monkeypatch):
    """A missing or stale ranking is recomputed when `group_by` is given."""
    original_record = ranked_adata.uns.pop("rank_genes_groups")
    calls = []

    def fake_rank_genes_groups(data, *, groupby, key_added, **kwargs):
        calls.append((groupby, key_added, kwargs))
        data.uns[key_added] = original_record

    monkeypatch.setattr("scanpy.tl.rank_genes_groups", fake_rank_genes_groups)

    frame = _build_volcano_frame(
        ranked_adata,
        "A",
        group_by="cluster",
        rank_genes_kwargs={"method": "wilcoxon"},
    )

    assert frame.height > 0
    assert calls == [("cluster", "rank_genes_groups", {"method": "wilcoxon"})]

    ranked_adata.uns["rank_genes_groups"]["params"] = {
        "groupby": "other",
        "method": "wilcoxon",
    }
    _build_volcano_frame(
        ranked_adata,
        "A",
        group_by="cluster",
        rank_genes_kwargs={"method": "wilcoxon"},
    )
    assert len(calls) == 2


def test_build_volcano_frame_reuses_matching_ranking(ranked_adata, monkeypatch):
    """A matching ranking is reused, not recomputed."""

    def unexpected_recompute(*args, **kwargs):
        pytest.fail("matching differential-expression results should be reused")

    monkeypatch.setattr("scanpy.tl.rank_genes_groups", unexpected_recompute)

    frame = _build_volcano_frame(ranked_adata, "A", group_by="cluster")

    assert frame.height > 0


def test_volcano_plot_options(ranked_adata):
    """`volcano` accepts its styling options."""
    plot = cl.volcano(
        ranked_adata,
        "A",
        use_adjusted_pvalue=True,
        mapping=aes(shape="significance"),
        threshold_lines=True,
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
    """Unknown tooltip columns raise."""
    with pytest.raises(ValueError, match="tooltip columns"):
        cl.volcano(ranked_adata, "A", tooltips=["missing"])


def test_volcano_without_thresholds_or_labels(ranked_adata):
    """`volcano` renders without threshold lines or labels."""
    plot = cl.volcano(
        ranked_adata,
        "A",
        threshold_lines=False,
        top_n=0,
        tooltips="none",
        nonsignificant_subsample=None,
    )

    assert isinstance(plot, PlotSpec)


def test_volcanos_plot_options(ranked_adata):
    """`volcanos` accepts its styling and grid options."""
    plot = cl.volcanos(
        ranked_adata,
        ["A", "B"],
        threshold_lines=False,
        top_n=None,
        share_labels=True,
        share_axis=True,
        layers=geom_point(size=0.1),
        interactive=True,
        ncol=1,
    )

    assert isinstance(plot, SupPlotsSpec)


def test_volcanos_rejects_invalid_groups(ranked_adata):
    """`volcanos` rejects a non-sequence `groups`."""
    with pytest.raises(TypeError, match="groups"):
        cl.volcanos(ranked_adata, "A")


def test_volcanos_rejects_empty_groups(ranked_adata):
    """`volcanos` rejects an empty `groups`."""
    with pytest.raises(ValueError, match="empty"):
        cl.volcanos(ranked_adata, [])


def test_volcanos_rejects_non_string_groups(ranked_adata):
    """`volcanos` rejects groups that are not in the ranking."""
    with pytest.raises(KeyNotFoundError, match="Group"):
        cl.volcanos(ranked_adata, [1])


def test_ranking_key_reads_a_custom_ranking_on_real_data(adata):
    """Every ranking consumer reads a ranking stored under a custom `ranking_key`."""
    import scanpy as sc

    data = adata.copy()
    sc.tl.rank_genes_groups(data, groupby="leiden_res_0.02", key_added="rank_leiden")
    names = data.uns["rank_leiden"]["names"]

    assert cl.marker_genes(data, groups=["0"], ranking_key="rank_leiden", n_genes=2) == [
        str(gene) for gene in names["0"][:2]
    ]
    assert list(cl.marker_genes_dict(data, ranking_key="rank_leiden", n_genes=1)) == list(
        names.dtype.names
    )
    assert isinstance(cl.markers(data, ranking_key="rank_leiden", n_genes=3), SupPlotsSpec)
    assert isinstance(cl.volcano(data, "0", ranking_key="rank_leiden"), PlotSpec)
    assert isinstance(cl.volcanos(data, ["0", "1"], ranking_key="rank_leiden"), SupPlotsSpec)
    for fn in (cl.heatmap, cl.matrixplot, cl.dotplot, cl.stacked_violin, cl.tracksplot):
        plot = fn(data, markers=True, ranking_key="rank_leiden", n_genes=2)
        assert isinstance(plot, PlotSpec), fn.__name__


def test_ranking_key_is_ignored_without_markers(adata):
    """`ranking_key` is not read when `markers` is False."""
    plot = cl.dotplot(adata, ["CD3D", "LYZ"], "leiden_res_0.02", ranking_key="missing")
    assert isinstance(plot, PlotSpec)
