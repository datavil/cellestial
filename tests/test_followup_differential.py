"""Follow-up fixes for the open items of the 2026-11-18 differential audit."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from ggrust import aes, geom_point, ggplot

import cellestial as cl
from cellestial.single.differential.utilities import _build_volcano_frame
from cellestial.util.errors import (
    EmptySelectionError,
    InvalidArgumentError,
    InvalidComparisonError,
    KeyNotFoundError,
    MissingAestheticError,
)

NAMES = ["broad_up", "rare_up", "broad_down", "rare_down", "flat"]


@pytest.fixture
def ranked_adata() -> AnnData:
    """A two-group ranking with detection rates recorded, as `pts=True` stores them."""
    data = AnnData(X=np.ones((4, len(NAMES))))
    data.var_names = NAMES
    data.obs["cluster"] = pd.Categorical(["A", "A", "B", "B"])
    rank = {
        "names": np.rec.fromarrays([NAMES, NAMES[::-1]], names=["A", "B"]),
        "scores": np.rec.fromarrays([[5.0, 4.0, 3.0, 2.0, 1.0]] * 2, names=["A", "B"]),
        "logfoldchanges": np.rec.fromarrays(
            [[2.0, 9.0, -2.0, -9.0, 0.0], [0.0, -9.0, -2.0, 9.0, 2.0]], names=["A", "B"]
        ),
        "pvals": np.rec.fromarrays([[0.001] * 5] * 2, names=["A", "B"]),
        "pvals_adj": np.rec.fromarrays([[0.001] * 5] * 2, names=["A", "B"]),
        "params": {"groupby": "cluster", "method": "wilcoxon"},
        # indexed by feature in var order, which differs from the ranking order
        "pts": pd.DataFrame(
            {"A": [0.8, 0.01, 0.5, 0.0, 0.3], "B": [0.3, 0.0, 0.5, 0.01, 0.8]},
            index=NAMES,
        ),
    }
    data.uns["rank_genes_groups"] = rank
    return data


# ---- min_fraction_expressed ----


def test_volcano_frame_drops_features_below_the_detection_rate(ranked_adata):
    """Features detected in fewer of the group's cells than asked are dropped."""
    frame = _build_volcano_frame(ranked_adata, "A", min_fraction_expressed=0.1)

    assert sorted(frame["variable"]) == ["broad_down", "broad_up", "flat"]


def test_volcano_frame_reads_detection_rates_of_the_requested_group(ranked_adata):
    """The filter reads the requested group's column, matched by feature name."""
    frame = _build_volcano_frame(ranked_adata, "B", min_fraction_expressed=0.3)

    assert sorted(frame["variable"]) == ["broad_down", "broad_up", "flat"]


def test_volcano_frame_keeps_every_feature_by_default(ranked_adata):
    """Without `min_fraction_expressed` every feature stays."""
    assert _build_volcano_frame(ranked_adata, "A").height == len(NAMES)


def test_volcano_without_detection_rates_names_the_fix(ranked_adata):
    """No stored detection rates raises, naming `group_by` and `pts=True`."""
    del ranked_adata.uns["rank_genes_groups"]["pts"]

    with pytest.raises(KeyNotFoundError, match=r"group_by.*pts=True"):
        cl.volcano(ranked_adata, "A", min_fraction_expressed=0.1)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_volcano_rejects_detection_rate_outside_unit_interval(ranked_adata, value):
    """A fraction outside [0, 1] is rejected before any ranking runs."""
    with pytest.raises(InvalidArgumentError, match="min_fraction_expressed"):
        cl.volcano(ranked_adata, "A", min_fraction_expressed=value)


def test_volcano_group_by_records_detection_rates(adata):
    """With `group_by` the recompute records detection rates and the filter uses them."""
    data = adata.copy()
    data.raw = None
    data.uns.pop("rank_genes_groups", None)

    plot = cl.volcano(data, "B Cells", group_by="cell_type_lvl1", min_fraction_expressed=0.1)

    assert "pts" in data.uns["rank_genes_groups"]
    pts = data.uns["rank_genes_groups"]["pts"]["B Cells"]
    shown = cl.retrieve(plot)["variable"].to_list()
    assert (pts.loc[shown] >= 0.1).all()


def test_volcano_group_by_recomputes_a_ranking_without_detection_rates(ranked_adata, monkeypatch):
    """A matching ranking without detection rates is recomputed with `pts=True`."""
    import scanpy as sc

    del ranked_adata.uns["rank_genes_groups"]["pts"]
    calls = []
    monkeypatch.setattr(sc.tl, "rank_genes_groups", lambda data, **kwargs: calls.append(kwargs))

    with pytest.raises(KeyNotFoundError):
        _build_volcano_frame(ranked_adata, "A", group_by="cluster", min_fraction_expressed=0.1)

    assert len(calls) == 1
    assert calls[0]["pts"] is True


def test_volcanos_forwards_the_detection_rate(ranked_adata):
    """Every panel of the grid is filtered."""
    grid = cl.volcanos(ranked_adata, ["A", "B"], min_fraction_expressed=0.1, top_n=0)
    shown = [sorted(figure["data"]["variable"]) for figure in grid.as_dict()["figures"]]

    assert shown == [["broad_down", "broad_up", "flat"]] * 2


# ---- KeyNotFoundError with close matches ----


def test_volcano_missing_group_suggests_a_close_match(ranked_adata):
    """An unknown group raises `KeyNotFoundError` with a case-insensitive close match."""
    with pytest.raises(KeyNotFoundError, match="did you mean 'A'"):
        cl.volcano(ranked_adata, "a")


def test_volcano_missing_key_suggests_a_close_match(ranked_adata):
    """An unknown ranking key raises `KeyNotFoundError` with a close match."""
    with pytest.raises(KeyNotFoundError, match="did you mean 'rank_genes_groups'"):
        cl.volcano(ranked_adata, "A", key="rank_gene_groups")


# ---- label ties ----


def test_volcano_label_ties_resolve_by_feature_name():
    """Labels tied on p-value and fold change are picked by feature name."""
    names = [f"gene_{letter}" for letter in "edcba"]
    data = AnnData(X=np.ones((2, len(names))))
    data.uns["rank_genes_groups"] = {
        "names": np.rec.fromarrays([names], names=["A"]),
        "scores": np.rec.fromarrays([[1.0] * 5], names=["A"]),
        "logfoldchanges": np.rec.fromarrays([[2.0] * 5], names=["A"]),
        "pvals": np.rec.fromarrays([[0.0] * 5], names=["A"]),
        "pvals_adj": np.rec.fromarrays([[0.0] * 5], names=["A"]),
        "params": {"groupby": "cluster"},
    }
    plot = cl.volcano(data, "A", top_n=2)
    label_layer = plot.as_dict()["layers"][-1]

    assert sorted(label_layer["data"]["variable"]) == ["gene_a", "gene_b"]


# ---- empty selections and ncol ----


def test_volcanos_rejects_empty_groups(ranked_adata):
    """An empty `groups` raises `EmptySelectionError` naming the parameter."""
    with pytest.raises(EmptySelectionError, match="groups"):
        cl.volcanos(ranked_adata, [])


def test_marker_genes_rejects_empty_groups(ranked_adata):
    """An empty `groups` raises instead of returning an empty list."""
    with pytest.raises(EmptySelectionError, match="groups"):
        cl.marker_genes(ranked_adata, groups=[])


@pytest.mark.parametrize(
    "call",
    [
        lambda data: cl.volcanos(data, ["A", "B"], ncol=0),
        lambda data: cl.markers(data, ncol=0),
    ],
    ids=["volcanos", "markers"],
)
def test_plural_functions_reject_ncol_below_one(ranked_adata, call):
    """`ncol` below 1 raises `InvalidArgumentError`."""
    with pytest.raises(InvalidArgumentError, match="ncol"):
        call(ranked_adata)


# ---- markers frame keeps n_genes finite scores ----


def test_markers_frame_skips_non_finite_scores_before_taking_the_top(ranked_adata):
    """A non-finite score does not cost a panel one of its `n_genes` genes or a rank."""
    from cellestial.single.differential.utilities import _build_markers_frame

    ranked_adata.uns["rank_genes_groups"]["scores"]["A"][0] = np.nan
    frame, _, _ = _build_markers_frame(
        ranked_adata,
        key=True,
        n_genes=3,
        groups=["A"],
        variable_column="variable",
        score_column="score",
        rank_column="rank",
        group_column="group",
    )

    assert frame["variable"].to_list() == ["rare_up", "broad_down", "rare_down"]
    assert frame["rank"].to_list() == [0, 1, 2]


# ---- _require_categories hint ----


def test_require_categories_suggests_a_close_match():
    """Unknown group values suggest a close match, ignoring case."""
    from cellestial.util.utilities import _require_categories

    with pytest.raises(KeyNotFoundError, match="did you mean 'B Cells'"):
        _require_categories(
            ["B cells"],
            ["B Cells", "Monocytes"],
            parameter="groups",
            column="cell_type",
            column_parameter="`group_by` column",
        )


# ---- bracket ----


def _constant_plot() -> ggplot:
    frame = pl.DataFrame({"group": ["a", "b", "c"] * 4, "value": [2.0] * 12})
    return ggplot(frame) + geom_point(aes(x="group", y="value"))


def test_bracket_on_constant_data_stacks_above_the_data():
    """Constant data still gets brackets stacked above it."""
    plot = _constant_plot() + cl.bracket()
    positions = plot.as_dict()["layers"][-1]["label_y"]

    assert len(positions) == 3
    assert positions[0] > 2.0
    assert positions[0] < positions[1] < positions[2]


def test_bracket_missing_y_column_names_the_column():
    """A `y` column absent from the plot raises `MissingAestheticError` with a close match."""
    with pytest.raises(MissingAestheticError, match="'missing'"):
        _ = _constant_plot() + cl.bracket(y="missing", x="group")
    with pytest.raises(MissingAestheticError, match="did you mean 'value'"):
        _ = _constant_plot() + cl.bracket(y="valeu")


def test_bracket_missing_x_column_names_the_column():
    """An `x` column absent from the plot raises `MissingAestheticError`."""
    with pytest.raises(MissingAestheticError, match="'grp'"):
        _ = _constant_plot() + cl.bracket(x="grp")


def test_bracket_without_finite_values_raises_a_comparison_error():
    """No finite values raises `InvalidComparisonError`."""
    frame = pl.DataFrame({"group": ["a", "b"] * 5, "value": [float("nan")] * 10})
    plot = ggplot(frame) + geom_point(aes(x="group", y="value"))

    with pytest.raises(InvalidComparisonError, match="finite"):
        _ = plot + cl.bracket()


def test_bracket_with_too_few_observations_raises_a_comparison_error():
    """Fewer than 2 observations per group raises `InvalidComparisonError`."""
    frame = pl.DataFrame({"group": ["a", "b"], "value": [1.0, 2.0]})
    plot = ggplot(frame) + geom_point(aes(x="group", y="value"))

    with pytest.raises(InvalidComparisonError, match="at least 2"):
        _ = plot + cl.bracket()


def test_bracket_rejects_an_unknown_test():
    """An unknown test name raises `InvalidArgumentError`."""
    with pytest.raises(InvalidArgumentError, match="`test`"):
        cl.bracket(test="anova")


def test_volcano_detection_rate_with_duplicate_feature_names_names_the_fix(ranked_adata):
    """Duplicate feature names cannot be matched to detection rates; the error says so."""
    rank = ranked_adata.uns["rank_genes_groups"]
    rank["pts"].index = ["broad_up", "broad_up", "broad_down", "rare_down", "flat"]

    with pytest.raises(InvalidArgumentError, match="unique"):
        _build_volcano_frame(ranked_adata, "A", min_fraction_expressed=0.1)
