"""Regression and edge-case tests from the 2026-11-18 audit of differential plots and brackets."""

import numpy as np
import polars as pl
import pytest
from anndata import AnnData
from ggrust import aes, geom_point, ggplot

import cellestial as cl
from cellestial.single.differential.utilities import _build_volcano_frame
from cellestial.util.errors import EmptySelectionError, InvalidArgumentError, KeyNotFoundError
from cellestial.util.utilities import _filter_groups


@pytest.fixture
def boundary_adata() -> AnnData:
    """A ranking whose features sit exactly on, and just past, the default thresholds."""
    names = ["on_up", "on_down", "on_pvalue", "past_up", "past_down", "zero_fold"]
    data = AnnData(X=np.ones((4, len(names))))
    data.obs["cluster"] = ["A", "A", "B", "B"]
    data.uns["rank_genes_groups"] = {
        "names": np.rec.fromarrays([names], names=["A"]),
        "scores": np.rec.fromarrays([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], names=["A"]),
        "logfoldchanges": np.rec.fromarrays([[1.0, -1.0, 3.0, 1.5, -1.5, 0.0]], names=["A"]),
        "pvals": np.rec.fromarrays([[0.01, 0.01, 0.05, 0.0, 0.01, 0.0]], names=["A"]),
        "pvals_adj": np.rec.fromarrays([[0.01, 0.01, 0.05, 0.0, 0.01, 0.0]], names=["A"]),
        "params": {"groupby": "cluster", "method": "wilcoxon"},
    }
    return data


def _significance(frame: pl.DataFrame) -> dict[str, str]:
    return dict(zip(frame["variable"], frame["significance"].cast(pl.String), strict=True))


# ---- volcano classification at the boundaries ----


def test_volcano_frame_classifies_threshold_boundaries(boundary_adata):
    """`|logFC| == threshold` counts as regulated, `p == threshold` does not."""
    labels = _significance(_build_volcano_frame(boundary_adata, "A"))

    assert labels == {
        "on_up": "up",
        "on_down": "down",
        "on_pvalue": "ns",
        "past_up": "up",
        "past_down": "down",
        "zero_fold": "ns",
    }


def test_volcano_frame_keeps_zero_pvalues_as_infinite(boundary_adata):
    """`-log10(0)` is `inf` in the frame; the plot caps it, the frame does not drop it."""
    frame = _build_volcano_frame(boundary_adata, "A")
    value = frame.filter(pl.col("variable") == "past_up")["negative_log_pvalue"].item()

    assert value == float("inf")


def test_volcano_caps_zero_pvalues_above_the_finite_maximum(boundary_adata):
    """Underflowed p-values are drawn at `1.05 * max + 1`, never as `inf`."""
    plot = cl.volcano(boundary_adata, "A", nonsignificant_subsample=None, top_n=0)
    heights = dict(
        zip(
            plot.as_dict()["data"]["variable"],
            plot.as_dict()["data"]["negative_log_pvalue"],
            strict=True,
        )
    )
    finite_max = -np.log10(0.01)

    assert heights["past_up"] == pytest.approx(finite_max * 1.05 + 1.0, abs=1e-3)
    assert all(np.isfinite(value) for value in heights.values())


@pytest.mark.parametrize("pvalue_threshold", [0.0, -0.05, 1.5])
def test_volcano_rejects_pvalue_threshold_outside_unit_interval(boundary_adata, pvalue_threshold):
    """A threshold outside (0, 1] has no meaning and used to fail in the renderer."""
    with pytest.raises(InvalidArgumentError, match="pvalue_threshold"):
        cl.volcano(boundary_adata, "A", pvalue_threshold=pvalue_threshold)


def test_volcano_rejects_negative_logfoldchange_threshold(boundary_adata):
    """A negative threshold would call features with negative fold change `up`."""
    with pytest.raises(InvalidArgumentError, match="logfoldchange_threshold"):
        cl.volcano(boundary_adata, "A", logfoldchange_threshold=-1.0)


def test_volcano_rejects_negative_top_n(boundary_adata):
    """`head(-1)` keeps all rows but the last, so a negative count must not pass."""
    with pytest.raises(InvalidArgumentError, match="top_n"):
        cl.volcano(boundary_adata, "A", top_n=-1)


def test_volcano_rejects_negative_nonsignificant_subsample(boundary_adata):
    """A negative sample size used to crash inside polars."""
    with pytest.raises(InvalidArgumentError, match="nonsignificant_subsample"):
        cl.volcano(boundary_adata, "A", nonsignificant_subsample=-1)


def test_volcano_names_missing_ranking_fields(boundary_adata):
    """A ranking without fold changes (e.g. logistic regression) fails with a clear message."""
    del boundary_adata.uns["rank_genes_groups"]["logfoldchanges"]

    with pytest.raises(KeyNotFoundError, match="logfoldchanges"):
        cl.volcano(boundary_adata, "A")


def test_volcanos_computes_the_ranking_once(boundary_adata, monkeypatch):
    """Arguments the stored parameters do not record must not recompute per panel."""
    record = boundary_adata.uns["rank_genes_groups"]
    record["names"] = np.rec.fromarrays([record["names"]["A"]] * 2, names=["A", "B"])
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        record[field] = np.rec.fromarrays([record[field]["A"]] * 2, names=["A", "B"])
    calls = []

    def fake_rank_genes_groups(data, *, groupby, key_added, **kwargs):
        calls.append(kwargs)
        data.uns[key_added] = record

    monkeypatch.setattr("scanpy.tl.rank_genes_groups", fake_rank_genes_groups)

    cl.volcanos(
        boundary_adata,
        ["A", "B"],
        group_by="cluster",
        rank_genes_kwargs={"method": "wilcoxon", "pts": True},
    )

    assert len(calls) == 1


# ---- markers ----


def test_markers_rejects_empty_groups(boundary_adata):
    """An empty selection names `groups` instead of failing in polars."""
    with pytest.raises(EmptySelectionError, match="groups"):
        cl.markers(boundary_adata, groups=[])


def test_markers_frame_keeps_ranking_order(boundary_adata):
    """Panels list genes and ranks in the stored ranking order."""
    plot = cl.markers(boundary_adata, n_genes=3)
    frame = pl.DataFrame(plot.as_dict()["figures"][0]["data"])

    assert frame["variable"].to_list() == ["on_up", "on_down", "on_pvalue"]
    assert frame["rank"].to_list() == [0, 1, 2]


# ---- _filter_groups with null categories ----


def _null_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"group": pl.Series(["a", "b", None, "c"], dtype=pl.Categorical), "value": [1, 2, 3, 4]}
    )


def test_filter_groups_drop_keeps_null_rows():
    """Dropping a category must not also drop rows whose category is missing."""
    frame = _filter_groups(_null_frame(), column="group", drop=["a"])

    assert frame["value"].to_list() == [2, 3, 4]


def test_filter_groups_keep_excludes_null_rows():
    """Keeping categories drops rows whose category is missing."""
    frame = _filter_groups(_null_frame(), column="group", groups=["a", "c"])

    assert frame["value"].to_list() == [1, 4]


def test_filter_groups_rejects_null_as_a_category():
    """A missing category cannot be selected by its string form."""
    with pytest.raises(KeyNotFoundError, match="not found"):
        _filter_groups(_null_frame(), column="group", groups=["None"])


# ---- bracket geometry ----


def _bracket_base() -> ggplot:
    frame = pl.DataFrame(
        {
            "group": ["a", "b", "c"] * 10,
            "value": [float(index % 7) + (index % 3) for index in range(30)],
        }
    )
    return ggplot(frame) + geom_point(aes(x="group", y="value"))


def test_bracket_below_the_data_never_shrinks_the_canvas():
    """An explicit low `y_position` must not give the plot a negative height."""
    plot = _bracket_base() + cl.bracket(y_position=-50.0, y_step=0.01)
    width, height = plot.size

    assert width == 600
    assert height >= 400


def test_bracket_rejects_a_column_without_finite_values():
    """No finite values raises the comparison error, not a `TypeError`."""
    frame = pl.DataFrame({"group": ["a", "b"] * 5, "value": [float("nan")] * 10})
    plot = ggplot(frame) + geom_point(aes(x="group", y="value"))

    with pytest.raises(ValueError, match="finite"):
        _ = plot + cl.bracket()
