import sys
import types

import numpy as np
import polars as pl
import pytest
from lets_plot import aes, geom_point, ggplot
from lets_plot.plot.core import PlotSpec

import cellestial as cl
from cellestial.layers import DeferredLayer
from cellestial.layers.bracket import _compute_bracket_frame, _correct_pvalues, _expand_comparisons
from cellestial.layers.ondata_legend import _compute_label_positions
from cellestial.util.errors import InvalidComparisonError


def test_arrow_axis_returns_deferred(adata):
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis()
    assert isinstance(arrow, DeferredLayer)
    # Addition should succeed
    combined = umap + arrow
    assert isinstance(combined, PlotSpec)


def test_arrow_axis_customization(adata):
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis(length=0.20, color="dark_violet")
    assert isinstance(arrow, DeferredLayer)
    combined = umap + arrow
    assert isinstance(combined, PlotSpec)


def test_arrow_axis_explicit_plot_kwarg(adata):
    # Explicit `plot=` pins the data source regardless of the receiving plot.
    umap = cl.umap(adata, "CD14")
    arrow = cl.arrow_axis(plot=umap, length=0.20)
    assert isinstance(arrow, DeferredLayer)
    # Even when added to an empty ggplot, the data source stays `umap`.
    combined = ggplot() + arrow
    assert isinstance(combined, PlotSpec)


def test_ondata_legend_uses_group_median_positions():
    frame = pl.DataFrame(
        {
            "x": [0.0, 0.0, 0.0, 100.0, 10.0, 20.0, 30.0],
            "y": [1.0, 1.0, 1.0, 101.0, 2.0, 4.0, 6.0],
            "group": ["a", "a", "a", "a", "b", "b", "b"],
        }
    )

    positions = _compute_label_positions(frame, x="x", y="y", group_by="group").sort(
        "group"
    )

    assert positions["x"].to_list() == [0.0, 20.0]
    assert positions["y"].to_list() == [1.0, 4.0]


def test_ondata_legend_layer_options_and_missing_aesthetics():
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 0.0, 1.0],
            "group": ["a", "a", "b", "b"],
        }
    )
    source = ggplot(frame) + geom_point(aes(x="x", y="y", color="group"))

    combined = ggplot() + cl.ondata_legend(
        plot=source,
        x="x",
        y="y",
        group_by="group",
        label=True,
        repel=False,
    )

    assert isinstance(combined, PlotSpec)
    with pytest.raises(Exception, match="`x`"):
        _ = (ggplot(frame) + geom_point(aes(y="y", color="group"))) + cl.ondata_legend()
    with pytest.raises(Exception, match="`y`"):
        _ = (ggplot(frame) + geom_point(aes(x="x", color="group"))) + cl.ondata_legend()
    with pytest.raises(Exception, match="`group_by`"):
        _ = (ggplot(frame) + geom_point(aes(x="x", y="y"))) + cl.ondata_legend()


def test_cluster_outlines_single_group(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(groups="B Cells")
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_multiple_groups(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(groups=["B Cells", "Erythroid"])
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_nested_groups(adata, group_key):
    umap = cl.umap(adata, key=group_key)
    outline = cl.cluster_outlines(groups=[["Lymphocytes", "Monocytes"], "B Cells"])
    combined = umap + outline
    assert isinstance(combined, PlotSpec)


def test_cluster_outlines_invalid_group_raises(adata, group_key):
    # DeferredLayer defers construction to `+` time, so the error fires there.
    umap = cl.umap(adata, key=group_key)
    with pytest.raises(Exception):
        _ = umap + cl.cluster_outlines(groups="NOT_A_GROUP_xyz")


def test_cluster_outlines_too_few_points_raises():
    """Raise clearly when no outline can be computed for small groups."""
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "group": ["a", "a", "a", "a"],
        }
    )
    plot = ggplot(frame) + geom_point(aes(x="x", y="y", color="group"))
    with pytest.raises(ValueError, match="No cluster outline could be computed"):
        _ = plot + cl.cluster_outlines(groups="a")


def test_bracket_too_few_observations_raises():
    """Raise clearly when every comparison has too few observations."""
    frame = pl.DataFrame(
        {
            "group": ["a", "b"],
            "value": [1.0, 2.0],
        }
    )
    plot = ggplot(frame) + geom_point(aes(x="group", y="value"))
    with pytest.raises(ValueError, match="No valid group comparisons available"):
        _ = plot + cl.bracket()


def test_bracket_preserves_group_order_for_default_comparisons():
    """Preserve first-seen group order when generating default comparisons."""
    frame = pl.DataFrame(
        {
            "group": ["b", "b", "a", "a", "c", "c"],
            "value": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        }
    )

    brackets = _compute_bracket_frame(
        frame,
        x="group",
        y="value",
        comparisons=None,
        test="mannwhitney",
        alternative="two-sided",
        correction="none",
        label="stars",
        label_format=".3g",
        prefix="",
        prefix_style="=",
        separator=" ",
        threshold=None,
        y_position=None,
        y_step=None,
        y_padding=0.08,
    )

    assert list(zip(brackets["xmin"], brackets["xmax"], strict=True)) == [
        ("b", "a"),
        ("b", "c"),
        ("a", "c"),
    ]


@pytest.mark.parametrize("comparisons", [[("a",)], ["ab"]])
def test_bracket_comparisons_must_be_pairs(comparisons):
    """Raise clearly when a comparison is not a pair of groups."""
    with pytest.raises(InvalidComparisonError, match="exactly two groups"):
        _expand_comparisons(comparisons, ["a", "b"])


def test_bracket_comparisons_unknown_group_raises():
    """Raise clearly when comparisons reference a missing group."""
    with pytest.raises(InvalidComparisonError, match="was not found"):
        _expand_comparisons([("a", "z")], ["a", "b"])


def test_bracket_comparisons_wildcard_still_expands():
    """Keep group-vs-rest wildcard expansion behavior."""
    assert _expand_comparisons([("a", "*")], ["a", "b", "c"]) == [
        ("a", "b"),
        ("a", "c"),
    ]


def test_bracket_comparisons_left_wildcard_expands():
    """Expand rest-vs-group wildcard comparisons."""
    assert _expand_comparisons([("*", "c")], ["a", "b", "c"]) == [
        ("a", "c"),
        ("b", "c"),
    ]


def test_bracket_comparisons_double_wildcard_expands_all_pairs():
    """Expand double wildcard comparisons to all pairwise groups."""
    assert _expand_comparisons([("*", "*")], ["a", "b", "c"]) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def test_bracket_comparisons_wildcard_deduplicates_pairs():
    """Avoid duplicate unordered pairs when wildcards overlap."""
    assert _expand_comparisons([("a", "*"), ("*", "a")], ["a", "b", "c"]) == [
        ("a", "b"),
        ("a", "c"),
    ]


def test_bracket_frame_accepts_group_vs_rest_comparisons():
    """Build brackets from group-vs-rest comparison syntax."""
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c"],
            "value": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        }
    )

    brackets = _compute_bracket_frame(
        frame,
        x="group",
        y="value",
        comparisons=[("a", "*")],
        test="mannwhitney",
        alternative="two-sided",
        correction="none",
        label="stars",
        label_format=".3g",
        prefix="",
        prefix_style="=",
        separator=" ",
        threshold=None,
        y_position=None,
        y_step=None,
        y_padding=0.08,
    )

    assert list(zip(brackets["xmin"], brackets["xmax"], strict=True)) == [
        ("a", "b"),
        ("a", "c"),
    ]


def test_correct_pvalues_methods_and_invalid_method():
    pvalues = np.array([0.01, 0.02, 0.2])

    assert _correct_pvalues(pvalues, "none").tolist() == pvalues.tolist()
    assert _correct_pvalues(pvalues, "bonferroni").round(3).tolist() == [0.03, 0.06, 0.6]
    assert _correct_pvalues(pvalues, "fdr_bh").round(3).tolist() == [0.03, 0.03, 0.2]
    with pytest.raises(ValueError, match="correction"):
        _correct_pvalues(pvalues, "bad")


def test_bracket_frame_ttest_pvalue_labels_and_threshold_empty():
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b", "b"],
            "value": [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
        }
    )

    brackets = _compute_bracket_frame(
        frame,
        x="group",
        y="value",
        comparisons=[("a", "b")],
        test="ttest",
        alternative="less",
        correction="bonferroni",
        label=["stars", "pvalue", "padj"],
        label_format=".2g",
        prefix="p",
        prefix_style="<",
        separator="\n",
        threshold=None,
        y_position=10.0,
        y_step=2.0,
        y_padding=0.1,
    )
    filtered = _compute_bracket_frame(
        frame,
        x="group",
        y="value",
        comparisons=[("a", "b")],
        test="ttest",
        alternative="less",
        correction="bonferroni",
        label="pvalue",
        label_format=".2g",
        prefix="p",
        prefix_style=None,
        separator=" ",
        threshold=1e-12,
        y_position=None,
        y_step=None,
        y_padding=0.1,
    )

    assert brackets["y"].to_list() == [10.0]
    assert "p <" in brackets["label"][0]
    assert filtered.is_empty()


def test_bracket_frame_validation_paths():
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [1.0, 1.1, 2.0, 2.1],
        }
    )
    kwargs = {
        "frame": frame,
        "x": "group",
        "y": "value",
        "comparisons": [("a", "b")],
        "test": "mannwhitney",
        "alternative": "two-sided",
        "correction": "none",
        "label_format": ".3g",
        "prefix": "",
        "prefix_style": "=",
        "separator": " ",
        "threshold": None,
        "y_position": None,
        "y_step": None,
        "y_padding": 0.1,
    }

    with pytest.raises(ValueError, match="test"):
        _compute_bracket_frame(**(kwargs | {"test": "bad", "label": "stars"}))
    with pytest.raises(ValueError, match="label"):
        _compute_bracket_frame(**(kwargs | {"label": ["bad"]}))
    with pytest.raises(ValueError, match="at least one"):
        _compute_bracket_frame(**(kwargs | {"label": []}))
    with pytest.raises(ValueError, match="prefix_style"):
        _compute_bracket_frame(**(kwargs | {"label": "pvalue", "prefix_style": "bad"}))


def test_bracket_layer_explicit_plot_and_mapping():
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [1.0, 1.1, 2.0, 2.1],
            "bracket_color": ["x", "x", "x", "x"],
        }
    )
    source = ggplot(frame) + geom_point(aes(x="group", y="value"))

    combined = ggplot() + cl.bracket(
        plot=source,
        x="group",
        y="value",
        correction="none",
        label="pvalue",
        mapping=aes(color="label"),
        label_size=6,
        threshold=1,
    )

    assert isinstance(combined, PlotSpec)


def test_bracket_layer_requires_x_and_y_aesthetics():
    frame = pl.DataFrame({"group": ["a", "a"], "value": [1.0, 2.0]})

    with pytest.raises(Exception, match="`x`"):
        _ = (ggplot(frame) + geom_point(aes(y="value"))) + cl.bracket()
    with pytest.raises(Exception, match="`y`"):
        _ = (ggplot(frame) + geom_point(aes(x="group"))) + cl.bracket()


def test_stream_requires_velocity(adata):
    # pbmc3k fixture has no velocity columns; stream should raise at `+` time.
    umap = cl.umap(adata)
    with pytest.raises(KeyError, match="Velocity columns not found"):
        _ = umap + cl.stream()


def test_stream_requires_x_and_y_aesthetics():
    frame = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})

    with pytest.raises(Exception, match="`x`"):
        _ = (ggplot(frame) + geom_point(aes(y="y"))) + cl.stream()
    with pytest.raises(Exception, match="`y`"):
        _ = (ggplot(frame) + geom_point(aes(x="x"))) + cl.stream()


def test_stream_velocity_key_requires_dimension_numbers():
    frame = pl.DataFrame(
        {
            "umap_x": [0.0, 1.0],
            "umap_y": [0.0, 1.0],
            "velocity_1": [0.1, 0.2],
            "velocity_2": [0.1, 0.2],
        }
    )
    plot = ggplot(frame) + geom_point(aes(x="umap_x", y="umap_y"))

    with pytest.raises(ValueError, match="Could not parse dimension number"):
        _ = plot + cl.stream(velocity_key="velocity_")


def test_stream_builds_layers_with_mocked_velocity_grid(monkeypatch):
    frame = pl.DataFrame(
        {
            "X_UMAP1": [0.0, 0.0, 1.0, 1.0],
            "X_UMAP2": [0.0, 1.0, 0.0, 1.0],
            "VELOCITY_UMAP1": [0.2, 0.2, 0.2, 0.2],
            "VELOCITY_UMAP2": [0.1, 0.1, 0.1, 0.1],
        }
    )
    plot = ggplot(frame) + geom_point(aes(x="X_UMAP1", y="X_UMAP2"))

    def compute_velocity_on_grid(**kwargs):
        grid = np.linspace(0.0, 1.0, 5)
        velocity = np.ones((5, 5))
        return (grid, grid), (velocity, velocity)

    fake_module = types.SimpleNamespace(compute_velocity_on_grid=compute_velocity_on_grid)
    monkeypatch.setitem(sys.modules, "scvelo.plotting.velocity_embedding_grid", fake_module)

    combined = plot + cl.stream(arrow_kwargs={"alpha": 0.5}, density=0.5)

    assert isinstance(combined, PlotSpec)
