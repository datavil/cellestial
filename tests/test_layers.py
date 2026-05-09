import polars as pl
import pytest
from lets_plot import aes, geom_point, ggplot
from lets_plot.plot.core import PlotSpec

import cellestial as cl
from cellestial.layers import DeferredLayer
from cellestial.layers.bracket import _compute_bracket_frame, _expand_comparisons
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


def test_stream_requires_velocity(adata):
    # pbmc3k fixture has no velocity columns; stream should raise at `+` time.
    umap = cl.umap(adata)
    with pytest.raises(KeyError, match="Velocity columns not found"):
        _ = umap + cl.stream()
