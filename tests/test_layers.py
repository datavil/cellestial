import numpy as np
import polars as pl
import pytest
from ggrust import aes, geom_point, ggplot
from ggrust.plot.core import PlotSpec

import cellestial as cl
from cellestial.layers.bracket import _bracket_layout, _expand_comparisons
from cellestial.layers.ondata_legend import _compute_label_positions
from cellestial.util import retrieve
from cellestial.util.errors import InvalidComparisonError, MissingAestheticError


def test_arrow_axis_adds_to_plot(adata):
    umap = cl.umap(adata, "CD14")
    combined = umap + cl.arrow_axis()
    assert isinstance(combined, PlotSpec)


def test_arrow_axis_customization(adata):
    umap = cl.umap(adata, "CD14")
    combined = umap + cl.arrow_axis(length=0.20, color="dark_violet")
    theme = combined.as_dict()["theme"]
    for key in ("axis_line_x", "axis_line_y"):
        assert theme[key]["extent"] == 0.20
        assert theme[key]["color"] == "dark_violet"


def test_arrow_axis_forwards_linetype_and_arrow_kwargs(adata):
    umap = cl.umap(adata, "CD14")
    combined = umap + cl.arrow_axis(arrow_kwargs={"type": "closed"}, linetype="dashed")
    theme = combined.as_dict()["theme"]
    for key in ("axis_line_x", "axis_line_y"):
        assert theme[key]["linetype"] == "dashed"
        assert theme[key]["arrow"]["type"] == "closed"


def test_ondata_legend_uses_group_median_positions():
    frame = pl.DataFrame(
        {
            "x": [0.0, 0.0, 0.0, 100.0, 10.0, 20.0, 30.0],
            "y": [1.0, 1.0, 1.0, 101.0, 2.0, 4.0, 6.0],
            "group": ["a", "a", "a", "a", "b", "b", "b"],
        }
    )

    positions = _compute_label_positions(frame, x="x", y="y", group_by="group").sort("group")

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


def _collect_keys(obj, name):
    """Recursively gather every value stored under `name` in a spec dict."""
    found = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == name:
                found.append(value)
            found.extend(_collect_keys(value, name))
    elif isinstance(obj, list):
        for value in obj:
            found.extend(_collect_keys(value, name))
    return found


def _build_ondata(**kwargs):
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 0.0, 1.0],
            "group": ["a", "a", "b", "b"],
        }
    )
    source = ggplot(frame) + geom_point(aes(x="x", y="y", color="group"))
    layer = cl.ondata_legend(plot=source, x="x", y="y", group_by="group", **kwargs)
    return layer._builder(source).as_dict()


def test_ondata_legend_halo_reaches_text_geom():
    spec = _build_ondata(halo_width=3.0, halo_color="black")
    assert _collect_keys(spec, "halo_width") == [3.0]
    assert _collect_keys(spec, "halo_color") == ["black"]


def test_ondata_legend_default_halo_width():
    spec = _build_ondata()
    assert _collect_keys(spec, "halo_width") == [0.5]


def test_ondata_legend_label_ignores_halo():
    # `geom_label` has no halo; halo params must not leak into the label geom.
    spec = _build_ondata(label=True, halo_width=3.0, halo_color="black")
    assert _collect_keys(spec, "halo_width") == []
    assert _collect_keys(spec, "halo_color") == []


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


def test_cluster_outlines_requires_coordinates_and_group_mapping():
    frame = pl.DataFrame(
        {
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 1.0, 1.0],
            "group": ["a", "a", "a", "a"],
        }
    )

    with pytest.raises(MissingAestheticError, match="`x`"):
        _ = (ggplot(frame) + geom_point(aes(y="y", color="group"))) + cl.cluster_outlines("a")
    with pytest.raises(MissingAestheticError, match="`y`"):
        _ = (ggplot(frame) + geom_point(aes(x="x", color="group"))) + cl.cluster_outlines("a")
    with pytest.raises(MissingAestheticError, match="`group_by`"):
        _ = (ggplot(frame) + geom_point(aes(x="x", y="y"))) + cl.cluster_outlines("a")


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


def test_bracket_grows_plot_with_stack_height():
    """Grow the canvas per bracket so the data panel keeps its height."""
    frame = pl.DataFrame(
        {
            "group": ["a", "b", "c", "d"] * 5,
            "value": [float(value) for value in range(20)],
        }
    )
    plot = ggplot(frame) + geom_point(aes(x="group", y="value"))
    two_brackets = plot + cl.bracket(comparisons=[("a", "b"), ("a", "c")])
    six_brackets = plot + cl.bracket()
    two_width, two_height = two_brackets.size
    six_width, six_height = six_brackets.size
    assert two_width == six_width == 600
    assert 400 < two_height < six_height
    # each extra bracket adds the same pixel step, whatever the data range
    assert six_height - two_height == pytest.approx(4 * 22)


def test_bracket_preserves_group_order_for_default_comparisons():
    """Preserve first-seen group order when generating default comparisons."""
    frame = pl.DataFrame(
        {
            "group": ["b", "b", "a", "a", "c", "c"],
            "value": [1.0, 1.1, 2.0, 2.1, 3.0, 3.1],
        }
    )

    _, pairs, _ = _bracket_layout(
        frame,
        x="group",
        y="value",
        comparisons=None,
        y_position=None,
        y_step=None,
        y_padding=0.08,
    )

    assert pairs == [
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

    _, pairs, _ = _bracket_layout(
        frame,
        x="group",
        y="value",
        comparisons=[("a", "*")],
        y_position=None,
        y_step=None,
        y_padding=0.08,
    )

    assert pairs == [
        ("a", "b"),
        ("a", "c"),
    ]


def test_bracket_layout_explicit_positions():
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c"],
            "value": [1.0, 1.1, 5.0, 5.1, 3.0, 3.1],
        }
    )

    _, _, positions = _bracket_layout(
        frame,
        x="group",
        y="value",
        comparisons=None,
        y_position=10.0,
        y_step=2.0,
        y_padding=0.1,
    )

    assert positions == [10.0, 12.0, 14.0]


@pytest.mark.parametrize(
    "kwargs",
    [{"test": "student"}, {"correction": "holm"}, {"label": "padj"}],
)
def test_bracket_rejects_unknown_options(kwargs):
    with pytest.raises(ValueError, match=f"`{next(iter(kwargs))}`"):
        cl.bracket(**kwargs)


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
        mapping=aes(color="bracket_color"),
        label_size=6,
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


def test_stream_builds_layers_from_a_velocity_field():
    # 400 cells on a jittered square, rotating about the centre: enough cells for
    # the default `n_neighbors` (n / 50) of ggrust's velocity grid.
    rng = np.random.default_rng(0)
    points = rng.uniform(-1.0, 1.0, size=(400, 2))
    frame = pl.DataFrame(
        {
            "X_UMAP1": points[:, 0],
            "X_UMAP2": points[:, 1],
            "VELOCITY_UMAP1": -points[:, 1],
            "VELOCITY_UMAP2": points[:, 0],
        }
    )
    plot = ggplot(frame) + geom_point(aes(x="X_UMAP1", y="X_UMAP2"))

    combined = plot + cl.stream(arrow_kwargs={"alpha": 0.5}, density=0.5)

    assert isinstance(combined, PlotSpec)
    assert "<path" in combined.to_svg()


def _adata_with_velocity(adata):
    """Copy the fixture and attach a velocity embedding paired with `X_umap`."""
    data = adata.copy()
    rng = np.random.default_rng(0)
    data.obsm["velocity_umap"] = rng.standard_normal(data.obsm["X_umap"].shape)
    return data


def test_umap_frame_includes_unplotted_velocity_embedding(adata):
    # Regression: `cl.umap` plots `X_umap` but must still materialise every
    # embedding into the frame, because the deferred `cl.stream()` layer reads a
    # different embedding (`velocity_umap` -> VELOCITY_UMAP1/2) from it. Narrowing
    # the built frame to only the plotted embedding broke `+ cl.stream()`.
    data = _adata_with_velocity(adata)
    frame = retrieve(cl.umap(data, tooltips="none"))
    assert {"VELOCITY_UMAP1", "VELOCITY_UMAP2"} <= set(frame.columns)


def test_stream_builds_on_umap_with_velocity_embedding(adata):
    # End-to-end mirror of the reported failure: `cl.umap(...) + cl.stream()`
    # must resolve velocity columns from the umap frame without raising.
    data = _adata_with_velocity(adata)
    umap = cl.umap(data, tooltips="none")

    combined = umap + cl.stream()
    assert isinstance(combined, PlotSpec)
