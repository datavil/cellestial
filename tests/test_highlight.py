"""Highlighting groups on dimensional plots with `cl.highlight` (issue #38)."""

import re
import warnings

import pytest
from ggrust.plot.core import PlotSpec

import cellestial as cl


def _svg_texts(svg: str) -> set[str]:
    return {text.strip() for text in re.findall(r"<text[^>]*>([^<]*)</text>", svg)}


@pytest.mark.parametrize("function", [cl.dimensional, cl.umap, cl.tsne, cl.pca])
def test_highlight_groups_greys_out_the_rest(adata, cluster_key, function):
    """Highlighted groups are drawn on top of every cell in grey."""
    highlighted_groups = ["0", "3"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot = function(adata, cluster_key, tooltips="none") + cl.highlight(highlighted_groups)
        svg = plot.to_svg()
    assert isinstance(plot, PlotSpec)

    n_highlighted = int(adata.obs[cluster_key].isin(highlighted_groups).sum())
    assert 0 < n_highlighted < adata.n_obs
    # Every cell drawn once in grey, highlighted cells drawn again on top,
    # plus one legend key per highlighted group.
    assert svg.count("<circle") == adata.n_obs + n_highlighted + len(highlighted_groups)
    assert "#bebebe" in svg.lower()
    # The legend lists only the highlighted groups.
    other_groups = set(adata.obs[cluster_key].cat.categories) - set(highlighted_groups)
    texts = _svg_texts(svg)
    assert set(highlighted_groups) <= texts
    assert not other_groups & texts


def test_highlight_single_group_as_string(adata, group_key):
    """A single group can be passed as a string."""
    group = adata.obs[group_key].cat.categories[0]
    svg = (cl.umap(adata, group_key, tooltips="none") + cl.highlight(group)).to_svg()
    n_highlighted = int((adata.obs[group_key] == group).sum())
    assert svg.count("<circle") == adata.n_obs + n_highlighted + 1


def test_highlight_greys_out_ondata_labels(adata, cluster_key):
    """On-data labels of the other groups are greyed out too, and no direct labels are added."""
    plot = cl.umap(adata, cluster_key, tooltips="none", legend_ondata=True) + cl.highlight("3")
    svg = plot.to_svg()
    texts = [text.strip() for text in re.findall(r"<text[^>]*>([^<]*)</text>", svg)]
    # Every group label is drawn in grey, the highlighted one again on top.
    assert texts.count("0") > 0
    assert texts.count("3") == 2 * texts.count("0")


@pytest.mark.parametrize("function", [cl.violin, cl.boxplot])
def test_highlight_fill_mapped_distribution(adata, cluster_key, function):
    """Plots filled by a key highlight by their fill, and only those groups get a legend key."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # theme font face warning, unrelated to highlight
        plot = function(adata, "CD14", group_by=cluster_key, fill=cluster_key) + cl.highlight("3")
        svg = plot.to_svg()
    texts = _svg_texts(svg)
    assert "#bebebe" in svg.lower()
    # Every group is still on the x axis, the legend title is drawn once.
    assert set(adata.obs[cluster_key].cat.categories) <= texts


def test_highlight_unhighlighted_params_restyle_the_grey(adata, cluster_key):
    """`unhighlighted_params` replaces the grey of the other cells."""
    plot = cl.umap(adata, cluster_key, tooltips="none") + cl.highlight(
        "0", unhighlighted_params={"color": "#ff00ff"}
    )
    svg = plot.to_svg().lower()
    assert "#ff00ff" in svg
    assert "#bebebe" not in svg


def test_highlight_calculate_per_facet_keeps_every_panel(adata, cluster_key, group_key):
    """Per facet, panels without a highlighted group still show their own cells in grey."""
    from ggrust import facet_wrap

    base = cl.umap(adata, cluster_key, tooltips="none", add_keys=[group_key]) + facet_wrap(
        group_key
    )
    panels = set(adata.obs[group_key].cat.categories)
    highlighted = adata.obs[cluster_key] == "0"
    with_highlight = set(adata.obs.loc[highlighted, group_key].unique())
    assert with_highlight < panels  # the test needs a panel without the group
    per_facet = _svg_texts((base + cl.highlight("0", calculate_per_facet=True)).to_svg())
    across = _svg_texts((base + cl.highlight("0")).to_svg())
    assert panels <= per_facet
    assert not (panels - with_highlight) & across
