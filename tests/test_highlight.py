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
