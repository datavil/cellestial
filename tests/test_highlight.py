"""Highlighting groups on dimensional plots with ggrust's `gghighlight` (issue #38)."""

import re
import warnings

import polars as pl
import pytest
from ggrust import gghighlight
from ggrust.plot.core import PlotSpec

import cellestial as cl


def _svg_texts(svg: str) -> list[str]:
    return [text.strip() for text in re.findall(r"<text[^>]*>([^<]*)</text>", svg)]


@pytest.mark.parametrize("function", [cl.dimensional, cl.umap, cl.tsne, cl.pca])
def test_highlight_groups_greys_out_the_rest(adata, cluster_key, function):
    """Highlighted groups are drawn on top of every cell in grey."""
    highlighted_groups = ["0", "3"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot = function(adata, cluster_key, tooltips="none") + gghighlight(
            pl.col(cluster_key).is_in(highlighted_groups).all()
        )
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
    texts = set(_svg_texts(svg))
    assert set(highlighted_groups) <= texts
    assert not other_groups & texts


def test_highlight_expression_threshold(adata):
    """Cells above an expression threshold are drawn on top of every cell in grey."""
    threshold = 2.0
    plot = cl.expression(adata, "CD14", tooltips="none") + gghighlight(pl.col("CD14") > threshold)
    svg = plot.to_svg()
    frame = cl.build_frame(adata, variable_keys=["CD14"], axis=0)
    n_highlighted = frame.filter(pl.col("CD14") > threshold).height
    assert 0 < n_highlighted < adata.n_obs
    assert svg.count("<circle") == adata.n_obs + n_highlighted
