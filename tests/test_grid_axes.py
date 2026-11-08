"""Rendered shared axes and titles of plural plots on a 3-column grid of 5."""

import re

import pytest

import cellestial as cl

KEYS = ["CD3D", "LYZ", "MS4A1", "NKG7", "CST3"]


def _labelled_axes(svg: str) -> tuple[int, int]:
    """Number of x and y axes that draw tick labels."""
    x_axes = re.findall(r'<g class="axis-x">(.*?)</g>', svg, re.S)
    y_axes = re.findall(r'<g class="axis-y">(.*?)</g>', svg, re.S)
    return sum("<text" in axis for axis in x_axes), sum("<text" in axis for axis in y_axes)


def test_umaps_share_axis_keeps_bottom_and_left_axes(adata):
    svg = cl.umaps(adata, keys=KEYS, ncol=3, axis_type="axis", share_axis=True).to_svg()
    assert _labelled_axes(svg) == (3, 2)


def test_umaps_without_share_axis_label_every_axis(adata):
    svg = cl.umaps(adata, keys=KEYS, ncol=3, axis_type="axis").to_svg()
    assert _labelled_axes(svg) == (5, 5)


def test_umaps_share_labels_keeps_one_title_per_run(adata):
    svg = cl.umaps(adata, keys=KEYS, ncol=3, share_labels=True).to_svg()
    # One x title centred under the two full columns, one under the third; one y per row.
    assert (svg.count(">UMAP1<"), svg.count(">UMAP2<")) == (2, 2)


@pytest.mark.parametrize("option", ["share_axis", "share_ticks"])
def test_violins_share_the_group_axis(adata, group_key, option):
    svg = cl.violins(adata, keys=KEYS, group_by=group_key, ncol=3, **{option: True}).to_svg()
    assert _labelled_axes(svg)[0] == 3


@pytest.mark.parametrize("option", ["share_axis", "share_ticks"])
def test_violins_share_the_value_axis_when_shared(adata, group_key, option):
    svg = cl.violins(
        adata, keys=KEYS, group_by=group_key, ncol=3, sharey=True, **{option: True}
    ).to_svg()
    assert _labelled_axes(svg)[1] == 2
