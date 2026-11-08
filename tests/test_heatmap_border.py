"""Rendered data-area border of dotplot and stacked_violin."""

import re

import pytest

import cellestial as cl

GROUPS = {"T cells": ["CD3D", "CD8A"], "Myeloid": ["CST3", "LYZ"]}
PLOTS = {"dotplot": cl.dotplot, "stacked_violin": cl.stacked_violin}
_RECT = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"')


def _panel(svg: str) -> tuple[float, ...]:
    """The panel's clip rectangle: the first sized rectangle after the background."""
    return tuple(float(value) for value in _RECT.findall(svg)[0])


def _border(svg: str, color: str) -> tuple[tuple[float, ...], float]:
    """The rectangle stroked with `color` and its stroke width."""
    index = svg.index(f'stroke="{color}"')
    width = float(re.search(r'stroke-width="([\d.]+)"', svg[index:]).group(1))
    rect = _RECT.search(svg, index - 200 if "<rect" in svg[index - 200 : index] else index)
    return tuple(float(value) for value in rect.groups()), width


@pytest.mark.parametrize("name", PLOTS)
@pytest.mark.parametrize("dendrogram", [False, True])
def test_border_boxes_the_data_area_only(adata, name, dendrogram):
    svg = PLOTS[name](
        adata,
        keys=GROUPS,
        group_by="leiden",
        dendrogram=dendrogram,
        rectangle_color="#ab1234",
        rectangle_size=2,
    ).to_svg()
    rect, width = _border(svg, "#ab1234")
    assert rect == pytest.approx(_panel(svg), abs=0.5)
    assert width == pytest.approx(4.4)


@pytest.mark.parametrize("name", PLOTS)
def test_rectangle_false_draws_no_border(adata, group_key, name):
    svg = PLOTS[name](adata, keys=GROUPS, group_by=group_key, rectangle=False).to_svg()
    assert 'stroke="#3f3f3f"' not in svg


def test_annotated_heatmap_panel_border_boxes_the_heatmap_only(adata):
    from ggrust import element_rect, theme

    plot = cl.annotated_heatmap(
        adata,
        keys=["CD3D", "LYZ", "MS4A1"],
        group_by="leiden",
        row_annotations="cell_type_lvl1",
        column_annotations=None,
        max_rows=50,
    ) + theme(panel_border=element_rect(color="#ab1234", size=2))
    svg = plot.to_svg()
    assert svg.count('stroke="#ab1234"') == 1
    rect, _ = _border(svg, "#ab1234")
    widest = max((float(width) for _, _, width, _ in _RECT.findall(svg)), default=0)
    assert rect[2] == pytest.approx(widest, abs=0.5)
