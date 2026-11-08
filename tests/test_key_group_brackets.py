"""Rendered key-group brackets: names above the plot, centred on their keys."""

import re

import pytest

import cellestial as cl

GROUPS = {"T cells": ["CD3D", "CD8A"], "B": ["MS4A1"], "Myeloid": ["CST3", "LYZ"]}
PLOTS = {
    "dotplot": lambda data, group_key, **kwargs: cl.dotplot(data, group_by=group_key, **kwargs),
    "stacked_violin": lambda data, group_key, **kwargs: cl.stacked_violin(
        data, group_by=group_key, **kwargs
    ),
    "matrixplot": lambda data, group_key, **kwargs: cl.matrixplot(
        data, group_by=group_key, **kwargs
    ),
    "heatmap_cells": lambda data, group_key, **kwargs: cl.heatmap(
        data, group_by=group_key, aggregate=False, **kwargs
    ),
}
_TEXT = re.compile(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*>([^<]*)</text>')


def _texts(svg: str) -> dict[str, tuple[float, float]]:
    return {label: (float(x), float(y)) for x, y, label in _TEXT.findall(svg)}


@pytest.mark.parametrize("name", PLOTS)
def test_key_group_names_sit_above_the_plot_centred_on_their_keys(adata, group_key, name):
    texts = _texts(PLOTS[name](adata, group_key, keys=GROUPS).to_svg())
    keys = [key for values in GROUPS.values() for key in values]
    lowest_name = max(texts[group][1] for group in GROUPS)
    assert lowest_name < min(texts[key][1] for key in keys)
    for group, members in GROUPS.items():
        first, last = texts[members[0]][0], texts[members[-1]][0]
        assert first - 1 <= texts[group][0] <= last + 1
    assert [texts[group][0] for group in GROUPS] == sorted(texts[group][0] for group in GROUPS)


@pytest.mark.parametrize("name", PLOTS)
def test_key_labels_false_draws_no_group_names(adata, group_key, name):
    texts = _texts(PLOTS[name](adata, group_key, keys=GROUPS, key_labels=False).to_svg())
    assert not set(GROUPS) & set(texts)
    assert "CD3D" in texts


@pytest.mark.parametrize("name", PLOTS)
def test_bracket_color_and_size_reach_the_brackets(adata, group_key, name):
    svg = PLOTS[name](
        adata,
        group_key,
        keys=GROUPS,
        key_labels_bracket_color="#ab1234",
        key_labels_bracket_size=3,
    ).to_svg()
    assert re.search(r'stroke="#ab1234" stroke-width="6.6"', svg)
