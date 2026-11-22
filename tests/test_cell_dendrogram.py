"""Rendered group tree of the per-cell heatmap and annotated_heatmap."""

import re
from itertools import pairwise

import pytest

import cellestial as cl
from cellestial.util.dendrogram import _get_dendrogram_tree

KEYS = ["CD3D", "LYZ", "MS4A1", "NKG7"]
_TEXT = re.compile(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*>([^<]*)</text>')
_PATH = re.compile(r'<path d="([^"]*)"[^>]*stroke="#ab1234"')


def _labels(svg: str, names: set[str]) -> list[tuple[float, float, str]]:
    return [(float(x), float(y), text) for x, y, text in _TEXT.findall(svg) if text in names]


def _path_xs(svg: str) -> list[float]:
    """X coordinates of the visible (non-zero length) paths stroked `#ab1234`."""
    paths = _PATH.findall(svg)
    for group in re.findall(r'<g[^>]*stroke="#ab1234"[^>]*>(.*?)</g>', svg, re.S):
        paths.extend(re.findall(r'd="([^"]*)"', group))
    xs: list[float] = []
    for path in paths:
        points = re.findall(r"[ML]([-\d.]+) ([-\d.]+)", path)
        if len(set(points)) > 1:
            xs.extend(float(x) for x, _ in points)
    return xs


def test_cell_heatmap_tree_orders_the_groups_and_sits_right_of_the_cells(adata):
    tree = _get_dendrogram_tree(adata, "leiden")
    svg = cl.heatmap(
        adata,
        keys=KEYS,
        group_by="leiden",
        aggregate=False,
        dendrogram=True,
        dendrogram_color="#ab1234",
    ).to_svg()
    key_xs = [x for x, _, _ in _labels(svg, set(KEYS))]
    # Group names left of the cells (the color bar also has numeric labels).
    groups = [label for label in _labels(svg, set(tree.ordered_labels)) if label[0] < min(key_xs)]
    assert len(groups) == len(tree.ordered_labels)
    bottom_to_top = [text for _, _, text in sorted(groups, key=lambda label: -label[1])]
    assert bottom_to_top == list(tree.ordered_labels)
    tree_xs = _path_xs(svg)
    assert tree_xs
    assert min(tree_xs) > max(key_xs)


def _segments(svg: str, color: str) -> list[list[tuple[float, float]]]:
    """Point lists of the visible paths stroked `color`."""
    paths = re.findall(rf'<path d="([^"]*)"[^>]*stroke="{color}"', svg)
    for group in re.findall(rf'<g[^>]*stroke="{color}"[^>]*>(.*?)</g>', svg, re.S):
        paths.extend(re.findall(r'd="([^"]*)"', group))
    segments = []
    for path in paths:
        points = [(float(x), float(y)) for x, y in re.findall(r"[ML]([-\d.]+) ([-\d.]+)", path)]
        if len(set(points)) > 1:
            segments.append(points)
    return segments


@pytest.mark.parametrize("transpose", [False, True])
def test_annotated_heatmap_tree_leaves_sit_at_the_group_centers(adata, transpose):
    tree = _get_dendrogram_tree(adata, "leiden")
    svg = cl.annotated_heatmap(
        adata,
        keys=KEYS,
        group_by="leiden",
        dendrogram=True,
        dendrogram_color="#ab1234",
        group_lines_color="#cd5678",
        transpose=transpose,
        max_rows=200,
    ).to_svg()
    # Along: the observation axis (y upright, x transposed); cross: the other one.
    along, cross = (0, 1) if transpose else (1, 0)
    points = [point for segment in _segments(svg, "#ab1234") for point in segment]
    assert points
    # Leaves sit at the cross position nearest the heatmap (below / right of the tree).
    leaf_cross = max(point[cross] for point in points)
    leaves = sorted({round(point[along], 1) for point in points if point[cross] == leaf_cross})
    assert len(leaves) == len(tree.ordered_labels)
    separators = sorted({segment[0][along] for segment in _segments(svg, "#cd5678")})
    assert len(separators) == len(leaves) - 1
    # Every inner group's leaf lies midway between its two separators.
    for (low, high), leaf in zip(pairwise(separators), leaves[1:-1], strict=True):
        assert leaf == pytest.approx((low + high) / 2, abs=1)
    # The tree lies outside the heatmap, before its first separator's start.
    assert leaf_cross <= min(segment[0][cross] for segment in _segments(svg, "#cd5678"))
