"""Rendered key labels of annotated_heatmap: chosen keys beside the heatmap, in order."""

import re

import cellestial as cl

_TEXT = re.compile(r'<text x="([-\d.]+)" y="([-\d.]+)"[^>]*>([^<]*)</text>')
_RECT = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([-\d.]+)" height="([-\d.]+)"')


def test_key_labels_sit_left_of_the_heatmap_in_key_order(adata, markers, group_key):
    keys = markers[:6]
    svg = cl.annotated_heatmap(
        adata,
        keys=keys,
        group_by=group_key,
        column_annotations=["highly_variable"],
        column_annotation_position="bottom",
        transpose=True,
        key_labels=[keys[4], keys[0], keys[2]],
        max_rows=20,
    ).to_svg()
    panel_x, panel_y, _, panel_height = (float(value) for value in _RECT.findall(svg)[0])
    labels = {text: (float(x), float(y)) for x, y, text in _TEXT.findall(svg) if text in keys}
    assert set(labels) == {keys[4], keys[0], keys[2]}
    assert all(x < panel_x for x, _ in labels.values())
    assert all(panel_y <= y <= panel_y + panel_height + 10 for _, y in labels.values())
    # Transposed, the first key is at the top.
    assert labels[keys[0]][1] < labels[keys[2]][1] < labels[keys[4]][1]
    # Leader lines run between the labels and the heatmap's left edge.
    heatmap_left = float(re.search(r'<image[^>]* x="([-\d.]+)"', svg).group(1))
    label_right = max(x for x, _ in labels.values())
    leader_paths = [
        [float(x) for x in re.findall(r"[ML]([-\d.]+) ", path)]
        for path in re.findall(r'<path d="([^"]*)"', svg)
    ]
    leaders = [
        xs
        for xs in leader_paths
        if xs and label_right <= min(xs) and max(xs) <= heatmap_left + 0.5
    ]
    assert len(leaders) >= 3
