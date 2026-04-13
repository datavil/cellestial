from cellestial.util.dendrogram import _get_dendrogram, _get_dendrogram_segment_frame
from cellestial.util.legends import _legend_ondata
from cellestial.util.operations import get_mapping, get_slice, retrieve
from cellestial.util.utilities import (
    _color_gradient,
    _determine_axis,
    _fill_gradient,
    _is_observation_key,
    _is_variable_key,
    _range_inclusive,
    _select_variable_keys,
    _share_axis,
    _share_labels,
    _share_ticks,
)

__all__ = [
    "_color_gradient",
    "_determine_axis",
    "_fill_gradient",
    "_get_dendrogram",
    "_get_dendrogram_segment_frame",
    "_is_observation_key",
    "_is_variable_key",
    "_legend_ondata",
    "_range_inclusive",
    "_select_variable_keys",
    "_share_axis",
    "_share_labels",
    "_share_ticks",
    "get_mapping",
    "get_slice",
    "retrieve",
]
