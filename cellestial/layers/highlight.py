from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from typing import TYPE_CHECKING

import polars as pl
from ggrust import gghighlight
from ggrust.plot.core import PlotSpec

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import get_mapping, retrieve
from cellestial.util.errors import (
    EmptySelectionError,
    InvalidPlotSpecError,
    MissingAestheticError,
    NonCategoricalColumnError,
)
from cellestial.util.utilities import _require_categories

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ggrust.plot.core import FeatureSpec


def highlight(groups: Sequence[str] | str) -> DeferredLayer:
    """
    Layer that draws the chosen groups in color over every other cell in grey.

    Parameters
    ----------
    groups : Sequence[str] | str
        The groups to highlight, among the categories the plot is colored by.
        e.g. `["0", "3"]` for two clusters of a plot colored by `'leiden'`.

    Returns
    -------
    DeferredLayer
        Highlight layer, to add to a plot with `+`.

    Raises
    ------
    EmptySelectionError
        If `groups` is empty.
    InvalidPlotSpecError
        If it is added to a grid of plots (e.g. from `umaps`) rather than a single plot.
    MissingAestheticError
        If the plot is not colored by a key.
    NonCategoricalColumnError
        If the plot is colored by numeric values (e.g. gene expression).
    KeyNotFoundError
        If a group is not among the categories the plot is colored by.

    Notes
    -----
    Add it last. It redraws the layers before it, so cellestial layers that read
    the plot's color (e.g. `ondata_legend`) must come before it. Layers without
    the color column (e.g. `arrow_axis`) are drawn as they are. Only the
    highlighted groups keep their colors and appear in the legend.

    Examples
    --------
    Highlight two clusters.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.umap(data, key="leiden") + cl.highlight(["0", "3"])

    Highlight a single cell type.

    .. jupyter-execute::

        cl.umap(data, key="cell_type_lvl1") + cl.highlight("B Cells")
    """
    groups = [groups] if isinstance(groups, str) else list(groups)
    if not groups:
        msg = "`groups` is empty, so nothing would be highlighted. Pass at least one group, e.g. `['0']`."
        raise EmptySelectionError(msg)

    def _build(receiving_plot: PlotSpec) -> FeatureSpec:
        if not isinstance(receiving_plot, PlotSpec):
            msg = (
                f"`highlight` adds to a single plot, but received `{type(receiving_plot).__name__}`. "
                "Add it to a singular plot, e.g. `cl.umap(data, key='leiden') + cl.highlight('0')`."
            )
            raise InvalidPlotSpecError(msg)
        color = get_mapping(receiving_plot, index=0).get("color")
        if color is None:
            msg = (
                "`highlight` needs a plot colored by a key, but the plot has no color mapping. "
                "Pass a categorical key to the plot, e.g. `cl.umap(data, key='leiden')`."
            )
            raise MissingAestheticError(msg)
        frame = retrieve(receiving_plot)
        if frame[color].dtype not in (pl.Categorical, pl.Enum, pl.String):
            msg = (
                f"`highlight` needs a plot colored by categories, but `{color}` holds "
                f"numeric values ({frame[color].dtype}). Color the plot by a categorical "
                "key, e.g. `cl.umap(data, key='leiden')`."
            )
            raise NonCategoricalColumnError(msg)
        available = frame[color].unique().drop_nulls().cast(pl.String).to_list()
        _require_categories(
            groups, available, parameter="groups", column=color, column_parameter="color"
        )
        return gghighlight(pl.col(color).is_in(groups), use_group_by=False, use_direct_label=False)

    return DeferredLayer(_build)
