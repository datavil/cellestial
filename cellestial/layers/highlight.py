from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from typing import TYPE_CHECKING

import polars as pl
from ggrust import gghighlight
from ggrust.plot.core import PlotSpec

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import get_mapping, retrieve
from cellestial.util.errors import (
    AmbiguousColumnError,
    EmptySelectionError,
    InvalidPlotSpecError,
    MissingAestheticError,
    NonCategoricalColumnError,
)
from cellestial.util.utilities import _require_categories

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ggrust.plot.core import FeatureSpec


def highlight(
    groups: Sequence[str] | str,
    *,
    unhighlighted_params: Mapping[str, object] | None = None,
    calculate_per_facet: bool = False,
) -> DeferredLayer:
    """
    Layer that draws the chosen groups in color over every other cell in grey.

    Parameters
    ----------
    groups : Sequence[str] | str
        The groups to highlight, among the categories the plot is colored (or
        filled) by. e.g. `["0", "3"]` for two clusters of a plot colored by `'leiden'`.
    unhighlighted_params : Mapping[str, object] | None, default=None
        Layer parameters of the grey copies, e.g. `{"color": "pink", "size": 0.3}`.
        None draws them in a translucent grey.
    calculate_per_facet : bool, default=False
        Whether each facet panel greys out only its own rows. Add the facet before
        `highlight`. False draws every row in grey in every panel, and panels without
        a highlighted group are dropped.

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
    AmbiguousColumnError
        If the plot is colored and filled by two different keys.
    MissingAestheticError
        If the plot is neither colored nor filled by a key.
    NonCategoricalColumnError
        If the plot is colored (or filled) by numeric values (e.g. gene expression).
    KeyNotFoundError
        If a group is not among the categories the plot is colored (or filled) by.

    Notes
    -----
    Add it last. It redraws the layers before it, so cellestial layers that read
    the plot's color (e.g. `ondata_legend`) must come before it. Layers without
    the color column (e.g. `arrow_axis`) are drawn as they are. Only the
    highlighted groups keep their colors and appear in the legend.

    The groups are read from the color mapping of the plot, or from its fill
    mapping when it has no color mapping (e.g. `violin` or `boxplot` with `fill`).
    A plot colored and filled by two different keys is refused.

    Examples
    --------
    Highlight two clusters.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.bone_marrow()

        cl.umap(data, key="leiden") + cl.highlight(["0", "3"])

    Highlight a single cell type.

    .. jupyter-execute::

        cl.umap(data, key="cell_type_lvl1") + cl.highlight("B Cells")

    Highlight one cluster of a violin plot.

    .. jupyter-execute::

        cl.violin(data, "CST3", group_by="leiden", fill="leiden") + cl.highlight("0")

    Draw the other cells in a lighter, smaller grey.

    .. jupyter-execute::

        cl.umap(data, key="leiden") + cl.highlight(
            "0", unhighlighted_params={"color": "#e5e5e5", "size": 0.3}
        )

    Highlight clusters within each cell type panel.

    .. jupyter-execute::

        from ggrust import facet_wrap

        (
            cl.umap(data, key="leiden", add_keys=["cell_type_lvl1"])
            + facet_wrap("cell_type_lvl1")
            + cl.highlight(["0", "1"], calculate_per_facet=True)
        )
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
        mapping = get_mapping(receiving_plot, index=0)
        color, fill = mapping.get("color"), mapping.get("fill")
        if color is not None and fill is not None and color != fill:
            msg = (
                f"`highlight` reads the groups from the color or fill of the plot, but the plot "
                f"is colored by `{color}` and filled by `{fill}`. Map color and fill to the "
                f"same key, or only one of them, e.g. `fill='{fill}'` without `color`."
            )
            raise AmbiguousColumnError(msg)
        aesthetic = "color" if mapping.get("color") is not None else "fill"
        column = mapping.get(aesthetic)
        if column is None:
            msg = (
                "`highlight` needs a plot colored or filled by a key, but the plot has no "
                "color or fill mapping. Pass a categorical key to the plot, e.g. "
                "`cl.umap(data, key='leiden')` or `cl.violin(data, 'CD14', group_by='leiden', "
                "fill='leiden')`."
            )
            raise MissingAestheticError(msg)
        frame = retrieve(receiving_plot)
        if frame[column].dtype not in (pl.Categorical, pl.Enum, pl.String):
            msg = (
                f"`highlight` needs a plot {aesthetic}ed by categories, but `{column}` holds "
                f"numeric values ({frame[column].dtype}). Use a categorical key for the "
                f"{aesthetic}, e.g. `cl.umap(data, key='leiden')`."
            )
            raise NonCategoricalColumnError(msg)
        available = frame[column].unique().drop_nulls().cast(pl.String).to_list()
        _require_categories(
            groups, available, parameter="groups", column=column, column_parameter=aesthetic
        )
        return gghighlight(
            pl.col(column).is_in(groups),
            use_group_by=False,
            use_direct_label=False,
            unhighlighted_params=unhighlighted_params,
            calculate_per_facet=calculate_per_facet,
        )

    return DeferredLayer(_build)
