from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from lets_plot import aes, geom_label, geom_label_repel, geom_text, geom_text_repel, theme

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import _drop_nonfinite_rows, get_mapping, retrieve
from cellestial.util.errors import MissingAestheticError

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpec, FeatureSpecArray, PlotSpec
    from polars import DataFrame


def _compute_label_positions(
    frame: DataFrame,
    *,
    x: str,
    y: str,
    group_by: str,
) -> DataFrame:
    """Aggregate per-group median coordinates for `geom_text` placement."""
    frame = _drop_nonfinite_rows(frame, [x, y]).filter(pl.col(group_by).is_not_null())
    return frame.group_by(group_by).agg(
        pl.col(x).median(),
        pl.col(y).median(),
        pl.selectors.categorical().mode().first(),
    )


def ondata_legend(
    *,
    plot: PlotSpec | None = None,
    x: str | None = None,
    y: str | None = None,
    group_by: str | None = None,
    size: float = 12,
    color: str = "#3f3f3f",
    fontface: str = "bold",
    family: str = "sans",
    alpha: float = 1,
    halo_width: float | None = 0.5,
    halo_color: str | None = None,
    label: bool = False,
    repel: bool = False,
    **geom_kwargs,
) -> DeferredLayer:
    """
    Layer of `geom_text` that places per-group labels at median coordinates.

    Parameters
    ----------
    plot : PlotSpec | None, default=None
        If provided, labels are computed from this plot's data and aesthetics
        regardless of which plot the resulting layer is added to. When `None`,
        the layer is deferred and introspects the plot it is added to via `+`.
    x : str | None, default=None
        The column name in the data used for x-axis coordinates. e.g 'X_UMAP1'.
        If None, it will be inferred from the plot aesthetics.
    y : str | None, default=None
        The column name in the data used for y-axis coordinates. e.g 'X_UMAP2'.
        If None, it will be inferred from the plot aesthetics.
    group_by : str | None, default=None
        The column name in the data used to group clusters by. e.g 'cell_type'.
        If None, it will be inferred from the plot's `color` aesthetic.
    size : float, default=12
        Size of the legend text.
    color : str, default='#3f3f3f'
        Color of the legend text.
    fontface : str, default='bold'
        Fontface of the legend text.
        https://lets-plot.org/python/pages/aesthetics.html#font-face
    family : str, default='sans'
        Font family of the legend text.
        https://lets-plot.org/python/pages/aesthetics.html#font-family
    alpha : float, default=1
        Alpha (transparency) of the legend text.
    halo_width : float | None, default = 0.5
        Width of the text halo (text outline), not rendered when 0.
    halo_color : str | None, default = None
        Color of the text halo (text outline).
    label : bool, default=False
        If True, draw labels with a filled background using `geom_label`.
    repel : bool, default=False
        If True, use `geom_text_repel` so labels are shifted to avoid overlapping
        each other. Repel-specific options (e.g. `box_padding`, `point_padding`,
        `max_iter`, `seed`) can be passed via `geom_kwargs`.
    **geom_kwargs
        Additional parameters for the underlying geom layer.
        For `geom_text` parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_text.html
        For `geom_label` parameters (when `label=True`), see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_label.html
        For `geom_text_repel` parameters (when `repel=True`), see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_text_repel.html
        For `geom_label_repel` parameters (when `label=True` and `repel=True`), see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_label_repel.html


    Returns
    -------
    DeferredLayer
        On-data legend layer.

    Raises
    ------
    MissingAestheticError
        If `x`, `y`, or `group_by` cannot be provided explicitly or inferred
        from the receiving plot.

    Examples
    --------
    A UMAP plot without on-data legends.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = cl.datasets.pbmc3k(cache_directory="data")

        umap = cl.umap(data, key="cell_type_lvl1", axis_type="arrow", size=1.5)
        umap

    Add the on-data legend.

    .. jupyter-execute::

        umap + cl.ondata_legend()

    Modify as needed.

    .. jupyter-execute::

        umap + cl.ondata_legend(size=10, family="mono", fontface="italic")
    """
    explicit_plot = plot
    explicit_x = x
    explicit_y = y
    explicit_group_by = group_by

    def _build(receiving_plot: PlotSpec) -> FeatureSpec | FeatureSpecArray:
        source = explicit_plot if explicit_plot is not None else receiving_plot
        _mapping = get_mapping(source, index=0)
        x = _mapping.get("x") if explicit_x is None else explicit_x
        y = _mapping.get("y") if explicit_y is None else explicit_y
        group_by = _mapping.get("color") if explicit_group_by is None else explicit_group_by
        if x is None:
            msg = "`x` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)
        if y is None:
            msg = "`y` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)
        if group_by is None:
            msg = "`group_by` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)

        frame = retrieve(source)
        grouped = _compute_label_positions(frame, x=x, y=y, group_by=group_by)
        layer_kwargs = dict(geom_kwargs)
        if label:
            geom = geom_label_repel if repel else geom_label
            layer_kwargs = {
                "fill": "white",
                "label_size": 0,
                **layer_kwargs,
            }
        else:
            geom = geom_text_repel if repel else geom_text

        if label:
            layer = geom(
                data=grouped,
                mapping=aes(x=x, y=y, label=group_by),
                size=size,
                color=color,
                fontface=fontface,
                family=family,
                alpha=alpha,
                **layer_kwargs,
            ) + theme(legend_position="none")
        else:
            layer = geom(
                data=grouped,
                mapping=aes(x=x, y=y, label=group_by),
                size=size,
                color=color,
                fontface=fontface,
                family=family,
                halo_width=halo_width,
                halo_color=halo_color,
                alpha=alpha,
                **layer_kwargs,
            ) + theme(legend_position="none")
        return layer

    return DeferredLayer(_build)
