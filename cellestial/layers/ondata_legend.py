from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from lets_plot import aes, geom_text, theme

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import get_mapping, retrieve

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpec, FeatureSpecArray, PlotSpec
    from polars import DataFrame


def _compute_label_positions(
    frame: DataFrame,
    *,
    x: str,
    y: str,
    group_by: str,
    weighted: bool,
) -> DataFrame:
    """Aggregate per-group label coordinates for `geom_text` placement."""
    if weighted:
        group_means = frame.group_by(group_by).agg(
            pl.col(x).mean().alias("mean_x"),
            pl.col(y).mean().alias("mean_y"),
        )
        frame = frame.join(group_means, on=group_by, how="left")
        frame = frame.with_columns(
            ((pl.col(x) - pl.col("mean_x")) ** 2 + (pl.col(y) - pl.col("mean_y")) ** 2)
            .sqrt()
            .alias("distance")
        )
        frame = frame.with_columns((1 / pl.col("distance").sqrt()).alias("weight"))
        return frame.group_by(group_by).agg(
            (pl.col(x) * pl.col("weight")).sum() / pl.col("weight").sum(),
            (pl.col(y) * pl.col("weight")).sum() / pl.col("weight").sum(),
            pl.selectors.categorical().mode().first(),
        )
    return frame.group_by(group_by).agg(
        pl.col(x).mean(), pl.col(y).mean(), pl.selectors.categorical().mode().first()
    )

# AI-MODIFIED: Claude 4.7
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line
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
    weighted: bool = True,
    **geom_kwargs,
) -> DeferredLayer:
    """
    Layer of `geom_text` that places per-group labels at the center of each cluster.

    Parameters
    ----------
    plot : PlotSpec | None, default=None
        If provided, labels are computed from this plot's data and aesthetics
        regardless of which plot the resulting layer is added to. When ``None``,
        the layer is deferred and introspects the plot it is added to via ``+``.
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
    weighted : bool, default=True
        Whether to use a weighted mean of group coordinates for label placement.
        If True, each point's contribution is weighted by its inverse distance to
        the group mean, pulling the label toward the cluster's dense core.
        If False, the arithmetic mean of group coordinates is used.
    **geom_kwargs
        Additional parameters for the `geom_text` layer.
        For more information on geom_text parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_text.html


    Returns
    -------
    DeferredLayer
        On-data legend layer.

    Examples
    --------
    A UMAP plot without on-data legends.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

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
            raise ValueError(msg)
        if y is None:
            msg = "`y` is present neither as argument nor in the plot aesthetics."
            raise ValueError(msg)
        if group_by is None:
            msg = "`group_by` is present neither as argument nor in the plot aesthetics."
            raise ValueError(msg)

        frame = retrieve(source)
        grouped = _compute_label_positions(
            frame, x=x, y=y, group_by=group_by, weighted=weighted
        )
        return geom_text(
            data=grouped,
            mapping=aes(x=x, y=y, label=group_by),
            size=size,
            color=color,
            fontface=fontface,
            family=family,
            alpha=alpha,
            **geom_kwargs,
        ) + theme(legend_position="none")

    return DeferredLayer(_build)
