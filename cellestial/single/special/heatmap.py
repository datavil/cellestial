from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import polars as pl
from lets_plot import aes, geom_tile, ggplot, ggtb, scale_fill_gradient

from cellestial.frames import build_frame
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import _fill_gradient

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.core import FeatureSpec, PlotSpec


def heatmap(
    data: AnnData,
    group_by: str,
    keys: Sequence[str] | None = None,
    *,
    mapping: FeatureSpec | None = None,
    value_column: str = "value",
    variable_column: str = "variable",
    color_low: str = "#0000ff",
    color_mid: str = "#ffffff",
    color_high: str = "#ff0000",
    mid_point: Literal["mean", "median", "mid"] | float = "mid",
    axis: Literal[0, 1] | None = 0,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool | int = False,
    interactive: bool = False,
    **geom_kwargs,
) -> PlotSpec:
    """
    Heatmap.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=0
        axis of the data, 0 for observations and 1 for variables.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    observations_name : str, default='Barcode'
        The name of the observations column.
    variables_name : str, default='Variable'
        Name for the variables index column.
    include_dimensions : bool | int, default=False
        Whether to include dimensions in the DataFrame.
        Providing an integer will limit the number of dimensions to given number.
    interactive : bool, default=False
        Whether to make the plot interactive.
    **geom_kwargs
        Additional parameters for the `geom_bar` layer.
        For more information on geom_bar parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_bar.html

    Returns
    -------
    PlotSpec
        Heatmap.
    """
    mapping = mapping or aes()

    # BUILD: dataframe
    frame = build_frame(
        data=data,
        variable_keys=keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
    )

    # unpivot on the keys
    frame = frame.unpivot(
        on=keys,
        index=group_by,
        variable_name=variable_column,
        value_name=value_column,
    )
    # aggregate
    frame = frame.group_by(group_by, variable_column).agg(pl.col(value_column).mean())

    # BUILD: heatmap
    htmp = (
        ggplot(frame)
        + geom_tile(
            aes(x=group_by, y=variable_column, fill=value_column, **mapping.as_dict()),
            tooltips=frame.columns,
            **geom_kwargs,
        )
        + scale_fill_gradient(low=color_low, high=color_high)
        + _THEME_HEATMAP
    )
    htmp += _fill_gradient(
        frame[value_column],
        color_low=color_low,
        color_mid=color_mid,
        color_high=color_high,
        mid_point=mid_point,

    )

    if interactive:
        htmp += ggtb(size_zoomin=-1)

    return htmp
