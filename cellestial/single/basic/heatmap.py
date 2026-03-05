from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lets_plot import geom_tile, ggtb

from cellestial.single.base import plot as baseplot
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import _determine_axis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.core import FeatureSpec, PlotSpec


def heatmap(
    data: AnnData,
    mapping: FeatureSpec | None = None,
    *,
    axis: Literal[0, 1] | None = None,
    variable_keys: Sequence[str] | None = None,
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
    axis : Literal[0,1] | None, default=None
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
    if mapping is not None:
        keys = [v for v in mapping.as_dict().values() if v is not None]
        axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    htmp = (
        baseplot(
            data=data,
            mapping=None,
            axis=axis,
            variable_keys=variable_keys,
            observations_name=observations_name,
            variables_name=variables_name,
            include_dimensions=include_dimensions,
        )
        + geom_tile(mapping=mapping, **geom_kwargs)
        + _THEME_HEATMAP
    )

    if interactive:
        htmp += ggtb(size_zoomin=-1)

    return htmp
