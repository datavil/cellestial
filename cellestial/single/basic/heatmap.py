from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anndata import AnnData
from lets_plot import geom_tile, ggplot, ggtb

from cellestial.frames import build_frame
from cellestial.themes import _THEME_HEATMAP
from cellestial.util import _determine_axis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec, PlotSpec

def heatmap(
    data: AnnData,
    mapping: FeatureSpec | None = None,
    *,
    axis: Literal[0, 1] | None = None,
    variable_keys: Sequence[str] | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool = False,
    interactive: bool = False,
    **geom_kwargs,
) -> PlotSpec:
    """
    Heatmap.

    Parameters
    ----------
    data
        The AnnData object of the single cell data.
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the plot, the result of `aes()`.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    variable_keys : str | Sequence[str] | None, default=None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    axis : Literal[0,1] | None, default=None
        The axis to build the frame for. 0 for observations, 1 for variables.
    observations_name : str, default="Barcode"
        The name of the observations column.
    variables_name : str, default="Variable"
        Name for the variables index column.
    include_dimensions : bool, default=False
        Whether to include dimensionality reductions fields.
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
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # BUILD: the dataframe
    if mapping is not None:
        keys = [v for v in vars(mapping)["_FeatureSpec__props"].values() if v is not None]
        axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
    )
    # BUILD: the bar plot
    htmp = ggplot(data=frame) + geom_tile(mapping=mapping, **geom_kwargs) + _THEME_HEATMAP

    if interactive:
        htmp += ggtb(size_zoomin=-1)

    return htmp
