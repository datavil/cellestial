from __future__ import annotations

from typing import TYPE_CHECKING, Literal

# Data retrieval
from lets_plot import (
    geom_point,
    ggtb,
)

from cellestial.single.base import plot as baseplot
from cellestial.util import _determine_axis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.core import FeatureSpec, PlotSpec


def scatter(
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
    Scatter Plot.

    data : AnnData
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
    include_dimensions : bool | int, default=False
        Whether to include dimensions in the DataFrame.
        Providing an integer will limit the number of dimensions to given number.
    interactive : bool, default=False
        Whether to make the plot interactive.
    **geom_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Scatter plot.

    """
    if mapping is not None:
        keys = [v for v in mapping.as_dict().values() if v is not None]
        axis = _determine_axis(data=data, keys=keys) if axis is None else axis
    # BUILD: the scatter plot
    scttr = (
        baseplot(
            data=data,
            mapping=None,
            axis=axis,
            variable_keys=variable_keys,
            observations_name=observations_name,
            variables_name=variables_name,
            include_dimensions=include_dimensions,
        )
        + geom_point(mapping=mapping, **geom_kwargs)
    )

    if interactive:
        scttr += ggtb(size_zoomin=-1)

    return scttr
