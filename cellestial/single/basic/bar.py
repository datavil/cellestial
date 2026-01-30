from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anndata import AnnData
from lets_plot import geom_bar, ggplot

from cellestial.frames import build_frame

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec, PlotSpec


def bar(
    data: AnnData,
    axis: Literal[0, 1] | None = None,
    *,
    mapping: FeatureSpec | None = None,
    variable_keys: Sequence[str] | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool = False,
    **geom_kwargs,
) -> PlotSpec:
    """
    Base plot for plots without data wrangling.

    Parameters
    ----------
    data
        The AnnData object of the single cell data.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the plot, the result of `aes()`.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    axis : Literal[0,1] | None
        The axis to build the frame for. 0 for observations, 1 for variables.
    observations_name : str
        The name of the observations column, default is "barcode".
    variables_name : str
        Name for the variables index column, default is 'variable'
    include_dimensions : bool
        Whether to include dimensionality reductions fields.
    **geom_kwargs
        Additional parameters for the `geom_bar` layer.
        For more information on geom_bar parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_bar.html

    Returns
    -------
    PlotSpec
        Base ggplot object.
    """
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
    )

    bar = ggplot(data=frame) + geom_bar(mapping=mapping, **geom_kwargs)

    return bar
