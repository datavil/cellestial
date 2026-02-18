from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lets_plot import ggplot

from cellestial.frames import build_frame
from cellestial.util import _determine_axis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.core import FeatureSpec, PlotSpec


def plot(
    data: AnnData,
    mapping: FeatureSpec | None = None,
    *,
    axis: Literal[0, 1] | None = None,
    variable_keys: Sequence[str] | None = None,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool | int = False,
) -> PlotSpec:
    """
    Base plot (for plots without data wrangling).

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    mapping : FeatureSpec | None, default=None
        Aesthetic mappings for the plot, the result of `aes()`.
    axis : Literal[0,1] | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    variable_keys : str | Sequence[str] | None
        Variable keys to add to the DataFrame. If None, no additional keys are added.
    observations_name : str
        The name of the observations column, default is "barcode".
    variables_name : str
        Name for the variables index column, default is 'variable'
    include_dimensions : bool | int, default=False
        Whether to include dimensions in the DataFrame.
        Providing an integer will limit the number of dimensions to given number.

    Returns
    -------
    PlotSpec
        Base ggplot object.
    """
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

    return ggplot(data=frame, mapping=mapping)
