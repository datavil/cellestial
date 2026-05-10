from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lets_plot import gggrid
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

if TYPE_CHECKING:
    from polars import DataFrame


def get_slice(
    grid: SupPlotsSpec, index: int | Sequence[int], **kwargs
) -> PlotSpec | SupPlotsSpec | None:
    """
    Slice a grid object (SupPlotsSpec) with the given index.

    Parameters
    ----------
    grid : SupPlotsSpec
        The grid to slice.
    index : int | Sequence[int]
        The index or indices of the plots to slice.
    **kwargs : dict[str, Any]
        Additional arguments for the `gggrid` function.
        see: https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    Returns
    -------
    PlotSpec | SupPlotsSpec
        The sliced grid.

    Raises
    ------
    TypeError
        If the grid is not a SupPlotsSpec object.
        If the index is not an int or Sequence[int].

    Examples
    --------
    Get a single plot from a grid.

    .. jupyter-execute ::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        grid = cl.expressions(
            data,
            keys=["MALAT1", "YBX3", "MNDA" ,"HLA-DRA"],
            axis_type="arrow",
            color_high="red",
            ncol=2,
        )

        cl.slice(grid, index=3)

    Get a multiple plots from a grid.

    .. jupyter-execute ::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        grid = cl.expressions(
            data,
            keys=["MALAT1", "YBX3", "MNDA" ,"HLA-DRA"],
            axis_type="arrow",
            color_high="red",
            ncol=2,
        )

        cl.slice(grid, index=[1,3])

    """
    if isinstance(grid, SupPlotsSpec):
        figures = vars(grid).get("_SupPlotsSpec__figures")

        if figures is not None:
            if isinstance(index, int):
                plot = figures[index]
                return plot
            elif isinstance(index, Sequence):
                list_plots = [figures[i] for i in index]
                return gggrid(list_plots, **kwargs)
            else:
                msg = f"Expected int or Sequence for index, but received {type(index)}"
                raise TypeError(msg)
    else:
        msg = f"Expected `SupPlotsSpec`, but received {type(grid)}"
        raise TypeError(msg)


def get_mapping(plot: PlotSpec, *, index: int = 0) -> dict:
    """
    Returns the mapping of the plot as a `dict`.

    Parameters
    ----------
    plot : PlotSpec
        The plot to get mapping from.
    index : int, default=0
        index of the layer to get the local mapping from.

    Returns
    -------
    dict
        The combined mapping of the plot as a dict.

    Examples
    --------
    Get the mapping of a plot.

    .. jupyter-execute ::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")
        umap = cl.umap(data,key="CD14",axis_type="arrow",color_high="red")

        cl.get_mapping(umap)
    """
    return {
        **plot.as_dict().get("mapping"),  # from the global mapping,
        **plot.as_dict().get("layers")[index].get("mapping"),  # from a layer.
    }


def retrieve(plot: PlotSpec | SupPlotsSpec, index: int = 0) -> DataFrame:
    """
    Retrieves the dataframe from a PlotSpec or SupPlotsSpec using the index.

    Parameters
    ----------
    plot : PlotSpec | SupPlotsSpec
        The plot to retrieve the dataframe from.
    index : int, default=0
        The index of the figure to retrieve the dataframe from.

    Returns
    -------
    DataFrame
        The dataframe utilized in the plot.

    Raises
    ------
    TypeError
        If the plot is not a PlotSpec or SupPlotsSpec object.

    Examples
    --------

    .. jupyter-execute ::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")
        umap = cl.umap(data,key="CD14",axis_type="arrow",color_high="red")

        cl.retrieve(umap).head()
    """
    if isinstance(plot, PlotSpec):
        frame = plot.as_dict().get("data")
    elif isinstance(plot, SupPlotsSpec):
        frame = plot.as_dict().get("figures")[index].get("data")
    else:
        msg = f"Plot MUST be a `PlotSpec` or `SupPlotsSpec` object, type={type(plot)}"
        raise TypeError(msg)

    if frame is None:
        msg = "Could not retrieve the dataframe from the plot."
        raise ValueError(msg)

    return frame
