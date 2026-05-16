from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from lets_plot import gggrid
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec

if TYPE_CHECKING:
    from polars import DataFrame


def _grid_figures(grid: SupPlotsSpec) -> list[PlotSpec]:
    # Lets-Plot does not expose the underlying PlotSpec list on SupPlotsSpec;
    # `.as_dict()["figures"]` returns serialized dicts that `gggrid` cannot consume.
    # Name-mangled access is the only path to the live PlotSpec objects.
    return vars(grid)["_SupPlotsSpec__figures"]


def get_figure(grid: SupPlotsSpec, index: int) -> PlotSpec:
    """
    Get a single figure from a grid (SupPlotsSpec) by index.

    Parameters
    ----------
    grid : SupPlotsSpec
        The grid to pull a figure from.
    index : int
        The index of the figure to return. Negative indices are supported.

    Returns
    -------
    PlotSpec
        The selected figure.

    Raises
    ------
    TypeError
        If `grid` is not a SupPlotsSpec or `index` is not an int.
    IndexError
        If `index` is out of range for the grid.

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

        cl.get_figure(grid, index=3)

    """
    if not isinstance(grid, SupPlotsSpec):
        msg = f"Expected `SupPlotsSpec`, but received {type(grid)}"
        raise TypeError(msg)
    if not isinstance(index, int) or isinstance(index, bool):
        msg = f"Expected int for index, but received {type(index)}"
        raise TypeError(msg)

    figures = _grid_figures(grid)
    if not -len(figures) <= index < len(figures):
        msg = f"index {index} is out of range for a grid with {len(figures)} figures."
        raise IndexError(msg)
    return figures[index]


def get_figures(grid: SupPlotsSpec, indices: Sequence[int], **kwargs) -> SupPlotsSpec:
    """
    Get multiple figures from a grid (SupPlotsSpec) as a new grid.

    Parameters
    ----------
    grid : SupPlotsSpec
        The grid to pull figures from.
    indices : Sequence[int]
        The indices of the figures to include. Negative indices are supported.
    **kwargs : dict[str, Any]
        Additional arguments forwarded to `gggrid`.
        see: https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    Returns
    -------
    SupPlotsSpec
        A new grid containing the selected figures.

    Raises
    ------
    TypeError
        If `grid` is not a SupPlotsSpec or `indices` is not a Sequence of ints.
    IndexError
        If any value in `indices` is out of range for the grid.

    Examples
    --------
    Get multiple plots from a grid.

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

        cl.get_figures(grid, indices=[1, 3])

    """
    if not isinstance(grid, SupPlotsSpec):
        msg = f"Expected `SupPlotsSpec`, but received {type(grid)}"
        raise TypeError(msg)
    if not isinstance(indices, Sequence) or isinstance(indices, str):
        msg = f"Expected Sequence[int] for indices, but received {type(indices)}"
        raise TypeError(msg)

    figures = _grid_figures(grid)
    out_of_range = [i for i in indices if not -len(figures) <= i < len(figures)]
    if out_of_range:
        msg = f"indices {out_of_range} are out of range for a grid with {len(figures)} figures"
        raise IndexError(msg)
    return gggrid([figures[i] for i in indices], **kwargs)


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
    ValueError
        If no DataFrame can be retrieved from the plot.

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
