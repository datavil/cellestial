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


def _normalize_widths(
    plots: Sequence[Sequence[PlotSpec | SupPlotsSpec]],
    widths: Sequence[float] | Sequence[Sequence[float]] | None,
) -> list[list[float] | None]:
    """Resolve `widths` into one width vector (or None) per row."""
    if widths is None:
        return [None] * len(plots)

    if not isinstance(widths, Sequence) or isinstance(widths, str):
        msg = f"`widths` must be a flat or nested sequence, but received {type(widths)}"
        raise TypeError(msg)

    is_nested = any(isinstance(row_widths, (list, tuple)) for row_widths in widths)
    if is_nested:
        if not all(isinstance(row_widths, (list, tuple)) for row_widths in widths):
            msg = "`widths` mixes numbers and sub-lists; use a fully nested list, one per row."
            raise ValueError(msg)
        if len(widths) != len(plots):
            msg = f"nested `widths` has {len(widths)} rows but `plots` has {len(plots)} rows."
            raise ValueError(msg)
        for index, (row, row_widths) in enumerate(zip(plots, widths, strict=True)):
            if len(row_widths) != len(row):
                msg = (
                    f"`widths[{index}]` has {len(row_widths)} values "
                    f"but row {index} has {len(row)} plots."
                )
                raise ValueError(msg)
        return [list(row_widths) for row_widths in widths]

    # flat widths: broadcast to every row, valid only when all rows share that length.
    uneven = [index for index, row in enumerate(plots) if len(row) != len(widths)]
    if uneven:
        msg = (
            f"flat `widths` of length {len(widths)} cannot broadcast to rows {uneven} "
            "with differing plot counts; pass a nested `widths` instead."
        )
        raise ValueError(msg)
    return [list(widths) for _ in plots]


def layout(
    plots: Sequence[Sequence[PlotSpec | SupPlotsSpec] | PlotSpec | SupPlotsSpec],
    *,
    widths: Sequence[float] | Sequence[Sequence[float]] | None = None,
    heights: Sequence[float] | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
) -> SupPlotsSpec:
    """
    Arrange plots into a grid of rows, where each inner list is one row.

    Unlike a uniform grid, rows may hold different numbers of plots. Each row is
    sized independently, so cells in different rows do not share column boundaries.

    Parameters
    ----------
    plots : Sequence[Sequence[PlotSpec | SupPlotsSpec] | PlotSpec | SupPlotsSpec]
        A sequence of rows. Each element is either a sequence of plots or a single
        plot, which is treated as a one-plot row.
    widths : Sequence[float] | Sequence[Sequence[float]] | None, default=None
        Relative cell widths within each row. A nested sequence assigns widths
        per row (each inner length must match that row's plot count). A flat
        sequence is broadcast to every row and is only valid when all rows have
        the same plot count. `None` distributes widths equally.
    heights : Sequence[float] | None, default=None
        Relative row heights, one value per row. `None` distributes heights equally.
    hspace : float | None, default=None
        Horizontal spacing between cells within a row, in pixels.
    vspace : float | None, default=None
        Vertical spacing between rows, in pixels.

    Returns
    -------
    SupPlotsSpec
        The composed grid.

    Raises
    ------
    TypeError
        If `plots` is not a nested sequence or `widths` is of an unexpected type.
    ValueError
        If any row is empty, or `widths`/`heights` lengths do not match the layout.

    Examples
    --------
    Arrange plots into ragged rows.

    .. jupyter-execute ::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k(cache_directory="data")

        first = cl.umap(data, key="CD14", axis_type="arrow")
        second = cl.umap(data, key="MALAT1", axis_type="arrow")
        third = cl.umap(data, key="leiden", axis_type="arrow")

        cl.layout(
            [[first, second],
             third],
            widths=[[1, 2], [1]],
            heights=[1, 1],
        )

    """
    if not isinstance(plots, Sequence) or isinstance(plots, str):
        msg = f"`plots` must be a nested sequence of rows, but received {type(plots)}"
        raise TypeError(msg)
    if len(plots) == 0:
        msg = "`plots` must contain at least one row."
        raise ValueError(msg)
    plot_rows: list[list[PlotSpec | SupPlotsSpec]] = []
    for index, row in enumerate(plots):
        if isinstance(row, (PlotSpec, SupPlotsSpec)):  # a bare plot becomes its own row.
            plot_rows.append([row])
            continue
        if not isinstance(row, Sequence) or isinstance(row, str):
            msg = f"row {index} must be a plot or a sequence of plots, but received {type(row)}"
            raise TypeError(msg)
        if len(row) == 0:
            msg = f"row {index} is empty; every row must contain at least one plot."
            raise ValueError(msg)
        plot_rows.append(list(row))

    if heights is not None and len(heights) != len(plot_rows):
        msg = f"`heights` has {len(heights)} values but `plots` has {len(plot_rows)} rows."
        raise ValueError(msg)

    row_widths = _normalize_widths(plot_rows, widths)
    rows = [
        gggrid(row, ncol=len(row), widths=row_widths[index], hspace=hspace)
        for index, row in enumerate(plot_rows)
    ]
    return gggrid(rows, ncol=1, heights=heights, vspace=vspace)


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
