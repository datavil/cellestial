from __future__ import annotations

__lazy_modules__ = ["anndata", "ggrust", "mudata", "pandas"]

from difflib import get_close_matches
from typing import TYPE_CHECKING, Literal

import polars as pl
from anndata import AnnData
from ggrust import (
    aes,
    facet_wrap,
    geom_point,
    geom_pointdensity,
    ggplot,
    labs,
    scale_color_gradient,
)
from mudata import MuData
from pandas import CategoricalDtype

from cellestial.frames import build_frame
from cellestial.layers import _modify_axis
from cellestial.themes import _THEME_DIMENSION
from cellestial.util import _drop_nonfinite_rows, _filter_groups, _resolve_embedding_key
from cellestial.util.errors import (
    EmptySelectionError,
    KeyNotFoundError,
    NonCategoricalColumnError,
    _unsupported_data_type,
)
from cellestial.util.utilities import _container

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ggrust.plot.core import PlotSpec


def embedding_density(
    data: AnnData | MuData,
    group_by: str | None = None,
    *,
    dimensions: Literal["umap", "pca", "tsne"] | str = "umap",
    use_key: str | None = None,
    xy: tuple[int, int] | Sequence[int] = (1, 2),
    groups: Sequence[str] | str | None = None,
    drop: Sequence[str] | str | None = None,
    method: Literal["auto", "neighbors", "kde2d"] = "auto",
    adjust: float = 1,
    size: float = 0.8,
    background_color: str | None = "#e6e6e6",
    color_low: str = "#ffeda0",
    color_high: str = "#bd0026",
    axis_type: Literal["axis", "arrow"] | None = None,
    arrow_length: float = 0.25,
    arrow_size: float = 1,
    arrow_color: str = "#3f3f3f",
    arrow_angle: float = 10,
    **geom_kwargs,
) -> PlotSpec:
    """
    Embedding plot with cells colored by the local density of cells around them.

    Parameters
    ----------
    data : AnnData | MuData
        The single-cell data object.
    group_by : str | None, default=None
        Categorical cell metadata to split the cells by, one panel per group.
        e.g. 'condition' or 'cell_type'. Each panel counts only that group's
        cells. If None, the density of all cells is shown in one panel.
    dimensions : str, default='umap'
        The dimensionality reduction to plot, named without the `X_` prefix.
        e.g. 'umap', 'pca', 'tsne', or a joint embedding such as 'wnn_umap'.
    use_key : str | None, default=None
        The specific key to use for the desired dimensions.
        e.g. 'X_umap_2d'. Otherwise, the key is decided based on `dimensions`.
    xy : tuple[int, int] | Sequence[int], default=(1, 2)
        The x and y axes to use for the plot.
        e.g. (1, 2) for UMAP1 and UMAP2.
    groups : str | Sequence[str] | None, default=None
        Show only specific groups of `group_by`.
    drop : str | Sequence[str] | None, default=None
        Drop specific groups of `group_by`.
    method : {'auto', 'neighbors', 'kde2d'}, default='auto'
        How the density is computed.
        - 'neighbors' counts the cells within a small ellipse around each cell.
        - 'kde2d' reads a 2D Gaussian kernel density estimate at each cell.
        - 'auto' uses 'kde2d' for groups above 20000 cells, else 'neighbors'.
    adjust : float, default=1
        Multiplies the neighbor radius or the kernel bandwidth.
        Larger values give smoother densities.
    size : float, default=0.8
        The size of the points.
    background_color : str | None, default='#e6e6e6'
        Color of every cell drawn behind each group's panel, so the shape of the
        whole embedding stays visible. Only used with `group_by`; None hides it.
    color_low : str, default='#ffeda0'
        The color of the lowest density.
    color_high : str, default='#bd0026'
        The color of the highest density.
    axis_type : {'axis', 'arrow'} | None, default=None
        Whether to use regular axis or arrows as the axis.
    arrow_length : float, default=0.25
        Span of each axis line as a fraction of the panel (0.25 covers 25%).
    arrow_size : float, default=1
        Width of the axis lines.
    arrow_color : str, default='#3f3f3f'
        Color of the arrows.
    arrow_angle : float, default=10
        Angle of the arrow head in degrees.
    **geom_kwargs
        Additional parameters for the `geom_pointdensity` layer, e.g. `alpha`.

    Returns
    -------
    PlotSpec
        Embedding density plot.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If `group_by` is not a cell metadata column, if a group in `groups` or
        `drop` is not found, or if no embedding matches `dimensions`.
    NonCategoricalColumnError
        If `group_by` holds numeric values.
    EmptySelectionError
        If no cell is left after `groups` and `drop`.
    KeyError
        If `xy` does not contain exactly two dimensions.

    Notes
    -----
    The density is computed while the plot is drawn, separately for each panel,
    and scaled to 1 at the densest cell of each panel, so panels of different
    sizes share one color scale. The neighbor radius and the kernel bandwidth
    follow the plot's axis ranges, which span every cell drawn, so they are the
    same in every panel.

    Examples
    --------
    Density of all cells.

    .. jupyter-execute::

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.embedding_density(data)

    One panel per cell type, on the t-SNE embedding.

    .. jupyter-execute::

        cl.embedding_density(data, group_by="cell_type_lvl1", dimensions="tsne")
    """
    # HANDLE: Data types
    if not isinstance(data, (AnnData, MuData)):
        raise _unsupported_data_type(data, AnnData, MuData)

    # HANDLE: group_by
    if group_by is not None:
        _require_observation_column(data, group_by)

    # HANDLE: XY
    if len(xy) != 2:
        msg = f"`xy` must hold exactly two dimensions, e.g. (1, 2), but got {tuple(xy)!r}."
        raise KeyError(msg)
    prefix = _resolve_embedding_key(data=data, dimensions=dimensions, use_key=use_key, xy=xy)
    x = f"{prefix}{xy[0]}"  # e.g. X_UMAP1
    y = f"{prefix}{xy[1]}"  # e.g. X_UMAP2

    # BUILD: dataframe
    frame = build_frame(
        data=data,
        axis=0,
        observation_id_column=None,
        include_dimensions=max(xy),
        metadata_columns=[group_by] if group_by is not None else [],
    )
    frame = _drop_nonfinite_rows(frame, [x, y])
    background = frame.select(x, y)

    if group_by is not None:
        if frame[group_by].dtype.is_numeric():
            msg = (
                f"`group_by` needs categories, but `{group_by}` holds numeric values "
                f"({frame[group_by].dtype}). Pass a categorical key, e.g. `group_by='leiden'`."
            )
            raise NonCategoricalColumnError(msg)
        frame = _filter_groups(frame, column=group_by, groups=groups, drop=drop)
        frame = frame.filter(pl.col(group_by).is_not_null())
        if frame.is_empty():
            msg = (
                f"No cells are left in `group_by` `{group_by}` after `groups` and `drop`. "
                "Keep at least one group, e.g. drop fewer groups."
            )
            raise EmptySelectionError(msg)
        levels = _category_levels(data, group_by)
        if levels is not None:
            # Panels follow the data's category order (e.g. 2 before 10), not text order.
            present = set(frame[group_by].cast(pl.String).unique().to_list())
            levels = [level for level in levels if level in present]
            frame = frame.with_columns(pl.col(group_by).cast(pl.String).cast(pl.Enum(levels)))

    # BUILD: plot
    plot = ggplot(data=frame)
    if group_by is not None and background_color is not None:
        # A layer without the facet column is not repeated in every panel by the
        # backend, so the background is copied into each group explicitly.
        panel_groups = frame.select(pl.col(group_by).unique(maintain_order=True))
        plot += geom_point(
            mapping=aes(x=x, y=y),
            data=background.join(panel_groups, how="cross"),
            color=background_color,
            size=size,
            tooltips="none",
        )
    plot += geom_pointdensity(
        mapping=aes(x=x, y=y, color="..ndensity.."),
        method=method,
        adjust=adjust,
        size=size,
        **geom_kwargs,
    )
    plot += scale_color_gradient(low=color_low, high=color_high, name="density")
    plot += _THEME_DIMENSION
    if group_by is not None:
        plot += facet_wrap(facets=group_by, order=0 if frame[group_by].dtype == pl.Enum else 1)

    # HANDLE: axis labels, UMAP1 rather than X_UMAP1
    if dimensions == "tsne":
        plot += labs(x=f"tSNE{xy[0]}", y=f"tSNE{xy[1]}")
    else:
        plot += labs(x=x.replace("X_", ""), y=y.replace("X_", ""))

    # HANDLE: arrow axis
    plot += _modify_axis(
        axis_type=axis_type,
        arrow_size=arrow_size,
        arrow_color=arrow_color,
        arrow_angle=arrow_angle,
        arrow_length=arrow_length,
    )
    return plot


def _require_observation_column(data: AnnData | MuData, column: str) -> None:
    """Raise `KeyNotFoundError` if `column` is not a cell metadata column, naming close matches."""
    available = list(_container(data).observation_columns())
    if column in available:
        return
    msg = f"`group_by` key `{column}` not found in cell metadata."
    close_matches = get_close_matches(column, available, n=3)
    if close_matches:
        msg += f" Did you mean {' or '.join(repr(match) for match in close_matches)}?"
    raise KeyNotFoundError(msg)


def _category_levels(data: AnnData | MuData, column: str) -> list[str] | None:
    """Return the category order of a categorical cell metadata `column`, or None if it has none."""
    values = _container(data).observation_metadata()[column]
    if isinstance(values.dtype, CategoricalDtype):
        return [str(level) for level in values.cat.categories]
    return None
