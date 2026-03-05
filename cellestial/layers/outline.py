from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl
from lets_plot import aes, geom_path
from lets_plot.plot.core import FeatureSpec, LayerSpec, PlotSpec
from polars import DataFrame

from cellestial.util import get_mapping


def _get_density_boundaries(
    frame: DataFrame,
    x: str,
    y: str,
    group_by: str,
    groups: str | Sequence[str | Sequence[str]],
    padding: float = 1,
    level: float = 0.1,
    grid_size: int = 200,
) -> pl.DataFrame:
    """Creates and Returns a DataFrame to encircle the cluster via `geom_path`."""
    from scipy.stats import gaussian_kde
    from skimage import measure

    # 1. Normalize input to a Sequence of Sequences: List[List[str]]
    if isinstance(groups, str):
        groups = [[groups]]
    else:
        groups = [
            [g] if isinstance(g, str) else list(g)
            for g in groups
        ]

    boundaries = []
    for group in groups:
        group_label = "+".join(group) if len(group) > 1 else group[0]
        points = frame.filter(pl.col(group_by).is_in(group)).select([x, y]).to_numpy()
        if len(points) < 5:
            continue

        # 2. KDE Calculation
        kde = gaussian_kde(points.T)

        # 3. Create Grid
        x_min, y_min = points.min(axis=0) - padding
        x_max, y_max = points.max(axis=0) + padding

        x_range = np.linspace(x_min, x_max, grid_size)
        y_range = np.linspace(y_min, y_max, grid_size)
        xi, yi = np.meshgrid(x_range, y_range)

        # 4. Evaluate Density
        zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

        # 5. Extract Contours (The skimage magic)
        threshold = zi.max() * level
        # find_contours returns a list of [row, col] arrays
        contours = measure.find_contours(zi, threshold)

        for i, contour in enumerate(contours):
            # Map grid indices back to DIM coordinates
            # Note: skimage returns (row, col) which maps to (y_index, x_index)
            actual_x = np.interp(contour[:, 1], np.arange(grid_size), x_range)
            actual_y = np.interp(contour[:, 0], np.arange(grid_size), y_range)

            boundaries.append(
                pl.DataFrame(
                    {
                        x: actual_x,
                        y: actual_y,
                        group_by: [group_label] * len(actual_x),
                        "path": [f"{group_label}_{i}"] * len(actual_x),
                    }
                )
            )

    return pl.concat(boundaries)


def cluster_outlines(
    plot: PlotSpec,
    /,
    groups: str | Sequence[str | Sequence[str]],
    *,
    padding: float = 1.5,
    level: float = 0.04,
    grid_size: int = 200,
    color: str = "#1f1f1f",
    linetype: Literal[
        "blank", "solid", "dashed", "dotted", "dotdash", "longdash", "twodash"
    ] = "dashed",
    mapping: FeatureSpec | None = None,
    size: float = 1,
    group_by: str | None = None,
    x: str | None = None,
    y: str | None = None,
    **geom_kwargs,
) -> LayerSpec:
    """
    Returns a Layer of `geom_path` that outlines the given clusters.

    Parameters
    ----------
    plot : PlotSpec
        The plot to which the layer will be added. Used to extract data and aesthetics.
    groups : str | Sequence[str | Sequence[str]]
        The group(s) to outline. Can be a single group name or a list of group
        Providing string(s) will outline clusters with those group names.
        Providing nested sequences of strings combines each sequence into a their own group.
        E.g `groups=[['A', 'B']]` will outline groups A and B as if they were one cluster.
    padding : float, default=1.5
        The spatial buffer added to the cluster's bounding box before calculating density.
        Increasing this value allows the density 'cloud' to expand further from the outermost
        points, ensuring the resulting curve is smooth and not truncated at the data edges.
    level : float, default=0.04
        The density threshold for the outline, expressed as a fraction of the peak density
        (0.0 to 1.0). A lower value creates a larger, more inclusive 'bubble' that captures
        outliers, while a higher value creates a tighter boundary around the cluster's core.
    grid_size : int, default=200
        The resolution of the internal sampling grid used for the Marching Squares algorithm.
        A higher value produces a smoother, more 'curvy' path by increasing the number of
        interpolation points, while a lower value improves calculation speed but may
        result in a more jagged or 'pixelated' appearance.
    color : str, default='#1f1f1f'
        The color of the outline.
    linetype : str, default='dashed'
        The linetype of the outline. E.g 'dashed', 'dotted',
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    size : float, default=1
        The size of the outline.
    group_by : str | None, default=None
        The column name in the data used to group clusters by. e.g 'cell_type'.
        If None, it will be inferred from the plot aesthetics.
    x : str | None, default=None
        The column name in the data used for x-axis coordinates. e.g 'X_UMAP1'.
        If None, it will be inferred from the plot aesthetics.
    y : str | None, default=None
        The column name in the data used for y-axis coordinates. e.g 'X_UMAP2'.
        If None, it will be inferred from the plot aesthetics.
    **geom_kwargs
        Additional parameters for the `geom_path` layer.
        For more information on geom_path parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_path.html

    Returns
    -------
    LayerSpec
        Cluster Outlines.
    """
    # get mapping
    _mapping = get_mapping(plot, index=0)
    x = _mapping.get("x") if x is None else x
    y = _mapping.get("y") if y is None else y
    group_by = _mapping.get("color") if group_by is None else group_by
    if x is None:
        msg = "`x` is present neither as argument nor in the plot aesthetics."
        raise ValueError(msg)
    if y is None:
        msg = "`y` is present neither as argument nor in the plot aesthetics."
        raise ValueError(msg)
    if group_by is None:
        msg = "`group_by` is present neither as argument nor in the plot aesthetics."
        raise ValueError(msg)
    # get data
    frame = plot.get_plot_shared_data()
    # get boundaries
    frame = _get_density_boundaries(
        frame,
        x=x,
        y=y,
        group_by=group_by,
        groups=groups,
        padding=padding,
        level=level,
        grid_size=grid_size,
    )
    # create and return the plot
    if mapping is None:
        mapping = aes()
    return geom_path(
        data=frame,
        mapping=aes(x=x, y=y, group="path", **mapping.as_dict()),
        color=color,
        linetype=linetype,
        size=size,
        **geom_kwargs,
    )
