from collections.abc import Sequence

import numpy as np
import polars as pl
from lets_plot import aes, geom_path
from lets_plot.plot.core import FeatureSpec, LayerSpec, PlotSpec
from polars import DataFrame
from scipy.stats import gaussian_kde
from skimage import measure

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
    if isinstance(groups, str):
        groups = [groups]

    boundaries = []
    for group in groups:
        # 1. Isolate cluster points
        if isinstance(group, str):
            _group = [group]
        elif isinstance(group, Sequence):
            _group = list(group)
        points = frame.filter(pl.col(group_by).is_in(_group)).select([x, y]).to_numpy()
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
                        group_by: [group] * len(actual_x),
                        "path": [f"{group}_{i}"] * len(actual_x),
                    }
                )
            )

    return pl.concat(boundaries)

def outline_clusters(
    plot: PlotSpec,
    /,
    groups: str | Sequence[str | Sequence[str]],
    *,
    padding: float = 1.5,
    level: float = 0.04,
    grid_size: int = 200,
    color: str = "#1f1f1f",
    linetype: str = "dashed",
    size: float = 1,
    group_by: str | None = None,
    mapping: FeatureSpec = None,
    x: str | None = None,
    y: str | None = None,
    **geom_kwargs,
) -> LayerSpec:
    """Returns a Layer of `geom_path` that outlines the given clusters."""
    # get mapping
    _mapping = get_mapping(plot,index=0)
    x = _mapping.get("x") if x is None else x
    y = _mapping.get("y") if y is None else y
    group_by = _mapping.get("color") if group_by is None else group_by
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
        mapping=aes(x=x, y=y, group="path",**mapping.as_dict()),
        color=color,
        linetype=linetype,
        size=size,
    )
