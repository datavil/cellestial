from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import polars as pl
from lets_plot import (
    aes,
    arrow,
    geom_path,
    geom_segment,
)

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import get_mapping, retrieve
from cellestial.util.errors import MissingAestheticError

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpec, FeatureSpecArray, PlotSpec


def stream(
    *,
    plot: PlotSpec | None = None,
    color: str = "#1f1f1f",
    alpha: float = 0.7,
    size: float = 1,
    velocity_prefix: str = "velocity_",
    velocity_key: str | None = None,
    grid_density: float = 1,
    smooth: float = 0.5,
    n_neighbors: int | None = None,
    min_mass: int = 1,
    cutoff_percentile: float | None = None,
    density: float = 2,
    max_length: float = 4,
    integration_direction: Literal["forward", "backward", "both"] = "both",
    mapping: FeatureSpec | None = None,
    arrow_color: str = "#1f1f1f",
    arrow_size: float = 1,
    arrow_alpha: float = 0.7,
    arrows: FeatureSpec | None = None,
    arrow_kwargs: dict | None = None,
    **geom_kwargs,
) -> DeferredLayer:
    """
    Layer of streams of velocities on the embedding.

    Parameters
    ----------
    plot : PlotSpec | None, default=None
        If provided, streams are computed from this plot's data and aesthetics
        regardless of which plot the resulting layer is added to. When `None`,
        the layer is deferred and introspects the plot it is added to via `+`.
    color : str, default="#1f1f1f"
        Color of the stream lines.
    alpha : float, default=0.7
        Alpha of the stream lines.
    size : float, default=1
        Size of the stream lines.
    velocity_prefix : str, default="velocity_"
        Prefix for the velocity columns in the data.
        The function will look for columns with this prefix followed by the embedding number.
        (e.g. "velocity_UMAP1", "velocity_UMAP2").
    velocity_key : str | None, default=None
        If provided, will be used as the column names for the velocity.
        E.g f"{velocity_key}1", f"{velocity_key}2".
        If None, defaults to using velocity_prefix + embedding name.
    grid_density : float, default=1
        Density of the grid for velocity embedding.
    smooth : float, default=0.5
        Multiplication factor for scale in Gaussian kernel around grid point.
    n_neighbors : int | None, default=None
        Number of neighbors to consider around grid point.
    min_mass : int, default=1
        Minimum threshold for mass to be shown.
        It can range between 0 (all velocities) and 5 (large velocities only).
    cutoff_percentile : float | None, default=None
        If set, mask small velocities below a percentile threshold (between 0 and 100).
    density : float, default=2
        Controls the closeness of streamlines. When density = 2 (default), the domain
        is divided into a 60x60 grid, whereas density linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use a tuple (density_x, density_y).
    max_length : float, default=4
        Maximum length of streamline in axes coordinates.
    integration_direction : {'forward', 'backward', 'both'}, default='both'
        Integrate the streamline in 'forward', 'backward' or 'both' directions.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    arrow_color : str | None, default='#1f1f1f'
        Color of the arrows. If None, defaults to the same color as the stream lines.
    arrow_size : float, default=1
        Size of the arrows.
    arrow_alpha : float, default=0.7
        Alpha of the arrows.
    arrows : FeatureSpec | None, default=None
        The result of `arrow()`, used to specify arrow aesthetics. If None, defaults to arrow().
        For more information on arrow parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.arrow.html
    arrow_kwargs : dict | None, default=None
        Additional keyword arguments for the arrows to be passed to geom_segment. E.g {"linestyle": "dashed"}.
        For more information on arrow parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_segment.html
    **geom_kwargs
        Additional parameters for the `geom_path` layer.
        For more information on geom_path parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_path.html

    Returns
    -------
    DeferredLayer

    Raises
    ------
    MissingAestheticError
        If `x` or `y` cannot be inferred from the receiving plot.
    ValueError
        If dimension numbers cannot be parsed when `velocity_key` is provided.
    KeyError
        If the required velocity columns are not present in the plot data.
    ImportError
        If `scvelo` is not installed.

    Notes
    -----
    This function requires the `scvelo` package to be installed.
    This function utilizes the `matplotlib`'s `streamplot` function to get stream lines.

    Examples
    --------
    Stream layers can be added to dimensionality reduction plots.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pancreas(cache_directory="data")

        plot = cl.umap(
            data,
            key="clusters_coarse",
            axis_type="arrow",
            size=4,
            alpha=0.4,
            legend_ondata=True,
            ondata_color="black",
        )
        plot + cl.stream()

    Stream as a separate layer to showcase the modularity of layers in Cellestial.
    Here the data source is passed explicitly because the receiving plot
    (`ggplot()`) has no embedding to introspect.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pancreas(cache_directory="data")

        plot = cl.umap(
            data,
            key="clusters_coarse",
            axis_type="arrow",
            size=4,
            alpha=0.4,
            legend_ondata=True,
            ondata_color="black",
        )
        gggrid([plot, ggplot() + cl.stream(plot=plot)])

    """
    explicit_plot = plot

    def _build(receiving_plot: PlotSpec) -> FeatureSpecArray:
        source = explicit_plot if explicit_plot is not None else receiving_plot
        # extract data and mapping from plot
        frame = retrieve(source)
        _mapping = get_mapping(source)
        x = _mapping.get("x")
        y = _mapping.get("y")
        if x is None:
            msg = "`x` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)
        if y is None:
            msg = "`y` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)

        # determine velocity column names
        if velocity_key is not None:  # if the velocity name is provided
            x_match = re.match(r"(.+?)(\d+)$", x)
            y_match = re.match(r"(.+?)(\d+)$", y)
            if x_match is None or y_match is None:
                msg = f"Could not parse dimension number from mapping keys: x={x!r}, y={y!r}"
                raise ValueError(msg)
            *_, x_number = x_match.groups()
            *_, y_number = y_match.groups()
            x_velocity = f"{velocity_key}{x_number}"
            y_velocity = f"{velocity_key}{y_number}"
        elif velocity_key is None:  # default to velocity_prefix + embedding name
            prefix = velocity_prefix.upper()  # cellestial converts embedding names to uppercase
            x_velocity = x.replace("X_", prefix)
            y_velocity = y.replace("X_", prefix)

        missing_velocity_columns = [
            column for column in (x_velocity, y_velocity) if column not in frame.columns
        ]
        if missing_velocity_columns:
            msg = (
                f"Velocity columns not found: {missing_velocity_columns}. "
                "Pass `velocity_prefix=` or `velocity_key=` if your velocity columns "
                "use different names."
            )
            raise KeyError(msg)

        # contained imports
        try:
            from matplotlib.figure import Figure
            from scvelo.plotting.velocity_embedding_grid import compute_velocity_on_grid
        except ImportError as err:
            msg = "`scvelo` must be installed to use the stream layer."
            raise ImportError(msg) from err

        # extract coordinates and velocities as numpy arrays
        dimensions = frame.select(pl.col(x), pl.col(y)).to_numpy()
        velocities = frame.select(pl.col(x_velocity), pl.col(y_velocity)).to_numpy()

        # compute velocity grid using scvelo's compute_velocity_on_grid
        X_grid, V_grid = compute_velocity_on_grid(
            X_emb=dimensions,  # UMAP/tSNE etc. coords
            V_emb=velocities,  # velocities
            density=grid_density,
            smooth=smooth,
            n_neighbors=n_neighbors,
            min_mass=min_mass,
            autoscale=False,
            adjust_for_stream=True,  # reshape output into the meshgrid format for streams
            cutoff_perc=cutoff_percentile,
        )

        # mock build streamplot to extract line segment coordinates
        ax = Figure().add_subplot(1, 1, 1)
        _stream = ax.streamplot(
            X_grid[0],
            X_grid[1],
            V_grid[0],
            V_grid[1],
            density=density,
            maxlength=max_length,
            integration_direction=integration_direction,
        )
        line_paths = _stream.lines.get_paths()  # extract line segment coordinates

        # extract path segments from streamplot and convert to DataFrame
        records_path = [
            {"x": float(x), "y": float(y), "group": i}
            for i, path in enumerate(line_paths)
            for x, y in path.vertices
        ]
        frame_streams = pl.DataFrame(records_path)
        # compute arrow coordinates (midpoint of each path segment)
        frame_arrows = (
            (
                frame_streams.group_by("group")
                .agg(pl.col("x"), pl.col("y"))
                .with_columns(mid=pl.col("x").list.len() // 2)
            )
            .with_columns(
                pl.col("x").list.get(pl.col("mid")).alias("x"),
                pl.col("y").list.get(pl.col("mid")).alias("y"),
                pl.col("x").list.get(pl.col("mid").add(1)).alias("xend"),
                pl.col("y").list.get(pl.col("mid").add(1)).alias("yend"),
            )
            .drop("mid")
            .sort("group")
        )

        # handle arrow kwargs
        _arrow_kwargs = {
            "color": arrow_color,
            "size": arrow_size,
            "alpha": arrow_alpha,
        }
        if arrow_kwargs is not None:
            _arrow_kwargs.update(arrow_kwargs)

        # handle mapping
        local_mapping = mapping or aes()

        # handle arrow
        local_arrows = arrow() if arrows is None else arrows

        # build the stream layer
        layer = geom_path(
            data=frame_streams,
            mapping=aes("x", "y", group="group", **local_mapping.as_dict()),
            color=color,
            alpha=alpha,
            size=size,
            **geom_kwargs,
        ) + geom_segment(
            data=frame_arrows,
            mapping=aes("x", "y", xend="xend", yend="yend"),
            arrow=local_arrows,
            **_arrow_kwargs,
        )
        return layer

    return DeferredLayer(_build)
