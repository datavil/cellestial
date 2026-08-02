from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from lets_plot import gggrid, ggtb
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.frames import build_frame
from cellestial.single.common.xyplot import xyplot
from cellestial.util import (
    _collect_aes_columns,
    _determine_axis,
    _resolve_tooltips,
    _select_variable_keys,
)
from cellestial.util.errors import ConflictingLengthError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from lets_plot.plot.subplots import SupPlotsSpec


def xyplots(
    data: AnnData,
    x: str | Sequence[str],
    y: str | Sequence[str],
    *,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    add_keys: Sequence[str] | str | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool | int = False,
    # multi plot args
    layers: Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None = None,
    # grid args
    ncol: int | None = None,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list[float] | None = None,
    heights: list[float] | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    guides: str = "auto",
    **point_kwargs,
) -> SupPlotsSpec:
    """
    Scatter Plots.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    x : str | Sequence[str]
        The key(s) for the x-axis.
    y : str | Sequence[str]
        The key(s) for the y-axis.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
    add_keys : str | Sequence[str] | None, default=None
        Extra metadata columns or variable names to materialise into the shared
        frame, on top of those inferred from `x`, `y`, `mapping`, and `tooltips`.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    observations_name : str, default='Barcode'
        The name to give to barcode (or index) column in the dataframe.
    variables_name : str, default='Variable'
        The name to give to variable index column in the dataframe.
    include_dimensions : bool | int, default=False
        Whether to include dimensions in the DataFrame.
        Providing an integer will limit the number of dimensions to given number.
    layers : Sequence[FeatureSpec|LayerSpec] | FeatureSpec | LayerSpec | None, default=None
        Layers to add to all the plots in the grid.
    ncol : int, default=None
        Number of columns in grid. If not specified, shows plots horizontally, in one row.
    sharex, sharey : str | None, default=None
        Controls sharing of axis limits between subplots in the grid.
        `all` - share limits between all subplots.
        `none` - do not share limits between subplots.
        `row` - share limits between subplots in the same row.
        `col` - share limits between subplots in the same column.
    widths : list[float], default=None
        Relative width of each column of grid, left to right.
    heights : list[float], default=None
        Relative height of each row of grid, top-down.
    hspace : float | None, default=None
        Cell horizontal spacing in px.
    vspace : float | None, default=None
        Cell vertical spacing in px.
    fit : bool | None, default=None
        Whether to stretch each plot to match the aspect ratio of its cell (fit=True),
        or to preserve the original aspect ratio of plots (fit=False).
    align : bool | None, default=None
        If True, align inner areas (i.e. “geom” bounds) of plots.
        However, cells containing other (sub)grids are not participating
        in the plot “inner areas” layouting.
    guides : str, default='auto'
        Specifies how guides (legends and colorbars) should be treated in the layout.
            - 'collect' collect guides from all subplots, removing duplicates.
            - 'keep' keep guides in their original subplots; do not collect at this level.
            - 'auto' allow guides to be collected if an upper-level layout uses guides='collect';

        otherwise, keep them in subplots.
        Duplicates are identified by comparing visual properties:
        For legends: title, labels, and all aesthetic values (colors, shapes, sizes, etc.).
        For colorbars: title, domain limits, breaks, and color gradient.

        For more information on gggrid parameters:
        https://lets-plot.org/python/pages/api/lets_plot.gggrid.html

    **point_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    SupPlotsSpec
        Scatter plot.

    Raises
    ------
    ConflictingLengthError
        If `x` and `y` have incompatible lengths for pairwise plotting or
        broadcasting.

    Examples
    --------
    xyplots allows providing sequences of x and y values.
    Matching the individual x and y values one-by-one.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.xyplots(
            data,
            x=["n_genes_by_counts","n_genes"],
            y=["pct_counts_in_top_200_genes","CD14"],
            mapping=aes(color="cell_type_lvl1"),
            alpha=0.6,
        )

    xyplots also allows broadcasting.
    Allowing one-to-many relationship.

    .. jupyter-execute::
        :emphasize-lines: 10

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()

        cl.xyplots(
            data,
            x="n_genes_by_counts", # to be broadcasted to y
            y=["pct_counts_in_top_200_genes","CD14"],
            mapping=aes(color="cell_type_lvl1"),
            alpha=0.6,
            layers=scale_color_viridis()
        )
    """
    # str to list for x and y
    if isinstance(x, str):
        x = [x]
    if isinstance(y, str):
        y = [y]

    # check for broadcasting
    if len(x) != len(y):
        if len(x) == 1:
            x = list(x) * len(y)
        elif len(y) == 1:
            y = list(y) * len(x)
        else:
            msg = f"Length of x ({len(x)}) and y ({len(y)}) must be the same, or one of them must be of length 1."
            raise ConflictingLengthError(msg)

    # BUILD: one shared frame for all pairs, instead of rebuilding per pair.
    # Mirrors `xyplot`'s build over the union of x/y and shared mapping keys.
    mapping_keys = (
        [key for key in mapping.as_dict().values() if key is not None] if mapping else []
    )
    all_keys = list(dict.fromkeys([*x, *y, *mapping_keys]))
    variable_keys = _select_variable_keys(data=data, keys=all_keys)
    # embedding (X_-prefixed) keys are not features; they request `include_dimensions`
    feature_keys = [key for key in all_keys if not key.startswith("X_")]
    has_dimensions = len(feature_keys) != len(all_keys)
    if has_dimensions:
        include_dimensions = True
    if axis is None:
        observation_dimensions = bool(include_dimensions and data.obsm) and all(
            key in data.obs.columns or key in data.var_names for key in feature_keys
        )
        axis = (
            0
            if (has_dimensions and not feature_keys) or observation_dimensions
            else _determine_axis(data=data, keys=feature_keys)
        )
    if isinstance(add_keys, str):
        add_keys = [add_keys]
    metadata_columns: list[str] = []
    _collect_aes_columns(
        data,
        keys=[*feature_keys, *(add_keys or [])],
        mapping=mapping,
        metadata_columns=metadata_columns,
        variable_keys=variable_keys,
        axis=axis,
    )
    # Tooltips are shared across subplots; pull their fields into the shared
    # frame so each subplot's tooltip validation can find them.
    _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=all_keys,
        metadata_columns=metadata_columns,
        axis=axis,
    )
    observation_column_name = None if tooltips == "none" else observations_name
    variable_column_name = None if tooltips == "none" else variables_name
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observation_column_name,
        variables_name=variable_column_name,
        include_dimensions=include_dimensions,
        metadata_columns=metadata_columns,
    )

    # build plots
    plots = []
    for xi, yi in zip(x, y, strict=True):
        plot = xyplot(
            data,
            x=xi,
            y=yi,
            frame=frame,
            mapping=mapping,
            axis=axis,
            add_keys=add_keys,
            tooltips=tooltips,
            observations_name=observations_name,
            variables_name=variables_name,
            include_dimensions=include_dimensions,
            **point_kwargs,
        )
        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        plots.append(plot)

    scttrs = gggrid(
        plots,
        ncol=ncol,  # ty:ignore[invalid-argument-type]
        sharex=sharex,  # ty:ignore[invalid-argument-type]
        sharey=sharey,  # ty:ignore[invalid-argument-type]
        widths=widths,  # ty:ignore[invalid-argument-type]
        heights=heights,  # ty:ignore[invalid-argument-type]
        hspace=hspace,  # ty:ignore[invalid-argument-type]
        vspace=vspace,  # ty:ignore[invalid-argument-type]
        fit=fit,  # ty:ignore[invalid-argument-type]
        align=align,  # ty:ignore[invalid-argument-type]
        guides=guides,
    )

    if interactive:
        scttrs += ggtb(size_zoomin=-1)

    return scttrs
