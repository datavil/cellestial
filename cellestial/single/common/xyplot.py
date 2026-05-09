from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from anndata import AnnData
from lets_plot import (
    aes,
    geom_point,
    ggplot,
    ggtb,
    labs,
)
from lets_plot.plot.core import FeatureSpec

from cellestial.frames import build_frame
from cellestial.themes import _THEME_SCATTER
from cellestial.util import (
    _determine_axis,
    _resolve_tooltips,
    _select_variable_keys,
    _validate_tooltips,
)
from cellestial.util.errors import UnsupportedDataTypeError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import PlotSpec


def xyplot(
    data: AnnData,
    x: str,
    y: str,
    *,
    mapping: FeatureSpec | None = None,
    axis: Literal[0, 1] | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    observations_name: str = "Barcode",
    variables_name: str = "Variable",
    include_dimensions: bool | int = False,
    **point_kwargs,
) -> PlotSpec:
    """
    Scatter Plot.

    Parameters
    ----------
    data : AnnData
        The AnnData object of the single cell data.
    x : str
        The key for the x-axis.
    y : str
        The key for the y-axis.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    axis : {0,1} | None, default=None
        axis of the data, 0 for observations and 1 for variables.
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
    **point_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Scatter plot.

    Examples
    --------
    xyplot requires x and y values to be provided directly.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        cl.xyplot(
            data,
            x="n_genes_by_counts",
            y="pct_counts_in_top_50_genes",
            color="grey",
            alpha=0.6,
        )

    xyplot also supports customization of aesthetics via mapping.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        cl.xyplot(
            data,
            x="n_genes_by_counts",
            y="pct_counts_in_top_50_genes",
            mapping=aes(color="cell_type_lvl1"),
            alpha=0.6,
        ) + scale_color_viridis()

    Faceting by categorical data is also possible.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        cl.xyplot(
            data,
            x="n_genes_by_counts",
            y="pct_counts_in_top_50_genes",
            mapping=aes(color="cell_type_lvl1"),
            alpha=0.6,
        ) + scale_color_viridis() + facet_wrap("cell_type_lvl1")


    """
    # Handling Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise UnsupportedDataTypeError(msg)

    # HANDLE: mapping
    _mapping = aes(x=x, y=y)
    if mapping is None:
        mapping = _mapping
    elif isinstance(mapping, FeatureSpec):
        merged = _mapping.as_dict() | mapping.as_dict()
        mapping = aes(**merged)

    # Determine variable keys
    keys = [key for key in mapping.as_dict().values() if key is not None]
    variable_keys = _select_variable_keys(data=data, keys=keys)
    # include dimensions if a dimensional key provided
    if not include_dimensions:
        _keys = keys.copy()
        for key in keys:
            if key.startswith("X_"):
                include_dimensions = True
                _keys.remove(key)

    # HANDLE: tooltips
    tooltips = _resolve_tooltips(
        tooltips,
        data=data,
        variable_keys=variable_keys,
        defaults=keys,
    )

    # BUILD: the dataframe
    axis = _determine_axis(data=data, keys=_keys) if axis is None else axis
    frame = build_frame(
        data=data,
        variable_keys=variable_keys,
        axis=axis,
        observations_name=observations_name,
        variables_name=variables_name,
        include_dimensions=include_dimensions,
    )
    _validate_tooltips(tooltips, frame)

    # BUILD: the scatterplot
    scttr = (
        ggplot(data=frame)
        + geom_point(
            mapping=mapping,
            tooltips=tooltips,
            **point_kwargs,
        )
        + labs(x=x, y=y)
        + _THEME_SCATTER
    )
    if interactive:
        scttr += ggtb(size_zoomin=-1)

    return scttr
