from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_label,
    geom_line,
    geom_point,
    ggplot,
    xlab,
    ylab,
)

from cellestial.frames import _pca_variance_frame
from cellestial.themes import _THEME_SCATTER_BASE
from cellestial.util.errors import _unsupported_data_type

if TYPE_CHECKING:
    from lets_plot.plot.core import FeatureSpec, PlotSpec


def elbow(
    data: AnnData,
    n_pcs: int | None = None,
    *,
    mapping: FeatureSpec | None = None,
    component_column: str = "Principal Component",
    variance_column: str = "Variance Ratio",
    color: str = "#1f1f1f",
    size: float = 4.0,
    line_size: float = 1,
    linetype: str = "dashed",
    label: bool = False,
    label_size: float = 4.0,
    label_every: int = 5,
    pca_key: str = "pca",
    **geom_kwargs,
) -> PlotSpec:
    """
    Elbow Plot.

    Parameters
    ----------
    data : AnnData
        The single-cell data object.
        Must have PCA results computed beforehand.
    n_pcs : int | None, default=None
        Number of principal components to display.
        If None, shows all available components.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the plot, the result of `aes()`.
    component_column : str, default='Principal Component'
        The name to give to the component (x-axis) column in the dataframe.
    variance_column : str, default='Variance Ratio'
        The name to give to the variance ratio (y-axis) column in the dataframe.
    color : str, default='#1f1f1f'
        Color of the points and connecting line.
    size : float, default=4.0
        Size of the points.
    line_size : float, default=1
        Size of the connecting line.
    linetype : str, default='dashed'
        Linetype of the connecting line (e.g., 'solid', 'dashed', 'dotted').
    label : bool, default=False
        If True, draw a vertical `PC{n}` label on each selected component using `geom_label`.
    label_size : float, default=4.0
        Text size for the component labels (used when `label=True`).
    label_every : int, default=5
        Only components whose number is a multiple of this value are labeled
        (used when `label=True`). For example, `5` labels PC5, PC10, PC15, ...
    pca_key : str, default='pca'
        Specific key holding the PCA results.
    **geom_kwargs
        Additional parameters for the `geom_point` layer.
        For more information on geom_point parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_point.html

    Returns
    -------
    PlotSpec
        Elbow plot.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    ValueError
        If PCA results are not available on `data`,
        or if `n_pcs` exceeds the number of available components.

    Notes
    -----
    Plots the variance ratio explained by each principal component, helping
    identify how many components to retain.

    Examples
    --------
    A simple elbow plot.

    .. jupyter-execute::

        import scanpy as sc
        from lets_plot import *

        import cellestial as cl

        data = cl.datasets.pbmc3k()
        sc.tl.pca(data)

        cl.elbow(data)

    Limit components and customize.

    .. jupyter-execute::

        cl.elbow(data, n_pcs=20, color="#0f4f8f", size=4)
    """
    # Handling Data types
    if not isinstance(data, AnnData):
        raise _unsupported_data_type(data, AnnData)

    # BUILD: variance ratio frame
    frame = _pca_variance_frame(
        data,
        n_pcs=n_pcs,
        component_column=component_column,
        variance_column=variance_column,
        use_key=pca_key,
    )

    # HANDLE: mapping
    defaults = aes(x=component_column, y=variance_column).as_dict()
    if mapping is not None:
        defaults.update(mapping.as_dict())
    mapping = aes(**defaults)

    # BUILD: elbow plot
    plot = ggplot(frame, mapping=mapping)
    plot += geom_line(color=color, size=line_size, linetype=linetype)
    plot += geom_point(color=color, size=size, **geom_kwargs)

    # ADD: per-component labels (PC{label_every}, PC{2*label_every}, ...)
    if label and label_every > 0:
        label_column = "PC"
        labeled_frame = frame.filter(
            pl.col(component_column) % label_every == 0,
        ).with_columns(
            (pl.lit("PC") + pl.col(component_column).cast(pl.Utf8)).alias(label_column),
        )
        plot += geom_label(
            mapping=aes(label=label_column),
            data=labeled_frame,
            size=label_size,
            direction="both",
            min_segment_length=0,
            point_padding=0.3,
            box_padding=0.4,
            color=color,
            angle=90,
        )

    return plot + _THEME_SCATTER_BASE + xlab(component_column) + ylab(variance_column)
