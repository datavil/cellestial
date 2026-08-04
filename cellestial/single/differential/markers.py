from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    element_text,
    geom_path,
    geom_text,
    gggrid,
    ggplot,
    ggtb,
    ggtitle,
    labs,
    scale_color_gradient,
    scale_y_continuous,
    theme,
    theme_bw,
)
from lets_plot.plot.core import FeatureSpec, LayerSpec
from mudata import MuData

from cellestial.single.differential.utilities import _build_markers_frame
from cellestial.util import _share_axis, _share_labels
from cellestial.util.errors import _unsupported_data_type
from cellestial.util.utilities import _container

if TYPE_CHECKING:
    from lets_plot.plot.subplots import SupPlotsSpec


def markers(
    data: AnnData | MuData,
    groups: Sequence[str] | None = None,
    *,
    modality: str | None = None,
    key: str = "rank_genes_groups",
    n_genes: int = 20,
    mapping: FeatureSpec | None = None,
    text_color: str = "#1f1f1f",
    text_size: float = 4.0,
    fontface: str = "bold",
    angle: float = 90.0,
    rank_color: bool = False,
    line: bool = False,
    line_color: str = "#3f3f3f",
    line_size: float = 0.4,
    line_alpha: float = 0.6,
    line_kwargs: dict | None = None,
    variable_column: str = "variable",
    score_column: str = "score",
    rank_column: str = "rank",
    group_column: str = "group",
    interactive: bool = False,
    # multi plot args
    share_labels: bool = True,
    share_axis: bool = False,
    layers: Sequence[FeatureSpec | LayerSpec] | FeatureSpec | LayerSpec | None = None,
    # grid args
    ncol: int | None = 3,
    sharex: str | None = None,
    sharey: str | None = None,
    widths: list[float] | None = None,
    heights: list[float] | None = None,
    hspace: float | None = None,
    vspace: float | None = None,
    fit: bool | None = None,
    align: bool | None = None,
    guides: str = "auto",
    **text_kwargs,
) -> SupPlotsSpec:
    """
    Grid of ranked genes per group.

    Parameters
    ----------
    data : AnnData
        The single-cell data object holding the precomputed differential
        expression ranking.
    groups : Sequence[str] | None, default=None
        Subset of groups to plot, one panel per group. `None` keeps all
        groups in their stored order.
    modality : str | None, default=None
        Which modality's stored analysis results to use. Required for a
        multimodal object holding more than one modality, and not accepted
        otherwise.
    key : str, default='rank_genes_groups'
        The key under which the precomputed ranking is stored on `data`.
    n_genes : int, default=20
        Number of top genes to show per panel.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings, the result of `aes()`. Merged on top of
        the default `aes(x=rank, y=score, label=variable)`.
    text_color : str, default='#1f1f1f'
        Color of the gene-name text.
    text_size : float, default=4.0
        Size of the gene-name text.
    fontface : str, default='bold'
        Font face of the gene-name text (e.g. 'plain', 'bold', 'italic').
    angle : float, default=90.0
        Rotation angle of the gene-name text, in degrees.
    rank_color : bool, default=False
        Whether to color the gene-name text by rank. When True, the best-ranked
        gene is pure red, fading toward light gray for the lowest-ranked.
    line : bool, default=False
        Whether to draw a dashed path connecting the genes' `(rank, score)`
        points, showing the score decay across the ranking.
    line_color : str, default='#3f3f3f'
        Color of the dashed score-curve line.
    line_size : float, default=0.4
        Size (thickness) of the dashed score-curve line.
    line_alpha : float, default=0.6
        Alpha (opacity) of the dashed score-curve line.
    line_kwargs : dict | None, default=None
        Additional parameters passed to the score-curve `geom_path` layer.
    variable_column : str, default='variable'
        Output column name for the gene/feature names.
    score_column : str, default='score'
        Output column name for the ranking scores.
    rank_column : str, default='rank'
        Output column name for the per-group rank index (0-based).
    group_column : str, default='group'
        Output column name for the group label.
    interactive : bool, default=False
        Whether to make the plot interactive.
    share_labels : bool, default=True
        Whether to share axis labels across the grid.
        If True, only X labels on bottom row and Y labels on left column are shown.
    share_axis : bool, default=False
        Whether to share axes across the grid.
        If True, only X axis on bottom row and Y axis on left column is shown.
    layers : Sequence[FeatureSpec|LayerSpec] | FeatureSpec | LayerSpec | None, default=None
        Layers to add to all the plots in the grid.
    ncol : int, default=3
        Number of columns in grid. If None, shows plots horizontally, in one row.
    sharex, sharey : str | None, default=None
        Controls sharing of axis limits between subplots in the grid.
        See :func:`lets_plot.gggrid`.
    widths : list[float], default=None
        Relative width of each column of the grid.
    heights : list[float], default=None
        Relative height of each row of the grid.
    hspace, vspace : float | None, default=None
        Cell horizontal/vertical spacing in px.
    fit : bool, default=None
        Whether to stretch each plot to match the aspect ratio of its cell.
    align : bool, default=None
        Whether to align inner areas of plots.
    guides : str, default='auto'
        How guides (legends/colorbars) should be treated in the layout.
        See :func:`lets_plot.gggrid`.
    **text_kwargs
        Additional parameters for the `geom_text` layer of each panel.

    Returns
    -------
    SupPlotsSpec
        Grid of ranked-genes panels, one per group.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If the ranking result or a requested group is missing.
    ValueError
        If `n_genes` is out of range.
    TypeError
        If `groups` is neither a Sequence of strings nor None.

    Notes
    -----
    Builds one panel per group from a precomputed ranking, placing the top
    `n_genes` gene names as text at their `(rank, score)` position. Each
    panel is titled `"{group} vs. rest"`.

    Examples
    --------
    A simple grid of ranked genes per group, with optional score-curve lines.

    .. jupyter-execute::

        import scanpy as sc

        import cellestial as cl

        data = sc.datasets.pbmc68k_reduced()

        cl.markers(data, line=True)

    Color the gene names by rank.

    .. jupyter-execute::

        cl.markers(data, rank_color=True)

    Select a subset of groups.

    .. jupyter-execute::

        cl.markers(
            data,
            groups=["CD14+ Monocyte", "CD4+/CD25 T Reg", "CD4+/CD45RO+ Memory", "CD34+"],
            ncol=2,
            rank_color=True,
        )

    """
    # HANDLE: Data types
    if not isinstance(data, (AnnData, MuData)):
        raise _unsupported_data_type(data, AnnData, MuData)
    # Stored analysis results live inside a single modality.
    data = _container(data).select_modality(modality)

    if groups is not None and (not isinstance(groups, Sequence) or isinstance(groups, str)):
        msg = "`groups` must be a Sequence of strings or None"
        raise TypeError(msg)

    # BUILD: long-form frame via the helper
    frame, selected_groups, _ = _build_markers_frame(
        data=data,
        key=key,
        n_genes=n_genes,
        groups=groups,
        variable_column=variable_column,
        score_column=score_column,
        rank_column=rank_column,
        group_column=group_column,
    )

    # DEFINE: mapping with defaults
    _mapping = {
        "x": rank_column,
        "y": score_column,
        "label": variable_column,
    }
    if rank_color:
        _mapping["color"] = rank_column
    if mapping is not None:
        _mapping.update(mapping.as_dict())

    # BUILD: one panel per group
    plots = []
    for i, group in enumerate(selected_groups):
        group_frame = frame.filter(pl.col(group_column) == group)
        plot = ggplot(group_frame)

        # ADD: optional dashed score-curve path (drawn under the text)
        if line:
            _line_kwargs = {
                "color": line_color,
                "size": line_size,
                "alpha": line_alpha,
                "linetype": "dashed",
            } | (line_kwargs or {})
            plot += geom_path(aes(x=rank_column, y=score_column), **_line_kwargs)

        # A constant `color=` would override the rank color aesthetic, so only
        # set it when not coloring by rank.
        _text_color = {} if rank_color else {"color": text_color}
        plot += geom_text(
            aes(**_mapping),
            size=text_size,
            fontface=fontface,
            angle=angle,
            **_text_color,
            **text_kwargs,
        )
        if rank_color:
            # rank 0 (best) is pure red, fading toward light gray for the rest.
            plot += scale_color_gradient(low="#ff0000", high="#e6e6e6")
        # Gene names run vertically (8-10 chars), so expand the y range to keep
        # the top- and bottom-most labels from being clipped.
        plot += scale_y_continuous(expand=[0.25, 0.0])
        plot += labs(x="ranking", y="score")
        plot += ggtitle(f"{group} vs. rest")
        plot += theme(
            plot_title=element_text(size=11, hjust=0.5, family="Georgia"),
            legend_position="none",
        )

        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        if share_labels:
            plot = _share_labels(plot, i, selected_groups, ncol)
        if share_axis:
            plot = _share_axis(plot, i, selected_groups, ncol, "axis")

        plots.append(plot)

    grid = (
        gggrid(
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
        + theme_bw()
    )

    if interactive:
        grid += ggtb(size_zoomin=-1)

    return grid
