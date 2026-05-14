from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
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

from cellestial.single.heatmap._rank_genes_groups import _resolve_rank_genes_groups_key
from cellestial.util import _share_axis, _share_labels
from cellestial.util.errors import KeyNotFoundError, UnsupportedDataTypeError

if TYPE_CHECKING:
    from lets_plot.plot.subplots import SupPlotsSpec
    from polars import DataFrame


def _build_markers_frame(
    data: AnnData,
    *,
    key: bool | str,
    n_genes: int,
    groups: Sequence[str] | None,
    variable_column: str,
    score_column: str,
    rank_column: str,
    group_column: str,
) -> tuple[DataFrame, list[str], str]:
    """
    Build a long-form frame of the top-N ranked genes and scores per group.

    Reads ``data.uns[key]`` (scanpy's ``rank_genes_groups`` convention) and
    pulls the top ``n_genes`` gene names and their scores for each selected
    group. Unlike the heatmap extractor, genes are kept per group with no
    cross-group de-duplication, since each group is drawn in its own panel.

    Parameters
    ----------
    data : AnnData
        Source object that already holds a ``rank_genes_groups`` result.
    key : bool | str
        ``True`` reads the default ``rank_genes_groups`` key; a string reads
        the matching custom key (e.g. ``"rank_genes_groups_wilcoxon"``).
    n_genes : int
        Number of top genes to pull per group.
    groups : Sequence[str] | None
        Subset of groups to include, in order. ``None`` keeps all groups in
        their stored order.
    variable_column : str
        Output column name for the gene/feature names.
    score_column : str
        Output column name for the ranking scores.
    rank_column : str
        Output column name for the per-group rank index (0-based).
    group_column : str
        Output column name for the group label.

    Returns
    -------
    frame : DataFrame
        Long-form polars DataFrame with one row per (group, gene).
    selected : list[str]
        The selected group labels, in plotting order.
    group_by : str
        The categorical key used when ``rank_genes_groups`` was computed.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If the ranking result, a requested group, or the stored ``groupby``
        is missing.
    ValueError
        If `n_genes` is out of range.
    """
    if n_genes < 1:
        msg = f"`n_genes` must be >= 1, got {n_genes}."
        raise ValueError(msg)

    uns_key = _resolve_rank_genes_groups_key(key)

    if isinstance(data, AnnData):
        if uns_key not in data.uns:
            msg = (
                f"`adata.uns[{uns_key!r}]` not found. "
                "Run `scanpy.tl.rank_genes_groups` first "
                "(or pass the correct `key_added` value as `key`)."
            )
            raise KeyNotFoundError(msg)

        record = data.uns[uns_key]
        if "names" not in record or "scores" not in record or "params" not in record:
            msg = (
                f"`adata.uns[{uns_key!r}]` does not look like a "
                "`rank_genes_groups` result (missing 'names', 'scores', or 'params')."
            )
            raise KeyNotFoundError(msg)

        names = record["names"]
        scores = record["scores"]
        available_groups = list(names.dtype.names)

        if groups is None:
            selected = available_groups
        else:
            selected = list(groups)
            unknown = [group for group in selected if group not in available_groups]
            if unknown:
                msg = (
                    f"Groups {unknown!r} not found in "
                    f"`adata.uns[{uns_key!r}]`. Available: {available_groups!r}."
                )
                raise KeyNotFoundError(msg)

        n_available = len(names)
        if n_genes > n_available:
            msg = (
                f"`n_genes={n_genes}` exceeds the {n_available} ranked genes "
                f"stored in `adata.uns[{uns_key!r}]`."
            )
            raise ValueError(msg)

        ranks = np.arange(n_genes)
        group_frames = [
            pl.DataFrame(
                {
                    group_column: group,
                    rank_column: ranks,
                    variable_column: [str(name) for name in names[group][:n_genes]],
                    score_column: np.asarray(scores[group][:n_genes], dtype=np.float64),
                }
            )
            for group in selected
        ]
        frame = pl.concat(group_frames)

        params = record["params"]
        stored_group_by = params["groupby"] if "groupby" in params else None
        if stored_group_by is None:
            msg = (
                f"`adata.uns[{uns_key!r}]['params']` is missing 'groupby'; "
                "cannot infer `group_by`."
            )
            raise KeyNotFoundError(msg)

        # Round scores to keep the embedded JSON compact when the frame is
        # serialized into the plot output.
        frame = frame.with_columns(pl.col(score_column).round(4))

        return frame, list(selected), str(stored_group_by)

    msg = f"Unsupported data type: `{type(data)}`"
    raise UnsupportedDataTypeError(msg)


def markers(
    data: AnnData,
    groups: Sequence[str] | None = None,
    *,
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

    Builds one panel per group from a precomputed ranking, placing the top
    ``n_genes`` gene names as text at their ``(rank, score)`` position. Each
    panel is titled ``"{group} vs. rest"``.

    Parameters
    ----------
    data : AnnData
        The AnnData object holding the differential expression results
        in ``data.uns[key]`` (scanpy's ``rank_genes_groups`` convention).
    groups : Sequence[str] | None, default=None
        Subset of groups to plot, one panel per group. ``None`` keeps all
        groups in their stored order.
    key : str, default='rank_genes_groups'
        The key under ``data.uns`` storing the differential expression results.
    n_genes : int, default=20
        Number of top genes to show per panel.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings, the result of `aes()`. Merged on top of
        the default ``aes(x=rank, y=score, label=variable)``.
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
        Whether to draw a dashed path connecting the genes' ``(rank, score)``
        points, showing the score decay across the ranking.
    line_color : str, default='#3f3f3f'
        Color of the dashed score-curve line.
    line_size : float, default=0.4
        Size (thickness) of the dashed score-curve line.
    line_alpha : float, default=0.6
        Alpha (opacity) of the dashed score-curve line.
    line_kwargs : dict | None, default=None
        Additional parameters passed to the score-curve ``geom_path`` layer.
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
        Additional parameters for the ``geom_text`` layer of each panel.

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
    if not isinstance(data, AnnData):
        msg = f"Unsupported data type: `{type(data)}`"
        raise UnsupportedDataTypeError(msg)

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

    grid = gggrid(
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
    ) + theme_bw()

    if interactive:
        grid += ggtb(size_zoomin=-1)

    return grid
