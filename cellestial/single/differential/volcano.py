from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from anndata import AnnData
from lets_plot import (
    aes,
    geom_hline,
    geom_point,
    geom_text_repel,
    geom_vline,
    gggrid,
    ggplot,
    ggtb,
    labs,
    layer_tooltips,
    scale_color_manual,
)
from lets_plot.plot.core import FeatureSpec, LayerSpec

from cellestial.themes import _THEME_SCATTER_BASE
from cellestial.util import _share_axis, _share_labels

# Default volcano palette (matches the conventional EnhancedVolcano-style look):
# brick-red for up, soft cornflower-blue for down, medium-gray for non-significant.

if TYPE_CHECKING:
    from lets_plot.plot.core import PlotSpec
    from polars import DataFrame


def _build_volcano_frame(
    data: AnnData,
    group: str,
    *,
    group_by: str | None = None,
    key: str = "rank_genes_groups",
    use_adjusted_pvalue: bool = True,
    logfoldchange_threshold: float = 1.0,
    pvalue_threshold: float = 0.05,
    variable_column: str = "variable",
    logfoldchange_column: str = "logfoldchange",
    pvalue_column: str = "pvalue",
    neg_log_pvalue_column: str = "neg_log_pvalue",
    significance_column: str = "significance",
    up_label: str = "up",
    down_label: str = "down",
    nonsignificant_label: str = "ns",
    rank_genes_kwargs: dict | None = None,
) -> DataFrame:
    """
    Build a volcano-plot-ready frame.

    Extracts differential expression results for ``group`` and adds derived
    columns ``-log10(pvalue)`` and a categorical significance label
    (up / down / ns) based on the supplied thresholds.

    Parameters
    ----------
    data : AnnData
        The AnnData object holding the differential expression results
        in ``data.uns[key]`` (scanpy's ``rank_genes_groups`` convention).
    group : str
        The group name to extract results for.
    group_by : str | None, default=None
        Observation column to group by. When provided, ``data.uns[key]`` is
        (re)computed via ``scanpy.tl.rank_genes_groups`` if it is missing or
        was computed with a different ``groupby``.
    key : str, default='rank_genes_groups'
        The key under ``data.uns`` storing the differential expression results.
    use_adjusted_pvalue : bool, default=True
        Whether to use adjusted p-values (``pvals_adj``) instead of raw
        p-values (``pvals``) for the significance call and transform.
    logfoldchange_threshold : float, default=1.0
        Absolute log fold change threshold used to label up/down regulated.
    pvalue_threshold : float, default=0.05
        P-value threshold used to label significance.
    variable_column : str, default='variable'
        Output column name for the gene/feature names.
    logfoldchange_column : str, default='logfoldchange'
        Output column name for the log fold changes.
    pvalue_column : str, default='pvalue'
        Output column name for the p-values.
    neg_log_pvalue_column : str, default='neg_log_pvalue'
        Output column name for the ``-log10(pvalue)`` transform.
    significance_column : str, default='significance'
        Output column name for the categorical significance label.
    up_label : str, default='up'
        Label for significantly up regulated features.
    down_label : str, default='down'
        Label for significantly down regulated features.
    nonsignificant_label : str, default='ns'
        Label for non-significant features.
    rank_genes_kwargs : dict | None, default=None
        Additional keyword arguments forwarded to ``scanpy.tl.rank_genes_groups``
        when ``group_by`` triggers a (re)compute (e.g. ``method='t-test'``,
        ``use_raw=False``, ``layer=...``, ``corr_method='bonferroni'``).

    Returns
    -------
    DataFrame
        A polars DataFrame with one row per feature and the columns named
        according to the ``*_column`` parameters above.
    """
    # HANDLE: Data types
    if isinstance(data, AnnData):
        # COMPUTE: rank_genes_groups if requested and missing/stale
        if group_by is not None:
            existing_group_by = data.uns.get(key, {}).get("params", {}).get("groupby")
            if key not in data.uns or existing_group_by != group_by:
                import scanpy as sc

                sc.tl.rank_genes_groups(
                    data,
                    groupby=group_by,
                    key_added=key,
                    **(rank_genes_kwargs or {}),
                )
        # CHECK: differential expression results exist
        if key not in data.uns:
            msg = (
                f"`{key}` not found in `data.uns`. "
                "Run `scanpy.tl.rank_genes_groups` first or pass `group_by`."
            )
            raise KeyError(msg)
        results = data.uns[key]
        # CHECK: group exists in the results
        available_groups = list(results["names"].dtype.names)
        if group not in available_groups:
            msg = f"Group `{group}` not found. Available groups: {available_groups}"
            raise KeyError(msg)
        # SELECT: which p-value field to pull
        pvalue_field = "pvals_adj" if use_adjusted_pvalue else "pvals"
        # EXTRACT: per-group columns from the recarrays
        frame = pl.DataFrame(
            {
                variable_column: list(results["names"][group]),
                logfoldchange_column: np.asarray(
                    results["logfoldchanges"][group], dtype=np.float64
                ),
                pvalue_column: np.asarray(results[pvalue_field][group], dtype=np.float64),
            }
        )
    else:
        msg = f"Unsupported data type: `{type(data)}`"
        raise TypeError(msg)

    # CRITICAL PARTS: Dataframe Operations
    # 1. -log10(pvalue) transform
    frame = frame.with_columns(
        pl.col(pvalue_column).log10().neg().alias(neg_log_pvalue_column),
    )
    # 2. Significance label based on log fold change and p-value thresholds
    frame = frame.with_columns(
        pl.when(
            (pl.col(pvalue_column) < pvalue_threshold)
            & (pl.col(logfoldchange_column) >= logfoldchange_threshold)
        )
        .then(pl.lit(up_label))
        .when(
            (pl.col(pvalue_column) < pvalue_threshold)
            & (pl.col(logfoldchange_column) <= -logfoldchange_threshold)
        )
        .then(pl.lit(down_label))
        .otherwise(pl.lit(nonsignificant_label))
        .cast(pl.Categorical)
        .alias(significance_column),
    )
    # 3. Round float columns to keep the embedded JSON compact when the frame
    #    is serialized into the plot output (significant on-disk size win).
    frame = frame.with_columns(
        pl.col(logfoldchange_column).round(4),
        pl.col(neg_log_pvalue_column).round(4),
    )

    return frame

# AI-GENERATED: Claude 4.7
# VERIFIED: behavior
# UNAUDITED: not reviewed line-by-line, edge cases unverified
def volcano(
    data: AnnData,
    group: str,
    *,
    group_by: str | None = None,
    key: str = "rank_genes_groups",
    use_adjusted_pvalue: bool = True,
    logfoldchange_threshold: float = 1.0,
    pvalue_threshold: float = 0.05,
    mapping: FeatureSpec | None = None,
    color_up: str = "#b22222",
    color_down: str = "#6495ed",
    color_nonsignificant: str = "#bebebe",
    size: float = 2.0,
    alpha: float = 0.8,
    show_threshold_lines: bool = True,
    threshold_color: str = "#3f3f3f",
    threshold_size: float = 0.4,
    threshold_linetype: str = "dashed",
    threshold_kwargs: dict | None = None,
    top_n: int | None = 10,
    label_color: str = "#1f1f1f",
    label_size: float = 4.0,
    segment_size: float = 0.4,
    label_kwargs: dict | None = None,
    nonsignificant_subsample: int | None = 2000,
    variable_column: str = "variable",
    logfoldchange_column: str = "logfoldchange",
    pvalue_column: str = "pvalue",
    neg_log_pvalue_column: str = "neg_log_pvalue",
    significance_column: str = "significance",
    up_label: str = "up",
    down_label: str = "down",
    nonsignificant_label: str = "ns",
    rank_genes_kwargs: dict | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    **point_kwargs,
) -> PlotSpec:
    """
    Volcano plot.

    Plots ``-log10(pvalue)`` against log fold change for the differential
    expression results of ``group`` (vs the rest), colored by significance and
    optionally annotated with the top up- and down-regulated genes.

    Parameters
    ----------
    data : AnnData
        The AnnData object holding the differential expression results
        in ``data.uns[key]`` (scanpy's ``rank_genes_groups`` convention).
    group : str
        The group name to plot results for.
    group_by : str | None, default=None
        Observation column to group by. When provided, ``data.uns[key]`` is
        (re)computed via ``scanpy.tl.rank_genes_groups`` if it is missing or
        was computed with a different ``groupby``.
    key : str, default='rank_genes_groups'
        The key under ``data.uns`` storing the differential expression results.
    use_adjusted_pvalue : bool, default=True
        Whether to use adjusted p-values (``pvals_adj``) instead of raw
        p-values (``pvals``) for the significance call and transform.
    logfoldchange_threshold : float, default=1.0
        Absolute log fold change threshold used to label up/down regulated.
    pvalue_threshold : float, default=0.05
        P-value threshold used to label significance.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings, the result of `aes()`. Merged on top of
        the default ``aes(x=logfoldchange, y=neg_log_pvalue, color=significance)``.
    color_up : str, default='#A83737'
        Color for significantly up-regulated points (brick red).
    color_down : str, default='#6FA8DC'
        Color for significantly down-regulated points (soft cornflower blue).
    color_nonsignificant : str, default='#A6A6A6'
        Color for non-significant points (medium gray).
    size : float, default=2.0
        Default point size; can be overridden via ``point_kwargs``.
    alpha : float, default=0.8
        Default point alpha; can be overridden via ``point_kwargs``.
    show_threshold_lines : bool, default=True
        Whether to draw the dashed threshold lines.
    threshold_color : str, default='#3f3f3f'
        Color of the threshold lines.
    threshold_size : float, default=0.4
        Size (thickness) of the threshold lines.
    threshold_linetype : str, default='dashed'
        Line type of the threshold lines.
    threshold_kwargs : dict | None, default=None
        Additional parameters passed to the threshold ``geom_hline`` and
        ``geom_vline`` layers.
    top_n : int | None, default=10
        Number of top up- and top down-regulated genes (by ``-log10(pvalue)``)
        to label. Set to ``None`` or ``0`` to disable labels.
    label_color : str, default='#1f1f1f'
        Color of the gene labels.
    label_size : float, default=4.0
        Size of the gene labels.
    segment_size : float, default=0.4
        Width of the line segment connecting the label to the point.
    label_kwargs : dict | None, default=None
        Additional parameters passed to the ``geom_text`` label layer.
    nonsignificant_subsample : int | None, default=2000
        Maximum number of non-significant points to keep. Significant points
        (up/down) are always kept in full. Reducing this shrinks the embedded
        data in the rendered plot (smaller HTML/notebook output) at no visible
        cost since the non-significant cloud is heavily overplotted. Set to
        ``None`` to keep every non-significant point. Sampling is deterministic
        (``seed=42``).
    variable_column : str, default='variable'
        Output column name for the gene/feature names.
    logfoldchange_column : str, default='logfoldchange'
        Output column name for the log fold changes.
    pvalue_column : str, default='pvalue'
        Output column name for the p-values.
    neg_log_pvalue_column : str, default='neg_log_pvalue'
        Output column name for the ``-log10(pvalue)`` transform.
    significance_column : str, default='significance'
        Output column name for the categorical significance label.
    up_label : str, default='up'
        Label for significantly up regulated features.
    down_label : str, default='down'
        Label for significantly down regulated features.
    nonsignificant_label : str, default='ns'
        Label for non-significant features.
    rank_genes_kwargs : dict | None, default=None
        Additional keyword arguments forwarded to ``scanpy.tl.rank_genes_groups``
        when ``group_by`` triggers a (re)compute (e.g. ``method='t-test'``,
        ``use_raw=False``, ``layer=...``, ``corr_method='bonferroni'``).
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
        Accepts Sequence[str] or result of `layer_tooltips()` for more complex tooltips.
        Use 'none' to disable tooltips.
    interactive : bool, default=False
        Whether to make the plot interactive.
    **point_kwargs
        Additional parameters for the ``geom_point`` layer.

    Returns
    -------
    PlotSpec
        Volcano plot.

    Examples
    --------
    A simple volcano plot.

    .. jupyter-execute::

        import scanpy as sc

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        cl.volcano(
            data,
            group="B Cells",
            group_by="cell_type_lvl1",
        )
    """
    # HANDLE: Data types
    if not isinstance(data, AnnData):
        msg = "data must be an `AnnData` object"
        raise TypeError(msg)

    # BUILD: dataframe via the helper
    frame = _build_volcano_frame(
        data=data,
        group=group,
        group_by=group_by,
        key=key,
        use_adjusted_pvalue=use_adjusted_pvalue,
        logfoldchange_threshold=logfoldchange_threshold,
        pvalue_threshold=pvalue_threshold,
        variable_column=variable_column,
        logfoldchange_column=logfoldchange_column,
        pvalue_column=pvalue_column,
        neg_log_pvalue_column=neg_log_pvalue_column,
        significance_column=significance_column,
        up_label=up_label,
        down_label=down_label,
        nonsignificant_label=nonsignificant_label,
        rank_genes_kwargs=rank_genes_kwargs,
    )
    # DROP: rows where -log10(pvalue) is non-finite (e.g. pvalue == 0 → +inf)
    frame = frame.filter(
        pl.col(neg_log_pvalue_column).is_finite() & pl.col(logfoldchange_column).is_finite()
    )

    # SUBSAMPLE: cap non-significant points to keep the embedded data small.
    # The non-significant cloud overplots itself heavily, so dropping most of
    # it has no visible effect but shrinks the rendered plot's on-disk size.
    if nonsignificant_subsample is not None:
        nonsignificant = frame.filter(pl.col(significance_column) == nonsignificant_label)
        if nonsignificant.height > nonsignificant_subsample:
            significant = frame.filter(pl.col(significance_column) != nonsignificant_label)
            nonsignificant = nonsignificant.sample(n=nonsignificant_subsample, seed=42)
            frame = pl.concat([significant, nonsignificant])

    # HANDLE: tooltips
    default_tooltip_columns = [
        variable_column,
        logfoldchange_column,
        pvalue_column,
        significance_column,
    ]
    if tooltips is None:
        tooltips_spec = layer_tooltips(default_tooltip_columns)
    elif tooltips == "none" or isinstance(tooltips, str):
        tooltips_spec = tooltips
    elif isinstance(tooltips, Sequence):
        tooltips = list(tooltips)
        if not set(tooltips).issubset(frame.columns):
            missing = set(tooltips) - set(frame.columns)
            msg = f"Some tooltip columns are not in the data: {missing}"
            raise ValueError(msg)
        tooltips_spec = layer_tooltips(tooltips)
    elif isinstance(tooltips, FeatureSpec):
        tooltips_spec = tooltips

    # DEFINE: mapping with defaults
    _mapping = {
        "x": logfoldchange_column,
        "y": neg_log_pvalue_column,
        "color": significance_column,
    }
    if mapping is not None:
        _mapping.update(mapping.as_dict())

    # BUILD: volcano
    vlcn = ggplot(frame) + geom_point(
        aes(**_mapping),
        tooltips=tooltips_spec,
        size=size,
        alpha=alpha,
        **point_kwargs,
    )

    # ADD: threshold lines
    if show_threshold_lines:
        _threshold_kwargs = {
            "color": threshold_color,
            "size": threshold_size,
            "linetype": threshold_linetype,
        } | (threshold_kwargs or {})
        vlcn += geom_hline(yintercept=-np.log10(pvalue_threshold), **_threshold_kwargs)
        vlcn += geom_vline(xintercept=logfoldchange_threshold, **_threshold_kwargs)
        vlcn += geom_vline(xintercept=-logfoldchange_threshold, **_threshold_kwargs)

    # ADD: top-N gene labels
    if top_n:
        significant = frame.filter(pl.col(significance_column) != nonsignificant_label)
        top_up = (
            significant.filter(pl.col(significance_column) == up_label)
            .sort(neg_log_pvalue_column, descending=True)
            .head(top_n)
        )
        top_down = (
            significant.filter(pl.col(significance_column) == down_label)
            .sort(neg_log_pvalue_column, descending=True)
            .head(top_n)
        )
        label_frame = pl.concat([top_up, top_down])
        if label_frame.height > 0:
            _label_kwargs = {
                "color": label_color,
                "size": label_size,
                "segment_size": segment_size,
                "point_padding": 0.3,
                "box_padding": 0.4,
                "max_overlaps": 50,
                "seed": 42,
            } | (label_kwargs or {})
            vlcn += geom_text_repel(
                data=label_frame,
                mapping=aes(
                    x=logfoldchange_column,
                    y=neg_log_pvalue_column,
                    label=variable_column,
                ),
                inherit_aes=False,
                **_label_kwargs,
            )

    # ADD: labels and theme
    vlcn += labs(
        x="logFC",
        y="-log10(Pvalue)",
        color=significance_column,
    )
    vlcn += (
        scale_color_manual(
            values={
                up_label: color_up,
                down_label: color_down,
                nonsignificant_label: color_nonsignificant,
            }
        )
        + _THEME_SCATTER_BASE
    )

    # HANDLE: interactive
    if interactive:
        vlcn += ggtb(size_zoomin=-1)

    return vlcn


def volcanos(
    data: AnnData,
    groups: Sequence[str],
    *,
    group_by: str | None = None,
    key: str = "rank_genes_groups",
    use_adjusted_pvalue: bool = True,
    logfoldchange_threshold: float = 1.0,
    pvalue_threshold: float = 0.05,
    mapping: FeatureSpec | None = None,
    color_up: str = "#b22222",
    color_down: str = "#6495ed",
    color_nonsignificant: str = "#bebebe",
    size: float = 2.0,
    alpha: float = 0.8,
    show_threshold_lines: bool = True,
    threshold_color: str = "#3f3f3f",
    threshold_size: float = 0.4,
    threshold_linetype: str = "dashed",
    threshold_kwargs: dict | None = None,
    top_n: int | None = 10,
    label_color: str = "#1f1f1f",
    label_size: float = 4.0,
    segment_size: float = 0.4,
    label_kwargs: dict | None = None,
    nonsignificant_subsample: int | None = 2000,
    variable_column: str = "variable",
    logfoldchange_column: str = "logfoldchange",
    pvalue_column: str = "pvalue",
    neg_log_pvalue_column: str = "neg_log_pvalue",
    significance_column: str = "significance",
    up_label: str = "up",
    down_label: str = "down",
    nonsignificant_label: str = "ns",
    rank_genes_kwargs: dict | None = None,
    tooltips: Literal["none"] | Sequence[str] | FeatureSpec | None = None,
    interactive: bool = False,
    # multi plot args
    share_labels: bool = False,
    share_axis: bool = False,
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
):
    """
    Grid of volcano plots, one per group.

    Builds a volcano plot for each entry in ``groups`` (vs the rest) and
    arranges them in a grid via ``gggrid``. All single-plot styling and
    threshold options behave the same as in :func:`volcano`.

    Parameters
    ----------
    data : AnnData
        The AnnData object holding the differential expression results
        in ``data.uns[key]`` (scanpy's ``rank_genes_groups`` convention).
    groups : Sequence[str]
        The group names to plot results for. One subplot per group.
    group_by : str | None, default=None
        Observation column to group by. When provided, ``data.uns[key]`` is
        (re)computed via ``scanpy.tl.rank_genes_groups`` if it is missing or
        was computed with a different ``groupby``. Computed once on the first
        subplot and reused thereafter.
    key : str, default='rank_genes_groups'
        The key under ``data.uns`` storing the differential expression results.
    use_adjusted_pvalue : bool, default=True
        Whether to use adjusted p-values (``pvals_adj``) instead of raw p-values.
    logfoldchange_threshold : float, default=1.0
        Absolute log fold change threshold used to label up/down regulated.
    pvalue_threshold : float, default=0.05
        P-value threshold used to label significance.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings, the result of `aes()`.
    color_up : str, default='#b22222'
        Color for significantly up-regulated points.
    color_down : str, default='#6495ed'
        Color for significantly down-regulated points.
    color_nonsignificant : str, default='#bebebe'
        Color for non-significant points.
    size : float, default=2.0
        Default point size; can be overridden via ``point_kwargs``.
    alpha : float, default=0.8
        Default point alpha; can be overridden via ``point_kwargs``.
    show_threshold_lines : bool, default=True
        Whether to draw the dashed threshold lines.
    threshold_color : str, default='#3f3f3f'
        Color of the threshold lines.
    threshold_size : float, default=0.4
        Size (thickness) of the threshold lines.
    threshold_linetype : str, default='dashed'
        Line type of the threshold lines.
    threshold_kwargs : dict | None, default=None
        Additional parameters passed to the threshold ``geom_hline`` and ``geom_vline`` layers.
    top_n : int | None, default=10
        Number of top up- and top down-regulated genes (by ``-log10(pvalue)``)
        to label per subplot. Set to ``None`` or ``0`` to disable labels.
    label_color : str, default='#1f1f1f'
        Color of the gene labels.
    label_size : float, default=4.0
        Size of the gene labels.
    segment_size : float, default=0.4
        Width of the line segment connecting the label to the point.
    label_kwargs : dict | None, default=None
        Additional parameters passed to the ``geom_text_repel`` label layer.
    nonsignificant_subsample : int | None, default=2000
        Maximum number of non-significant points to keep per subplot. Significant
        points (up/down) are always kept in full. Reducing this shrinks the
        embedded data in the rendered grid (smaller HTML/notebook output) at no
        visible cost since the non-significant cloud is heavily overplotted. Set
        to ``None`` to keep every non-significant point. Sampling is deterministic
        (``seed=42``).
    variable_column : str, default='variable'
        Output column name for the gene/feature names.
    logfoldchange_column : str, default='logfoldchange'
        Output column name for the log fold changes.
    pvalue_column : str, default='pvalue'
        Output column name for the p-values.
    neg_log_pvalue_column : str, default='neg_log_pvalue'
        Output column name for the ``-log10(pvalue)`` transform.
    significance_column : str, default='significance'
        Output column name for the categorical significance label.
    up_label : str, default='up'
        Label for significantly up regulated features.
    down_label : str, default='down'
        Label for significantly down regulated features.
    nonsignificant_label : str, default='ns'
        Label for non-significant features.
    rank_genes_kwargs : dict | None, default=None
        Additional keyword arguments forwarded to ``scanpy.tl.rank_genes_groups``
        when ``group_by`` triggers a (re)compute.
    tooltips: {'none'} | Sequence[str] | FeatureSpec | None, default=None
        Tooltips to show when hovering over the geom.
    interactive : bool, default=False
        Whether to make the plot interactive.
    share_labels : bool, default=False
        Whether to share axis labels across the grid.
        If True, only X labels on bottom row and Y labels on left column are shown.
    share_axis : bool, default=False
        Whether to share axes across the grid.
        If True, only X axis on bottom row and Y axis on left column is shown.
    layers : Sequence[FeatureSpec|LayerSpec] | FeatureSpec | LayerSpec | None, default=None
        Layers to add to all the plots in the grid.
    ncol : int, default=None
        Number of columns in grid. If not specified, shows plots horizontally, in one row.
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
    **point_kwargs
        Additional parameters for the ``geom_point`` layer of each subplot.

    Returns
    -------
    SupPlotsSpec
        Grid of volcano plots.

    Examples
    --------
    .. jupyter-execute::

        import scanpy as sc

        import cellestial as cl

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        cl.volcanos(
            data,
            groups=["B Cells", "Monocytes", "Erythroid", "Lymphocytes"],
            group_by="cell_type_lvl1",
            ncol=2,
        )
    """
    if not isinstance(groups, Sequence) or isinstance(groups, str):
        msg = "groups must be a Sequence of strings"
        raise TypeError(msg)

    plots = []
    for i, group in enumerate(groups):
        plot = volcano(
            data=data,
            group=group,
            group_by=group_by,
            key=key,
            use_adjusted_pvalue=use_adjusted_pvalue,
            logfoldchange_threshold=logfoldchange_threshold,
            pvalue_threshold=pvalue_threshold,
            mapping=mapping,
            color_up=color_up,
            color_down=color_down,
            color_nonsignificant=color_nonsignificant,
            size=size,
            alpha=alpha,
            show_threshold_lines=show_threshold_lines,
            threshold_color=threshold_color,
            threshold_size=threshold_size,
            threshold_linetype=threshold_linetype,
            threshold_kwargs=threshold_kwargs,
            top_n=top_n,
            label_color=label_color,
            label_size=label_size,
            segment_size=segment_size,
            label_kwargs=label_kwargs,
            nonsignificant_subsample=nonsignificant_subsample,
            variable_column=variable_column,
            logfoldchange_column=logfoldchange_column,
            pvalue_column=pvalue_column,
            neg_log_pvalue_column=neg_log_pvalue_column,
            significance_column=significance_column,
            up_label=up_label,
            down_label=down_label,
            nonsignificant_label=nonsignificant_label,
            rank_genes_kwargs=rank_genes_kwargs,
            tooltips=tooltips,
            interactive=interactive,
            **point_kwargs,
        )

        # handle the layers
        if layers is not None:
            if isinstance(layers, (FeatureSpec, LayerSpec)):
                layers = [layers]
            for layer in layers:
                plot += layer
        if share_labels:
            plot = _share_labels(plot, i, groups, ncol)
        if share_axis:
            plot = _share_axis(plot, i, groups, ncol, "axis")

        plots.append(plot)

    return gggrid(
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
