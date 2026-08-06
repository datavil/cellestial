from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from anndata import AnnData

from cellestial.single.heatmap.utilities import _resolve_rank_genes_groups_key
from cellestial.util.errors import KeyNotFoundError, _unsupported_data_type

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polars import DataFrame


def _resolve_pvalue_column(pvalue_column: str | None, *, use_adjusted_pvalue: bool) -> str:
    """Name the p-value column after the values it actually holds."""
    if pvalue_column is not None:
        return pvalue_column
    return "pvalue_adj" if use_adjusted_pvalue else "pvalue"


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
    pvalue_column: str | None = None,
    neg_log_pvalue_column: str = "neg_log_pvalue",
    significance_column: str = "significance",
    up_label: str = "up",
    down_label: str = "down",
    nonsignificant_label: str = "ns",
    rank_genes_kwargs: dict | None = None,
) -> DataFrame:
    """
    Build a volcano-plot-ready frame.

    Parameters
    ----------
    data : AnnData
        The single-cell data object holding a precomputed differential
        expression ranking.
    group : str
        The group name to extract results for.
    group_by : str | None, default=None
        Observation column to group by. When provided, the ranking is
        (re)computed if it is missing or was computed with a different
        grouping.
    key : str, default='rank_genes_groups'
        The key under which the precomputed ranking is stored on `data`.
    use_adjusted_pvalue : bool, default=True
        Whether to use adjusted p-values instead of raw p-values for the
        significance call and transform.
    logfoldchange_threshold : float, default=1.0
        Absolute log fold change threshold used to label up/down regulated.
    pvalue_threshold : float, default=0.05
        P-value threshold used to label significance.
    variable_column : str, default='variable'
        Output column name for the gene/feature names.
    logfoldchange_column : str, default='logfoldchange'
        Output column name for the log fold changes.
    pvalue_column : str | None, default=None
        Output column name for the p-values. Defaults to `pvalue_adj` when
        `use_adjusted_pvalue` is True and `pvalue` otherwise.
    neg_log_pvalue_column : str, default='neg_log_pvalue'
        Output column name for the `-log10(pvalue)` transform.
    significance_column : str, default='significance'
        Output column name for the categorical significance label.
    up_label : str, default='up'
        Label for significantly up regulated features.
    down_label : str, default='down'
        Label for significantly down regulated features.
    nonsignificant_label : str, default='ns'
        Label for non-significant features.
    rank_genes_kwargs : dict | None, default=None
        Additional keyword arguments forwarded to the ranking call when
        `group_by` triggers a (re)compute (e.g. `method='t-test'`,
        `use_raw=False`, `layer=...`, `corr_method='bonferroni'`).

    Returns
    -------
    DataFrame
        A polars DataFrame with one row per feature and the columns named
        according to the `*_column` parameters above.

    Notes
    -----
    Extracts the ranking for `group` and adds derived columns
    `-log10(pvalue)` and a categorical significance label (up / down / ns)
    based on the supplied thresholds.
    """
    pvalue_column = _resolve_pvalue_column(pvalue_column, use_adjusted_pvalue=use_adjusted_pvalue)

    # HANDLE: Data types
    if isinstance(data, AnnData):
        # COMPUTE: rank_genes_groups if requested and missing/stale
        if group_by is not None:
            existing_params = data.uns.get(key, {}).get("params", {})
            desired_params = {"groupby": group_by, **(rank_genes_kwargs or {})}
            stale = any(existing_params.get(k) != v for k, v in desired_params.items())
            if key not in data.uns or stale:
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
        raise _unsupported_data_type(data, AnnData)

    frame = frame.filter(
        pl.col(logfoldchange_column).is_not_null()
        & pl.col(logfoldchange_column).is_finite()
        & pl.col(pvalue_column).is_not_null()
        & pl.col(pvalue_column).is_finite()
    )

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

    Parameters
    ----------
    data : AnnData
        Source object that already holds a precomputed ranking result.
    key : bool | str
        `True` reads the default ranking key; a string reads
        the matching custom key (e.g. `"rank_genes_groups_wilcoxon"`).
    n_genes : int
        Number of top genes to pull per group.
    groups : Sequence[str] | None
        Subset of groups to include, in order. `None` keeps all groups in
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
        The categorical key used when the ranking was computed.

    Raises
    ------
    UnsupportedDataTypeError
        If `data` is not a supported single-cell data object.
    KeyNotFoundError
        If the ranking result, a requested group, or the stored grouping
        is missing.
    ValueError
        If `n_genes` is out of range.

    Notes
    -----
    Pulls the top `n_genes` gene names and their scores for each selected
    group from a precomputed ranking. Unlike the heatmap extractor, genes
    are kept per group with no cross-group de-duplication, since each group
    is drawn in its own panel.
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
        stored_group_by = params.get("groupby")
        if stored_group_by is None:
            msg = (
                f"`adata.uns[{uns_key!r}]['params']` is missing 'groupby'; "
                "cannot infer `group_by`."
            )
            raise KeyNotFoundError(msg)

        # Round scores to keep the embedded JSON compact when the frame is
        # serialized into the plot output.
        frame = frame.with_columns(pl.col(score_column).round(4))
        frame = frame.filter(pl.col(score_column).is_not_null() & pl.col(score_column).is_finite())

        return frame, list(selected), str(stored_group_by)

    raise _unsupported_data_type(data, AnnData)
