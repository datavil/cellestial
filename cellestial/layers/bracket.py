from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from lets_plot import aes, geom_bracket

from cellestial.util import get_mapping, retrieve

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lets_plot.plot.core import FeatureSpec, LayerSpec, PlotSpec
    from polars import DataFrame


def _correct_pvalues(
    pvalues: np.ndarray,
    method: Literal["none", "bonferroni", "fdr_bh"],
) -> np.ndarray:
    """Adjust p-values for multiple testing."""
    if method == "none":
        return pvalues
    if method == "bonferroni":
        return np.minimum(pvalues * len(pvalues), 1.0)
    if method == "fdr_bh":
        n = len(pvalues)
        order = np.argsort(pvalues)
        ranked = pvalues[order]
        adjusted = ranked * n / np.arange(1, n + 1)
        # enforce monotonicity from the largest p-value downwards
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.minimum(adjusted, 1.0)
        result = np.empty_like(adjusted)
        result[order] = adjusted
        return result
    msg = f"`correction` must be one of 'none', 'bonferroni', 'fdr_bh'. Received: {method!r}"
    raise ValueError(msg)


def _significance_stars(pvalue: float) -> str:
    """Convert a p-value into an asterisk significance label."""
    if pvalue < 0.0001:
        return "****"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def _compute_bracket_frame(
    frame: DataFrame,
    *,
    x: str,
    y: str,
    comparisons: Sequence[Sequence[str]] | None,
    test: Literal["mannwhitney", "ttest"],
    alternative: Literal["two-sided", "less", "greater"],
    correction: Literal["none", "bonferroni", "fdr_bh"],
    label: Literal["stars", "pvalue", "padj"],
    label_format: str,
    threshold: float | None,
    y_position: float | None,
    y_step: float | None,
    y_padding: float,
) -> DataFrame:
    """Build a `geom_bracket` DataFrame from pairwise significance tests."""
    from scipy.stats import mannwhitneyu, ttest_ind

    # determine which pairs to compare
    if comparisons is None:
        groups = frame[x].unique().to_list()
        pairs = list(combinations(groups, 2))
    else:
        pairs = [tuple(pair) for pair in comparisons]

    if len(pairs) == 0:
        msg = "No group pairs available to compare."
        raise ValueError(msg)

    # run the pairwise test
    test_functions = {
        "mannwhitney": lambda a, b: mannwhitneyu(a, b, alternative=alternative),
        "ttest": lambda a, b: ttest_ind(a, b, alternative=alternative),
    }
    if test not in test_functions:
        msg = f"`test` must be one of {list(test_functions)}. Received: {test!r}"
        raise ValueError(msg)
    run_test = test_functions[test]

    records = []
    for group_a, group_b in pairs:
        values_a = frame.filter(pl.col(x) == group_a)[y].to_numpy()
        values_b = frame.filter(pl.col(x) == group_b)[y].to_numpy()
        if len(values_a) < 2 or len(values_b) < 2:
            continue
        result = run_test(values_a, values_b)
        records.append(
            {
                "xmin": group_a,
                "xmax": group_b,
                "statistic": float(result.statistic),
                "pvalue": float(result.pvalue),
            }
        )

    brackets = pl.DataFrame(records)

    # adjust for multiple testing
    pvalue_adjusted = _correct_pvalues(brackets["pvalue"].to_numpy(), method=correction)
    brackets = brackets.with_columns(pvalue_adj=pl.Series(pvalue_adjusted))

    # filter by threshold
    if threshold is not None:
        pvalue_column = "pvalue_adj" if correction != "none" else "pvalue"
        brackets = brackets.filter(pl.col(pvalue_column) <= threshold)
        if len(brackets) == 0:
            return brackets

    # build the label column
    pvalues_for_label = brackets["pvalue_adj" if correction != "none" else "pvalue"].to_list()
    if label == "stars":
        labels = [_significance_stars(pvalue) for pvalue in pvalues_for_label]
    elif label in {"pvalue", "padj"}:
        source = "pvalue_adj" if label == "padj" else "pvalue"
        labels = [f"{pvalue:{label_format}}" for pvalue in brackets[source].to_list()]
    else:
        msg = f"`label` must be one of 'stars', 'pvalue', 'padj'. Received: {label!r}"
        raise ValueError(msg)
    brackets = brackets.with_columns(label=pl.Series(labels))

    # compute y positions so brackets stack above the data without overlapping
    values = frame[y].to_numpy()
    data_min = float(values.min())
    data_max = float(values.max())
    data_range = data_max - data_min
    _y_position = data_max + data_range * y_padding if y_position is None else y_position
    _y_step = data_range * y_padding if y_step is None else y_step
    positions = [_y_position + _y_step * index for index in range(len(brackets))]
    brackets = brackets.with_columns(y=pl.Series(positions))

    return brackets


def bracket(
    plot: PlotSpec,
    *,
    comparisons: Sequence[Sequence[str]] | None = None,
    test: Literal["mannwhitney", "ttest"] = "mannwhitney",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    correction: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh",
    label: Literal["stars", "pvalue", "padj"] = "stars",
    label_format: str = ".3g",
    threshold: float | None = None,
    y_position: float | None = None,
    y_step: float | None = None,
    y_padding: float = 0.08,
    color: str = "#1f1f1f",
    label_size: float | None = None,
    segment_size: float = 1,
    x: str | None = None,
    y: str | None = None,
    mapping: FeatureSpec | None = None,
    **geom_kwargs,
) -> LayerSpec:
    """
    Returns a Layer of `geom_bracket` annotating pairwise significance between groups.

    Pairwise tests are computed from the plot's retrieved DataFrame using the `x`
    aesthetic as the grouping column and the `y` aesthetic as the value column.

    Parameters
    ----------
    plot : PlotSpec
        The plot to which the layer will be added. Used to extract data and aesthetics.
        Expected to be a distribution plot (e.g. `violin`, `boxplot`).
    comparisons : Sequence[Sequence[str]] | None, default=None
        Specific group pairs to test, e.g. ``[("A", "B"), ("A", "C")]``.
        If None, every pair of groups present in the plot is compared.
    test : {'mannwhitney', 'ttest'}, default='mannwhitney'
        Statistical test used for each pairwise comparison.
        'mannwhitney' is non-parametric and recommended for expression data.
        'ttest' is an independent two-sample t-test (assumes normality).
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        The alternative hypothesis passed to the underlying test.
    correction : {'none', 'bonferroni', 'fdr_bh'}, default='fdr_bh'
        Multiple-testing correction applied across the pairwise p-values.
    label : {'stars', 'pvalue', 'padj'}, default='stars'
        The bracket label to draw.
        'stars' maps p-values to asterisks (``****`` < 0.0001, ``***`` < 0.001,
        ``**`` < 0.01, ``*`` < 0.05, ``ns`` otherwise).
        'pvalue' prints the raw p-value and 'padj' prints the adjusted p-value.
    label_format : str, default='.3g'
        Format string used when `label` is 'pvalue' or 'padj'.
    threshold : float | None, default=None
        If provided, only comparisons whose p-value (adjusted if `correction` is not
        'none') is below this threshold are drawn.
    y_position : float | None, default=None
        Vertical position of the lowest bracket. If None, it is placed above
        the maximum `y` value with a gap of ``y_padding * y_range``.
    y_step : float | None, default=None
        Vertical spacing between stacked brackets. If None, defaults to
        ``y_padding * y_range``.
    y_padding : float, default=0.08
        Fraction of the `y` range used as vertical padding above the data
        and as the default spacing between stacked brackets.
    color : str, default='#1f1f1f'
        Color of the brackets and labels.
    label_size : float | None, default=None
        Font size of the label text. If None, `geom_bracket`'s default is used.
    segment_size : float, default=1
        Line width of the bracket segments.
    x : str | None, default=None
        The column name holding the grouping categories. If None, inferred from
        the plot's `x` aesthetic.
    y : str | None, default=None
        The column name holding the numerical values. If None, inferred from
        the plot's `y` aesthetic.
    mapping : FeatureSpec | None, default=None
        Additional aesthetic mappings for the layer, the result of `aes()`.
    **geom_kwargs
        Additional parameters for the `geom_bracket` layer.
        For more information on geom_bracket parameters, see:
        https://lets-plot.org/python/pages/api/lets_plot.geom_bracket.html

    Returns
    -------
    LayerSpec
        Pairwise significance brackets.

    Examples
    --------
    Annotate a violin plot with pairwise significance stars.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = sc.read_h5ad("data/pbmc3k_pped.h5ad")

        violin = cl.violin(
            data,
            key="CD3D",
            fill="cell_type_lvl1",
            threshold=0.1,
        )
        violin + cl.bracket(violin)

    Restrict the comparisons and show adjusted p-values instead of stars.

    .. jupyter-execute::

        violin = cl.violin(
            data,
            key="CD3D",
            fill="cell_type_lvl1",
            threshold=0.1,
        )
        violin + cl.bracket(
            violin,
            comparisons=[("Lymphocytes", "Monocytes"), ("Lymphocytes", "B Cells")],
            label="padj",
        )

    Hide non-significant brackets by setting a threshold on the adjusted p-value.

    .. jupyter-execute::

        violin = cl.violin(
            data,
            key="CD3D",
            fill="cell_type_lvl1",
            threshold=0.1,
        )
        violin + cl.bracket(violin, threshold=0.05)
    """
    # extract data and mapping from the plot
    frame = retrieve(plot)
    _mapping = get_mapping(plot)
    x = _mapping.get("x") if x is None else x
    y = _mapping.get("y") if y is None else y
    if x is None:
        msg = "`x` is present neither as argument nor in the plot aesthetics."
        raise ValueError(msg)
    if y is None:
        msg = "`y` is present neither as argument nor in the plot aesthetics."
        raise ValueError(msg)

    # compute the pairwise significance frame
    brackets = _compute_bracket_frame(
        frame,
        x=x,
        y=y,
        comparisons=comparisons,
        test=test,
        alternative=alternative,
        correction=correction,
        label=label,
        label_format=label_format,
        threshold=threshold,
        y_position=y_position,
        y_step=y_step,
        y_padding=y_padding,
    )

    # build and return the layer
    mapping = mapping or aes()
    size_kwargs = {} if label_size is None else {"size": label_size}
    return geom_bracket(
        data=brackets,
        mapping=aes(xmin="xmin", xmax="xmax", y="y", label="label", **mapping.as_dict()),
        color=color,
        segment_size=segment_size,
        **size_kwargs,
        **geom_kwargs,
    )
