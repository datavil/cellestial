from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl
from lets_plot import aes, geom_bracket

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import get_mapping, retrieve
from cellestial.util.errors import InvalidComparisonError, MissingAestheticError

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


_STAR_THRESHOLDS = (
    (0.0001, "****"),
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
)


def _significance_stars(pvalue: float) -> str:
    """Convert a p-value into an asterisk significance label."""
    for threshold, stars in _STAR_THRESHOLDS:
        if pvalue < threshold:
            return stars
    return "ns"


def _star_threshold(pvalue: float) -> float | None:
    """Return the star-bracket upper bound for a p-value, or None if not significant."""
    for threshold, _ in _STAR_THRESHOLDS:
        if pvalue < threshold:
            return threshold
    return None


def _expand_comparisons(
    comparisons: Sequence[Sequence[str]],
    groups: list[str],
) -> list[tuple[str, str]]:
    """Expand `"*"` wildcards in user comparisons against the available groups."""
    expanded: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    available_groups = set(groups)

    def _validate_group(group: str) -> None:
        if group == "*":
            return
        if group not in available_groups:
            msg = (
                f"Group {group!r} in `comparisons` was not found. "
                f"Available groups: {groups}"
            )
            raise InvalidComparisonError(msg)

    def _add(group_a: str, group_b: str) -> None:
        key = frozenset((group_a, group_b))
        if len(key) < 2 or key in seen:
            return
        seen.add(key)
        expanded.append((group_a, group_b))

    for pair in comparisons:
        try:
            pair_length = len(pair)
        except TypeError:
            pair_length = 0
        if isinstance(pair, str) or pair_length != 2:
            msg = "Each comparison must contain exactly two groups, e.g. ('A', 'B')."
            raise InvalidComparisonError(msg)
        group_a, group_b = pair
        _validate_group(group_a)
        _validate_group(group_b)
        if group_a == "*" and group_b == "*":
            for left, right in combinations(groups, 2):
                _add(left, right)
        elif group_b == "*":
            for other in groups:
                _add(group_a, other)
        elif group_a == "*":
            for other in groups:
                _add(other, group_b)
        else:
            _add(group_a, group_b)
    return expanded


def _compute_bracket_frame(
    frame: DataFrame,
    *,
    x: str,
    y: str,
    comparisons: Sequence[Sequence[str]] | None,
    test: Literal["mannwhitney", "ttest"],
    alternative: Literal["two-sided", "less", "greater"],
    correction: Literal["none", "bonferroni", "fdr_bh"],
    label: Literal["stars", "pvalue", "padj"] | Sequence[Literal["stars", "pvalue", "padj"]],
    label_format: str,
    prefix: str,
    prefix_style: Literal["=", "<"] | None,
    separator: str,
    threshold: float | None,
    y_position: float | None,
    y_step: float | None,
    y_padding: float,
) -> DataFrame:
    """Build a `geom_bracket` DataFrame from pairwise significance tests."""
    from scipy.stats import mannwhitneyu, ttest_ind

    # determine which pairs to compare
    groups = frame[x].drop_nulls().unique(maintain_order=True).to_list()
    if comparisons is None:
        pairs = list(combinations(groups, 2))
    else:
        pairs = _expand_comparisons(comparisons, groups)

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

    if not records:
        msg = (
            "No valid group comparisons available. "
            "Each comparison needs at least 2 observations per group."
        )
        raise ValueError(msg)

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

    # build the label column - one component per label kind, merged row-wise
    label_kinds = [label] if isinstance(label, str) else list(label)
    valid_kinds = {"stars", "pvalue", "padj"}
    invalid = [kind for kind in label_kinds if kind not in valid_kinds]
    if invalid:
        msg = (
            "`label` entries must be from 'stars', 'pvalue', 'padj'. "
            f"Received invalid: {invalid!r}"
        )
        raise ValueError(msg)
    if len(label_kinds) == 0:
        msg = "`label` must contain at least one entry."
        raise ValueError(msg)

    pvalues_raw = brackets["pvalue"].to_list()
    pvalues_adj = brackets["pvalue_adj"].to_list()
    pvalues_for_stars = brackets["pvalue_adj" if correction != "none" else "pvalue"].to_list()

    def _format_pvalue(pvalue: float) -> str:
        if prefix_style == "<":
            threshold = _star_threshold(pvalue)
            symbol, value = (">", 0.05) if threshold is None else ("<", threshold)
            return f"{prefix} {symbol} {value:{label_format}}".lstrip()
        if prefix_style == "=":
            if not prefix:
                return f"{pvalue:{label_format}}"
            return f"{prefix} = {pvalue:{label_format}}"
        if prefix_style is None:
            return f"{prefix}{pvalue:{label_format}}"
        msg = f"`prefix_style` must be '=', '<', or None. Received: {prefix_style!r}"
        raise ValueError(msg)

    components = []
    for kind in label_kinds:
        if kind == "stars":
            components.append([_significance_stars(pvalue) for pvalue in pvalues_for_stars])
        elif kind == "pvalue":
            components.append([_format_pvalue(pvalue) for pvalue in pvalues_raw])
        else:  # "padj"
            components.append([_format_pvalue(pvalue) for pvalue in pvalues_adj])

    labels = [separator.join(parts) for parts in zip(*components, strict=True)]
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
    *,
    plot: PlotSpec | None = None,
    comparisons: Sequence[Sequence[str]] | None = None,
    test: Literal["mannwhitney", "ttest"] = "mannwhitney",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    correction: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh",
    label: Literal["stars", "pvalue", "padj"]
    | Sequence[Literal["stars", "pvalue", "padj"]] = "stars",
    label_format: str = ".3g",
    prefix: str = "",
    prefix_style: Literal["=", "<"] | None = "=",
    separator: str = " ",
    threshold: float | None = None,
    y_position: float | None = None,
    y_step: float | None = None,
    y_padding: float = 0.08,
    color: str | None = "#1f1f1f",
    label_size: float | None = None,
    segment_size: float = 0.5,
    x: str | None = None,
    y: str | None = None,
    mapping: FeatureSpec | None = None,
    **geom_kwargs,
) -> DeferredLayer:
    """
    Layer of `geom_bracket` annotating pairwise significance between groups.

    Pairwise tests are computed from the plot's retrieved DataFrame using the `x`
    aesthetic as the grouping column and the `y` aesthetic as the value column.

    Parameters
    ----------
    plot : PlotSpec | None, default=None
        If provided, brackets are computed from this plot's data and aesthetics
        regardless of which plot the resulting layer is added to. When ``None``,
        the layer is deferred and introspects the plot it is added to via ``+``.
        Expected to be a distribution plot (e.g. `violin`, `boxplot`).
    comparisons : Sequence[Sequence[str]] | None, default=None
        Specific group pairs to test, e.g. ``[("A", "B"), ("A", "C")]``.
        If None, every pair of groups present in the plot is compared.
        ``"*"`` on either side expands to all other groups, so
        ``[("A", "*")]`` compares ``A`` against every remaining group.
    test : {'mannwhitney', 'ttest'}, default='mannwhitney'
        Statistical test used for each pairwise comparison.
        'mannwhitney' is non-parametric and recommended for expression data.
        'ttest' is an independent two-sample t-test (assumes normality).
    alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
        The alternative hypothesis passed to the underlying test.
    correction : {'none', 'bonferroni', 'fdr_bh'}, default='fdr_bh'
        Multiple-testing correction applied across the pairwise p-values.
    label : {'stars', 'pvalue', 'padj'} or sequence of those, default='stars'
        The bracket label to draw.
        'stars' maps p-values to asterisks (``****`` < 0.0001, ``***`` < 0.001,
        ``**`` < 0.01, ``*`` < 0.05, ``ns`` otherwise).
        'pvalue' prints the raw p-value and 'padj' prints the adjusted p-value.
        A sequence merges each component with a newline, e.g.
        ``label=("stars", "padj")`` stacks the stars on top of the adjusted p-value.
    label_format : str, default='.3g'
        Format string used when `label` contains 'pvalue' or 'padj'.
    prefix : str, default=''
        Text prepended to the numeric p-value/padj labels. Ignored for the
        'stars' component. When `prefix_style` is '=' or '<', the symbol is
        inserted between `prefix` and the value with whitespace around it,
        so ``prefix='p'`` with ``prefix_style='='`` renders ``'p = 0.001'``.
        When `prefix_style` is None, `prefix` is used as-is, so the caller
        is responsible for including any symbol and spacing.
    prefix_style : {'=', '<'} or None, default='='
        How the numeric portion of the label is rendered.
        '=' shows the actual p-value with ``' = '`` inserted after `prefix`.
        '<' replaces the p-value with the star-bracket threshold it falls
        below, with ``' < '`` inserted after `prefix`, giving
        ``'p < 0.0001'``, ``'p < 0.001'``, ``'p < 0.01'``, or ``'p < 0.05'``.
        Non-significant comparisons render as ``'p > 0.05'``.
        None disables symbol insertion; the caller supplies any symbol via
        `prefix`.
    separator : str, default=' '
        Separator used to join components when `label` is a sequence with
        more than one entry (e.g. ``('stars', 'padj')``).
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

    Raises
    ------
    MissingAestheticError
        If `x` or `y` cannot be provided explicitly or inferred from the
        receiving plot.
    InvalidComparisonError
        If `comparisons` contains malformed pairs or groups not present in
        the plot data.
    ValueError
        If a statistical option or label option is invalid, or there are too
        few observations to compute brackets.

    Examples
    --------
    Annotate a violin plot (or boxplot) with pairwise significance stars.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from lets_plot import *

        data = cl.datasets.pbmc3k(cache_directory="data")

        violin = cl.violin(
            data,
            key="CD3D",
            fill="cell_type_lvl1",
            threshold=0.1,
        )
        violin + cl.bracket(y_padding=0.2)

    Restrict the comparisons and show adjusted p-values instead of stars.

    .. jupyter-execute::

        box = cl.boxplot(
            data,
            key="CD3D",
            fill="cell_type_lvl1",
            threshold=0.1,
        )
        box + cl.bracket(
            comparisons=[("Monocytes", "Erythroid"), ("Monocytes", "B Cells")],
            label="padj",
            y_padding=0.2,
        )

    Use ``"*"`` to compare one group against every other.

    .. jupyter-execute::
        :emphasize-lines: 2

        box + cl.bracket(
            comparisons=[("Lymphocytes", "*")],
            y_padding=0.2,
        )

    Hide non-significant brackets by setting a threshold on the adjusted p-value.

    .. jupyter-execute::

        box + cl.bracket(threshold=0.001,y_padding=0.2)

    Adjust the label format places and add a prefix.

    .. jupyter-execute::
        :emphasize-lines: 4-5

        box + cl.bracket(
            threshold=0.05,
            label="pvalue",
            prefix="p",
            prefix_style="=",
            y_padding=0.2,
        )

    Use "<" notation instead.

    .. jupyter-execute::
        :emphasize-lines: 4

        box + cl.bracket(
            threshold=0.05,
            label="pvalue",
            prefix="p",
            prefix_style="<",
            y_padding=0.2,
        )
    """
    explicit_plot = plot
    explicit_x = x
    explicit_y = y

    def _build(receiving_plot: PlotSpec) -> LayerSpec:
        source = explicit_plot if explicit_plot is not None else receiving_plot
        # extract data and mapping from the plot
        frame = retrieve(source)
        _mapping = get_mapping(source)
        x = _mapping.get("x") if explicit_x is None else explicit_x
        y = _mapping.get("y") if explicit_y is None else explicit_y
        if x is None:
            msg = "`x` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)
        if y is None:
            msg = "`y` is present neither as argument nor in the plot aesthetics."
            raise MissingAestheticError(msg)

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
            prefix=prefix,
            prefix_style=prefix_style,
            separator=separator,
            threshold=threshold,
            y_position=y_position,
            y_step=y_step,
            y_padding=y_padding,
        )

        # build and return the layer
        local_mapping = mapping or aes()
        size_kwargs = {} if label_size is None else {"size": label_size}
        local_color = None if "color" in local_mapping.as_dict() else color
        return geom_bracket(
            data=brackets,
            mapping=aes(xmin="xmin", xmax="xmax", y="y", label="label", **local_mapping.as_dict()),
            color=local_color,
            segment_size=segment_size,
            **size_kwargs,
            **geom_kwargs,
        )

    return DeferredLayer(_build)
