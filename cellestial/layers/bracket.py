from __future__ import annotations

__lazy_modules__ = ["ggrust"]

from itertools import combinations
from typing import TYPE_CHECKING, Literal

import polars as pl
from ggrust import aes, ggsize, stat_compare_means

from cellestial.layers._deferred import DeferredLayer
from cellestial.util import get_mapping, retrieve
from cellestial.util.errors import InvalidComparisonError, MissingAestheticError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ggrust.plot.core import FeatureSpec, FeatureSpecArray, LayerSpec, PlotSpec
    from polars import DataFrame

_TEST_METHODS = {"mannwhitney": "wilcox.test", "ttest": "t.test"}
_CORRECTION_METHODS = {"none": "none", "bonferroni": "bonferroni", "fdr_bh": "BH"}
_LABELS = {"stars": "p.signif", "pvalue": "p.format"}
# canvas used by ggrust when the plot sets no `ggsize`
_DEFAULT_PLOT_SIZE = (600, 400)
# canvas height outside the panel (x axis, margins), in pixels
_NON_PANEL_HEIGHT = 50
# fraction of the y range the default continuous scale adds on each side
_Y_EXPAND = 0.05
# vertical room per stacked bracket at the default label size, in pixels
_BRACKET_STEP_PIXELS = 22
# bracket label size the step above is calibrated for
_DEFAULT_LABEL_SIZE = 5


def _option(name: str, value: str, options: dict[str, str]) -> str:
    """Translate a cellestial option into its `stat_compare_means` name."""
    if value not in options:
        msg = f"`{name}` must be one of {list(options)}. Received: {value!r}"
        raise ValueError(msg)
    return options[value]


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
            msg = f"Group {group!r} in `comparisons` was not found. Available groups: {groups}"
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


def _bracket_layout(
    frame: DataFrame,
    *,
    x: str,
    y: str,
    comparisons: Sequence[Sequence[str]] | None,
    y_position: float | None,
    y_step: float | None,
    y_padding: float,
) -> tuple[DataFrame, list[tuple[str, str]], list[float]]:
    """Return the finite rows, the group pairs and the bracket heights."""
    finite_frame = frame.filter(pl.col(y).is_not_null() & pl.col(y).is_finite())

    # determine which pairs to compare
    groups = frame[x].drop_nulls().unique(maintain_order=True).to_list()
    if comparisons is None:
        pairs = list(combinations(groups, 2))
    else:
        pairs = _expand_comparisons(comparisons, groups)

    if len(pairs) == 0:
        msg = "No group pairs available to compare."
        raise ValueError(msg)

    # every compared group needs at least 2 finite observations
    counts = dict(finite_frame[x].value_counts().iter_rows())
    invalid_comparisons = [
        f"{group_a!r} ({counts.get(group_a, 0)}) vs {group_b!r} ({counts.get(group_b, 0)})"
        for group_a, group_b in pairs
        if counts.get(group_a, 0) < 2 or counts.get(group_b, 0) < 2
    ]
    if invalid_comparisons:
        prefix = (
            "No valid group comparisons available. "
            if len(invalid_comparisons) == len(pairs)
            else "Invalid group comparisons. "
        )
        details = ", ".join(invalid_comparisons)
        msg = (
            f"{prefix}Each comparison needs at least 2 observations per group after "
            f"removing non-finite values. Received: {details}."
        )
        raise ValueError(msg)

    # stack the brackets above the data without overlapping
    data_min = finite_frame[y].min()
    data_max = finite_frame[y].max()
    data_range = data_max - data_min  # ty:ignore[unsupported-operator]
    _y_position = data_max + data_range * y_padding if y_position is None else y_position
    _y_step = data_range * y_padding if y_step is None else y_step
    positions = [_y_position + _y_step * index for index in range(len(pairs))]
    return finite_frame, pairs, positions


def bracket(
    *,
    plot: PlotSpec | None = None,
    comparisons: Sequence[Sequence[str]] | None = None,
    test: Literal["mannwhitney", "ttest"] = "mannwhitney",
    correction: Literal["none", "bonferroni", "fdr_bh"] = "fdr_bh",
    label: Literal["stars", "pvalue"] = "stars",
    hide_ns: bool = False,
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
    Layer of `stat_compare_means` annotating pairwise significance between groups.

    Parameters
    ----------
    plot : PlotSpec | None, default=None
        If provided, brackets are computed from this plot's data and aesthetics
        regardless of which plot the resulting layer is added to. When `None`,
        the layer is deferred and introspects the plot it is added to via `+`.
        Expected to be a distribution plot (e.g. `violin`, `boxplot`).
    comparisons : Sequence[Sequence[str]] | None, default=None
        Specific group pairs to test, e.g. `[("A", "B"), ("A", "C")]`.
        If None, every pair of groups present in the plot is compared.
        `"*"` on either side expands to all other groups, so
        `[("A", "*")]` compares `A` against every remaining group.
    test : {'mannwhitney', 'ttest'}, default='mannwhitney'
        Two-sided statistical test used for each pairwise comparison, as in R.
        'mannwhitney' is the Wilcoxon rank-sum test, non-parametric and
        recommended for expression data; it is exact below 50 tie-free values
        per group (R's `wilcox.test`).
        'ttest' is Welch's independent two-sample t-test (assumes normality).
    correction : {'none', 'bonferroni', 'fdr_bh'}, default='fdr_bh'
        Multiple-testing correction applied across the pairwise p-values.
    label : {'stars', 'pvalue'}, default='stars'
        The bracket label to draw, both from the adjusted p-value.
        'stars' maps p-values to asterisks (`****` <= 0.0001, `***` <= 0.001,
        `**` <= 0.01, `*` <= 0.05, `ns` otherwise).
        'pvalue' prints the p-value with 2 significant digits.
    hide_ns : bool, default=False
        With `label='stars'`, drop the `ns` (not significant) brackets.
    y_position : float | None, default=None
        Vertical position of the lowest bracket. If None, it is placed above
        the maximum `y` value with a gap of `y_padding * y_range`.
    y_step : float | None, default=None
        Vertical spacing between stacked brackets. If None, each bracket gets
        a fixed height on screen sized to its label.
    y_padding : float, default=0.08
        Fraction of the `y` range used as vertical padding above the data.
    color : str, default='#1f1f1f'
        Color of the brackets and labels.
    label_size : float | None, default=None
        Font size of the label text. If None, ggrust's default is used.
    segment_size : float, default=0.5
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
        Additional parameters for the `stat_compare_means` layer, e.g.
        `tip_length`.

    Returns
    -------
    DeferredLayer
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
        If `test`, `correction` or `label` is invalid, or there are too few
        observations to compute brackets.

    Notes
    -----
    Pairwise tests are computed by ggrust's `stat_compare_means` from the plot's
    retrieved DataFrame, using the `x` aesthetic as the grouping column and the
    `y` aesthetic as the value column.

    Rows the plot dropped are absent from the test too, so a plot built with
    `threshold` compares only the observations above it.

    The plot grows taller by the height of the bracket stack, so the data keeps
    its size however many brackets are drawn. A `ggsize` added after the
    brackets overrides this.

    Examples
    --------
    Annotate a violin plot (or boxplot) with pairwise significance stars.

    .. jupyter-execute::

        import cellestial as cl
        import scanpy as sc

        from ggrust import *

        data = cl.datasets.pbmc3k()

        violin = cl.violin(
            data,
            key="CD3D",
            fill="cell_type_lvl1",
            threshold=0.1,
        )
        violin + cl.bracket(y_padding=0.2)

    The `threshold` narrows the comparison too: these p-values cover only the
    cells above `0.1`.

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
            label="pvalue",
            y_padding=0.2,
        )

    Use `"*"` to compare one group against every other.

    .. jupyter-execute::
        :emphasize-lines: 2

        box + cl.bracket(
            comparisons=[("Lymphocytes", "*")],
            y_padding=0.2,
        )

    Hide non-significant brackets.

    .. jupyter-execute::

        box + cl.bracket(hide_ns=True, y_padding=0.2)
    """
    method = _option("test", test, _TEST_METHODS)
    p_adjust_method = _option("correction", correction, _CORRECTION_METHODS)
    p_label = _option("label", label, _LABELS)
    explicit_plot = plot
    explicit_x = x
    explicit_y = y

    def _build(receiving_plot: PlotSpec) -> LayerSpec | FeatureSpecArray:
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

        # size the stacking step in pixels so labels keep their room on any y range
        plot_width, plot_height = receiving_plot.size or _DEFAULT_PLOT_SIZE
        values = frame.filter(pl.col(y).is_finite())[y]
        data_max = values.max()
        data_range = data_max - values.min()  # ty:ignore[unsupported-operator]
        panel_height = plot_height - _NON_PANEL_HEIGHT
        units_per_pixel = data_range * (1 + 2 * _Y_EXPAND) / panel_height
        step_pixels = _BRACKET_STEP_PIXELS * (label_size or _DEFAULT_LABEL_SIZE)
        step_pixels /= _DEFAULT_LABEL_SIZE
        resolved_y_step = step_pixels * units_per_pixel if y_step is None else y_step

        finite_frame, pairs, positions = _bracket_layout(
            frame,
            x=x,
            y=y,
            comparisons=comparisons,
            y_position=y_position,
            y_step=resolved_y_step,
            y_padding=y_padding,
        )

        # build the layer
        local_mapping = mapping or aes()
        size_kwargs = {} if label_size is None else {"size": label_size}
        local_color = None if "color" in local_mapping.as_dict() else color
        layer = stat_compare_means(
            aes(x=x, y=y, **local_mapping.as_dict()),
            data=finite_frame,
            method=method,
            comparisons=pairs,
            hide_ns=hide_ns,
            label=p_label,
            label_y=positions,
            p_adjust_method=p_adjust_method,
            inherit_aes=False,
            color=local_color,
            segment_size=segment_size,
            **size_kwargs,
            **geom_kwargs,
        )
        if data_range == 0:
            return layer

        # grow the canvas by the stack height so the data panel keeps its size
        stack_pixels = (positions[-1] - data_max) / units_per_pixel + step_pixels
        return layer + ggsize(plot_width, plot_height + stack_pixels)

    return DeferredLayer(_build)
