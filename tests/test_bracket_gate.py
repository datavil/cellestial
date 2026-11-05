"""
Gate: ggrust `stat_compare_means` against cellestial's own bracket statistics.

`stat_compare_means` labels carry the p-value at 2 significant digits, so the
comparison is made at that precision.
"""

import re

import numpy as np
import polars as pl
import pytest
from ggrust import aes, geom_boxplot, ggplot, stat_compare_means
from scipy.stats import mannwhitneyu, ttest_ind

from cellestial.layers.bracket import _compute_bracket_frame, _significance_stars

PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]
ADJUSTMENTS = {"none": "none", "bonferroni": "bonferroni", "fdr_bh": "BH"}
TESTS = {"mannwhitney": "wilcox.test", "ttest": "t.test"}


def _frame(n: int, shifts: tuple[float, float, float], seed: int = 0, *, ties: bool = False):
    rng = np.random.default_rng(seed)
    values = [rng.normal(shift, 1, n) for shift in shifts]
    if ties:
        values = [np.round(group) for group in values]
    return pl.DataFrame({"g": [g for g in "ABC" for _ in range(n)], "v": np.concatenate(values)})


def _ggrust_labels(frame: pl.DataFrame, **kwargs) -> list[str]:
    plot = ggplot(frame, aes("g", "v")) + geom_boxplot()
    plot += stat_compare_means(comparisons=PAIRS, **kwargs)
    texts = re.findall(r"<text[^>]*>([^<]*)</text>", plot.to_svg())
    return texts[: len(PAIRS)]


def _cellestial_frame(frame: pl.DataFrame, test: str, correction: str) -> pl.DataFrame:
    return _compute_bracket_frame(
        frame,
        x="g",
        y="v",
        comparisons=PAIRS,
        test=test,
        alternative="two-sided",
        correction=correction,
        label="stars",
        label_format=".3g",
        prefix="",
        prefix_style="=",
        separator=" ",
        threshold=None,
        y_position=None,
        y_step=None,
        y_padding=0.08,
    )


def _two_digits(values) -> list[float]:
    return [float(f"{value:.2g}") for value in values]


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("correction", list(ADJUSTMENTS))
@pytest.mark.parametrize("test", list(TESTS))
def test_pvalues_and_adjustment_match_for_large_groups(test, correction, ties):
    """With >= 50 values per group both paths agree for every supported adjustment."""
    frame = _frame(60, (0.0, 0.4, 0.8), ties=ties)
    labels = _ggrust_labels(frame, method=TESTS[test], p_adjust_method=ADJUSTMENTS[correction])
    expected = _cellestial_frame(frame, test, correction)["pvalue_adj"]
    assert [float(label) for label in labels] == _two_digits(expected)


@pytest.mark.parametrize("correction", list(ADJUSTMENTS))
@pytest.mark.parametrize("shift", [0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
def test_stars_match(shift, correction):
    """Stars come from the adjusted p-value in both paths and share the thresholds."""
    frame = _frame(60, (0.0, shift, 2 * shift), seed=1)
    labels = _ggrust_labels(frame, label="p.signif", p_adjust_method=ADJUSTMENTS[correction])
    expected = _cellestial_frame(frame, "mannwhitney", correction)["label"].to_list()
    assert labels == expected


def test_star_bands_are_all_covered():
    """The star test above reaches every band from 'ns' to '****'."""
    seen = set()
    for shift in [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]:
        frame = _frame(60, (0.0, shift, 2 * shift), seed=1)
        pvalues = _cellestial_frame(frame, "mannwhitney", "none")["pvalue"]
        seen |= {_significance_stars(pvalue) for pvalue in pvalues}
    assert seen == {"ns", "*", "**", "***", "****"}


def test_hide_ns_drops_only_ns_brackets():
    """`hide_ns` removes exactly the brackets labelled `ns`."""
    frame = _frame(60, (0.0, 0.1, 0.2), seed=1)
    stars = _ggrust_labels(frame, label="p.signif")
    hidden = _ggrust_labels(frame, label="p.signif", hide_ns=True)
    assert [star for star in stars if star != "ns"] == [
        label for label in hidden if label in {"*", "**", "***", "****"}
    ]


# Differences ------------------------------------------------------------------


def test_difference_small_groups_wilcoxon_is_exact():
    """
    Below 50 tie-free values per group ggrust runs R's exact Wilcoxon test.

    cellestial (scipy `method="auto"`) uses the normal approximation from 9
    values per group on, so its p-values differ there.
    """
    frame = _frame(30, (0.0, 0.5, 1.0))
    groups = [frame.filter(pl.col("g") == g)["v"].to_numpy() for g in "ABC"]
    exact = [
        mannwhitneyu(groups[i], groups[j], method="exact").pvalue
        for i, j in [(0, 1), (0, 2), (1, 2)]
    ]
    labels = _ggrust_labels(frame, method="wilcox.test")
    assert [float(label) for label in labels] == _two_digits(exact)
    approximate = _cellestial_frame(frame, "mannwhitney", "none")["pvalue"]
    assert _two_digits(approximate) != _two_digits(exact)


def test_difference_student_is_welch():
    """`method="student"` in ggrust is an alias of Welch's `t.test`, not pooled variance."""
    frame = _frame(30, (0.0, 0.5, 1.0))
    groups = [frame.filter(pl.col("g") == g)["v"].to_numpy() for g in "ABC"]
    welch = [
        ttest_ind(groups[i], groups[j], equal_var=False).pvalue
        for i, j in [(0, 1), (0, 2), (1, 2)]
    ]
    labels = _ggrust_labels(frame, method="student")
    assert [float(label) for label in labels] == _two_digits(welch)
