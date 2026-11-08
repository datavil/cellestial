"""
`bracket` statistics, computed by ggrust's `stat_compare_means`, against scipy.

The bracket labels carry the p-value at 2 significant digits, so the comparison
is made at that precision.
"""

import re

import numpy as np
import polars as pl
import pytest
from ggrust import aes, geom_boxplot, ggplot
from scipy.stats import false_discovery_control, mannwhitneyu, ttest_ind

import cellestial as cl

PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]
STARS = ((0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*"))


def _frame(n: int, shifts: tuple[float, float, float], seed: int = 0, *, ties: bool = False):
    rng = np.random.default_rng(seed)
    values = [rng.normal(shift, 1, n) for shift in shifts]
    if ties:
        values = [np.round(group) for group in values]
    return pl.DataFrame({"g": [g for g in "ABC" for _ in range(n)], "v": np.concatenate(values)})


def _labels(frame: pl.DataFrame, **kwargs) -> list[str]:
    plot = ggplot(frame, aes("g", "v")) + geom_boxplot()
    plot += cl.bracket(comparisons=PAIRS, **kwargs)
    texts = re.findall(r"<text[^>]*>([^<]*)</text>", plot.to_svg())
    return texts[: len(PAIRS)]


def _reference(frame: pl.DataFrame, test: str, correction: str, **test_kwargs) -> list[float]:
    groups = {g: frame.filter(pl.col("g") == g)["v"].to_numpy() for g in "ABC"}
    run = {
        "mannwhitney": lambda a, b: mannwhitneyu(a, b, **test_kwargs),
        "ttest": lambda a, b: ttest_ind(a, b, equal_var=False),
    }[test]
    pvalues = np.array([run(groups[a], groups[b]).pvalue for a, b in PAIRS])
    if correction == "bonferroni":
        return np.minimum(pvalues * len(pvalues), 1.0).tolist()
    if correction == "fdr_bh":
        return false_discovery_control(pvalues).tolist()
    return pvalues.tolist()


def _two_digits(values) -> list[float]:
    return [float(f"{value:.2g}") for value in values]


def _stars(pvalue: float) -> str:
    return next((stars for threshold, stars in STARS if pvalue <= threshold), "ns")


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("correction", ["none", "bonferroni", "fdr_bh"])
@pytest.mark.parametrize("test", ["mannwhitney", "ttest"])
def test_pvalues_and_adjustment_match_scipy(test, correction, ties):
    """With >= 50 values per group, the asymptotic scipy tests agree."""
    frame = _frame(60, (0.0, 0.4, 0.8), ties=ties)
    labels = _labels(frame, test=test, correction=correction, label="pvalue")
    assert [float(label) for label in labels] == _two_digits(_reference(frame, test, correction))


@pytest.mark.parametrize("correction", ["none", "bonferroni", "fdr_bh"])
@pytest.mark.parametrize("shift", [0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
def test_stars_follow_the_adjusted_pvalue(shift, correction):
    """Stars come from the adjusted p-value with `<=` cutoffs."""
    frame = _frame(60, (0.0, shift, 2 * shift), seed=1)
    expected = [_stars(p) for p in _reference(frame, "mannwhitney", correction)]
    assert _labels(frame, correction=correction) == expected


def test_star_bands_are_all_covered():
    """The star test above reaches every band from 'ns' to '****'."""
    seen = set()
    for shift in [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]:
        frame = _frame(60, (0.0, shift, 2 * shift), seed=1)
        seen |= {_stars(p) for p in _reference(frame, "mannwhitney", "none")}
    assert seen == {"ns", "*", "**", "***", "****"}


def test_hide_ns_drops_only_ns_brackets():
    """`hide_ns` removes exactly the brackets labelled `ns`."""
    frame = _frame(60, (0.0, 0.1, 0.2), seed=1)
    stars = _labels(frame, correction="none")
    hidden = _labels(frame, correction="none", hide_ns=True)
    assert [star for star in stars if star != "ns"] == [
        label for label in hidden if label in {"*", "**", "***", "****"}
    ]


def test_small_groups_wilcoxon_is_exact():
    """Below 50 tie-free values per group the Wilcoxon test is exact, as in R."""
    frame = _frame(30, (0.0, 0.5, 1.0))
    labels = _labels(frame, correction="none", label="pvalue")
    exact = _reference(frame, "mannwhitney", "none", method="exact")
    assert [float(label) for label in labels] == _two_digits(exact)
