"""Regression tests for behavior and robustness issues found during the library audit."""

import re
import tomllib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from anndata import AnnData
from lets_plot import aes
from scipy.stats import ttest_ind

import cellestial as cl
from cellestial.datasets import datasets
from cellestial.layers.bracket import _compute_bracket_frame
from cellestial.util import retrieve
from cellestial.util.utilities import _color_gradient, _fill_gradient, _range_inclusive


def _axis_probe_data() -> AnnData:
    """Build data whose observation and variable axes expose the same column names."""
    data = AnnData(
        X=np.ones((3, 2)),
        obs=pd.DataFrame(
            {"x": [10.0, 20.0, 30.0], "y": [40.0, 50.0, 60.0]},
            index=["cell_1", "cell_2", "cell_3"],
        ),
        var=pd.DataFrame(
            {"x": [1.0, 2.0], "y": [3.0, 4.0]},
            index=["gene_1", "gene_2"],
        ),
    )
    data.obsm["X_demo"] = np.array(
        [
            [0.0, 1.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )
    return data


def _compute_single_bracket(frame: pl.DataFrame) -> pl.DataFrame:
    """Compute one uncorrected two-sided Welch t-test bracket."""
    return _compute_bracket_frame(
        frame,
        x="group",
        y="value",
        comparisons=[("a", "b")],
        test="ttest",
        alternative="two-sided",
        correction="none",
        label="stars",
        label_format=".3g",
        prefix="",
        prefix_style="=",
        separator=" ",
        threshold=None,
        y_position=None,
        y_step=None,
        y_padding=0.1,
    )


@pytest.mark.parametrize("nonfinite_value", [None, np.nan, np.inf, -np.inf])
def test_bracket_omits_nonfinite_observations(nonfinite_value):
    """Non-finite observations must not turn a computable test into a NaN result."""
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "a", "b", "b", "b"],
            "value": [1.0, 2.0, nonfinite_value, 10.0, 11.0, 12.0],
        },
        schema={"group": pl.String, "value": pl.Float64},
    )
    expected = ttest_ind([1.0, 2.0], [10.0, 11.0, 12.0], equal_var=False)

    brackets = _compute_single_bracket(frame)

    assert brackets["statistic"][0] == pytest.approx(expected.statistic)
    assert brackets["pvalue"][0] == pytest.approx(expected.pvalue)
    assert brackets["pvalue_adj"][0] == pytest.approx(expected.pvalue)
    assert np.isfinite(brackets["pvalue"][0])
    assert np.isfinite(brackets["y"][0])


@pytest.mark.parametrize("nonfinite_value", [None, np.nan, np.inf, -np.inf])
def test_bracket_rejects_groups_with_too_few_finite_observations(nonfinite_value):
    """Validity checks must count finite observations rather than raw rows."""
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "value": [1.0, nonfinite_value, 10.0, 11.0],
        },
        schema={"group": pl.String, "value": pl.Float64},
    )

    with pytest.raises(ValueError, match="at least 2 observations"):
        _compute_single_bracket(frame)


def test_bracket_rejects_partial_results_with_an_invalid_comparison():
    """A valid pair must not hide another requested pair with insufficient data."""
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c"],
            "value": [1.0, 2.0, 10.0, 11.0, 20.0, np.nan],
        }
    )

    with pytest.raises(ValueError, match=r"'a'.*'c'.*\(1\)"):
        _compute_bracket_frame(
            frame,
            x="group",
            y="value",
            comparisons=[("a", "b"), ("a", "c")],
            test="ttest",
            alternative="two-sided",
            correction="none",
            label="stars",
            label_format=".3g",
            prefix="",
            prefix_style="=",
            separator=" ",
            threshold=None,
            y_position=None,
            y_step=None,
            y_padding=0.1,
        )


@pytest.mark.parametrize(
    ("plotter", "mapping"),
    [
        (cl.scatter, aes(x="x", y="y")),
        (cl.bar, aes(x="x", fill="y")),
    ],
)
def test_basic_plots_respect_explicit_observation_axis(plotter, mapping):
    """An explicit axis zero must take precedence over automatic axis inference."""
    data = _axis_probe_data()

    frame = retrieve(plotter(data, mapping=mapping, axis=0))

    assert frame.height == data.n_obs
    assert "Barcode" in frame.columns
    assert "Variable" not in frame.columns
    assert frame["x"].to_list() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("include_dimensions", [True, 1, 2])
def test_xyplot_explicit_include_dimensions_builds_observation_frame(include_dimensions):
    """The documented include_dimensions option must not leave axis keys uninitialized."""
    data = _axis_probe_data()

    plot = cl.xyplot(
        data,
        x="x",
        y="y",
        include_dimensions=include_dimensions,
        tooltips="none",
    )
    frame = retrieve(plot)

    assert frame.height == data.n_obs
    assert "X_DEMO1" in frame.columns


def test_xyplot_infers_observation_axis_from_embedding_columns():
    """An embedding-only xyplot must materialize obsm rather than varm."""
    data = _axis_probe_data()

    frame = retrieve(cl.xyplot(data, "X_DEMO1", "X_DEMO2", tooltips="none"))

    assert frame.height == data.n_obs
    assert {"X_DEMO1", "X_DEMO2"} <= set(frame.columns)
    assert frame["X_DEMO1"].to_list() == [0.0, 2.0, 4.0]


def test_xyplots_infers_observation_axis_from_embedding_columns():
    """Embedding-only plural plots must use their shared observation frame."""
    data = _axis_probe_data()

    grid = cl.xyplots(data, x="X_DEMO1", y="X_DEMO2", tooltips="none")
    figures = grid.as_dict()["figures"]

    assert len(figures) == 1
    frame = figures[0]["data"]
    assert frame.height == data.n_obs
    assert {"X_DEMO1", "X_DEMO2"} <= set(frame.columns)


@pytest.mark.parametrize("gradient", [_color_gradient, _fill_gradient])
def test_gradients_reject_unknown_midpoint_modes(gradient):
    """Invalid midpoint modes must raise a public input error, not UnboundLocalError."""
    series = pl.Series("value", [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="midpoint"):
        gradient(
            series,
            color_low="white",
            color_mid="gray",
            color_high="black",
            midpoint="unsupported",
        )


@pytest.mark.parametrize("gradient", [_color_gradient, _fill_gradient])
@pytest.mark.parametrize("midpoint", [0.0, 4.0])
def test_gradients_reject_midpoints_outside_data_range(gradient, midpoint):
    """A diverging scale midpoint must lie within its finite data range."""
    series = pl.Series("value", [1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="within the finite data range"):
        gradient(
            series,
            color_low="white",
            color_mid="gray",
            color_high="black",
            midpoint=midpoint,
        )


def test_runtime_metadata_declares_direct_numeric_dependencies():
    """Packages imported directly at runtime must not be supplied only transitively."""
    project_root = Path(__file__).resolve().parents[1]
    configuration = tomllib.loads((project_root / "pyproject.toml").read_text())
    dependencies = configuration["project"]["dependencies"]
    declared = {re.split(r"[\s\[<>=!~]", dependency, maxsplit=1)[0] for dependency in dependencies}

    assert {"numpy", "pandas", "scipy"} <= declared


def test_from_url_sets_a_positive_network_timeout(tmp_path, monkeypatch):
    """Dataset downloads must not be able to block forever at urlopen."""
    calls = []

    class EmptyResponse:
        def __init__(self):
            self.headers = {"Content-Length": "0"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self, block_size):
            return b""

    def recording_urlopen(url, **kwargs):
        calls.append((url, kwargs))
        return EmptyResponse()

    expected = AnnData(X=np.ones((1, 1)))
    monkeypatch.setattr(urllib.request, "urlopen", recording_urlopen)
    monkeypatch.setattr(datasets, "read_h5ad", lambda path: expected)

    result = datasets.from_url(
        "https://example.test/example.h5ad",
        cache_directory=tmp_path,
        bring=False,
    )

    assert result is expected
    assert len(calls) == 1
    timeout = calls[0][1].get("timeout")
    assert isinstance(timeout, int | float)
    assert timeout > 0


def _tied_pvalue_adata() -> AnnData:
    """Build a ranking whose top features all share an underflowed p-value of zero."""
    names = ["gene_a", "gene_b", "gene_c", "gene_d", "gene_e"]
    data = AnnData(X=np.ones((4, len(names))))
    data.obs["cluster"] = ["A", "A", "B", "B"]
    data.uns["rank_genes_groups"] = {
        "names": np.rec.fromarrays([names], names=["A"]),
        "scores": np.rec.fromarrays([[5.0, 4.0, 3.0, 2.0, 1.0]], names=["A"]),
        # gene_c carries the largest fold change but sits last among the ties
        "logfoldchanges": np.rec.fromarrays([[2.0, 3.0, 8.0, -9.0, -1.5]], names=["A"]),
        "pvals": np.rec.fromarrays([[0.0, 0.0, 0.0, 0.0, 0.001]], names=["A"]),
        "pvals_adj": np.rec.fromarrays([[0.0, 0.0, 0.0, 0.0, 0.002]], names=["A"]),
        "params": {"groupby": "cluster"},
    }
    return data


def _label_layer_data(plot) -> pl.DataFrame:
    """Return the frame behind the volcano's repelled gene labels."""
    layers = [layer for layer in plot.as_dict()["layers"] if layer["geom"] == "text_repel"]
    assert len(layers) == 1
    return pl.DataFrame(layers[0]["data"])


def test_volcano_breaks_label_ties_by_fold_change():
    """Features sharing a zero p-value must be labelled by fold change, not sort order."""
    plot = cl.volcano(_tied_pvalue_adata(), "A", top_n=2, nonsignificant_subsample=None)

    labelled = _label_layer_data(plot)
    up_labels = labelled.filter(pl.col("logfoldchange") > 0)["variable"].to_list()

    # gene_c (logFC 8.0) and gene_b (3.0) beat gene_a (2.0) at the same p-value
    assert up_labels == ["gene_c", "gene_b"]


def test_volcano_label_selection_is_deterministic():
    """Repeated builds of the same volcano must label the same features."""
    data = _tied_pvalue_adata()
    selections = {
        tuple(_label_layer_data(cl.volcano(data, "A", top_n=2))["variable"].to_list())
        for _ in range(5)
    }

    assert len(selections) == 1


def test_volcano_names_adjusted_pvalues_as_adjusted():
    """The default plot reports FDR values, so nothing may call them raw p-values."""
    data = _tied_pvalue_adata()

    adjusted = cl.volcano(data, "A").as_dict()
    raw = cl.volcano(data, "A", use_adjusted_pvalue=False).as_dict()

    assert "pvalue_adj" in adjusted["data"].columns
    assert "pvalue" not in adjusted["data"].columns
    assert adjusted["guides"]["y"]["title"] == "-log10(Padj)"
    assert adjusted["layers"][0]["tooltips"]["variables"] == [
        "variable",
        "logfoldchange",
        "pvalue_adj",
        "significance",
    ]

    assert "pvalue" in raw["data"].columns
    assert raw["guides"]["y"]["title"] == "-log10(Pvalue)"

    # an explicit name still wins over the resolved default
    named = cl.volcano(data, "A", pvalue_column="p").as_dict()
    assert "p" in named["data"].columns


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [(0, 10, 4), (0, 7, 5), (0, 0.5, 5), (2, 3, 4), (0, 1, 3), (-1, 1, 5), (0, 0.001, 4)],
)
def test_range_inclusive_hits_both_endpoints(start, stop, step):
    """Rounding the increment used to drop the endpoint or overshoot the range."""
    values = _range_inclusive(start, stop, step)

    assert values[0] == pytest.approx(start)
    assert values[-1] == pytest.approx(stop)
    assert values == sorted(values)


def test_range_inclusive_accepts_a_descending_range():
    """A stop below the start used to raise from `log10` of a negative span."""
    assert _range_inclusive(1.0, 0.5, 3) == [0.5, 0.75, 1.0]


def test_range_inclusive_rejects_a_nonpositive_step():
    """A step count below one has no evenly spaced values to return."""
    with pytest.raises(ValueError, match="`step` must be >= 1"):
        _range_inclusive(0, 1, 0)


def _stream_arrow_frame(paths: list[list[tuple[float, float]]]) -> pl.DataFrame:
    """Run the stream layer's arrow-placement expression over raw path vertices."""
    from cellestial.layers.stream import stream  # noqa: F401  (import guard only)

    frame_streams = pl.DataFrame(
        [
            {"x": float(px), "y": float(py), "group": index}
            for index, vertices in enumerate(paths)
            for px, py in vertices
        ]
    )
    return (
        (
            frame_streams.group_by("group")
            .agg(pl.col("x"), pl.col("y"))
            .filter(pl.col("x").list.len() >= 2)
            .with_columns(
                mid=pl.min_horizontal(
                    pl.col("x").list.len() // 2,
                    pl.col("x").list.len() - 2,
                )
            )
        )
        .with_columns(
            pl.col("x").list.get(pl.col("mid")).alias("x"),
            pl.col("y").list.get(pl.col("mid")).alias("y"),
            pl.col("x").list.get(pl.col("mid").add(1)).alias("xend"),
            pl.col("y").list.get(pl.col("mid").add(1)).alias("yend"),
        )
        .drop("mid")
        .filter((pl.col("x") != pl.col("xend")) | (pl.col("y") != pl.col("yend")))
        .sort("group")
    )


def test_stream_arrows_survive_short_and_degenerate_paths():
    """A two-vertex streamline used to index past the end of its own path."""
    arrows = _stream_arrow_frame(
        [
            [(0.0, 0.0)],  # single vertex: no segment to point along
            [(0.0, 0.0), (1.0, 1.0)],  # two vertices: `mid + 1` was out of bounds
            [(0.0, 0.0), (1.0, 1.0), (2.0, 3.0)],
            [(5.0, 5.0), (5.0, 5.0)],  # repeated vertex: zero-length arrow
        ]
    )

    assert arrows["group"].to_list() == [1, 2]
    assert arrows["xend"].null_count() == 0
    assert arrows["yend"].null_count() == 0


def _unequal_variance_frame() -> pl.DataFrame:
    """Two groups whose sizes and variances both differ, as cell types do."""
    rng = np.random.default_rng(0)
    return pl.DataFrame(
        {
            "group": ["a"] * 60 + ["b"] * 12,
            "value": np.concatenate([rng.normal(0.0, 1.0, 60), rng.normal(0.6, 0.1, 12)]),
        }
    )


def _bracket_pvalue(frame: pl.DataFrame, test: str) -> float:
    """Run one uncorrected pairwise bracket and return its p-value."""
    brackets = _compute_bracket_frame(
        frame,
        x="group",
        y="value",
        comparisons=[("a", "b")],
        test=test,
        alternative="two-sided",
        correction="none",
        label="stars",
        label_format=".3g",
        prefix="",
        prefix_style="=",
        separator=" ",
        threshold=None,
        y_position=None,
        y_step=None,
        y_padding=0.1,
    )
    return brackets["pvalue"][0]


def test_bracket_ttest_does_not_assume_equal_variances():
    """`ttest` must be Welch's, matching R's `t.test` and scanpy's own ranking."""
    frame = _unequal_variance_frame()
    values_a = frame.filter(pl.col("group") == "a")["value"].to_numpy()
    values_b = frame.filter(pl.col("group") == "b")["value"].to_numpy()
    welch = ttest_ind(values_a, values_b, equal_var=False)

    assert _bracket_pvalue(frame, "ttest") == pytest.approx(welch.pvalue)


def test_bracket_student_keeps_the_pooled_variance_test():
    """The pooled-variance test stays reachable under its own name."""
    frame = _unequal_variance_frame()
    values_a = frame.filter(pl.col("group") == "a")["value"].to_numpy()
    values_b = frame.filter(pl.col("group") == "b")["value"].to_numpy()
    student = ttest_ind(values_a, values_b, equal_var=True)

    assert _bracket_pvalue(frame, "student") == pytest.approx(student.pvalue)
    # the assumption matters: the two tests disagree on this data
    assert _bracket_pvalue(frame, "ttest") != pytest.approx(student.pvalue)
