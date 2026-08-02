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
from cellestial.util.utilities import _color_gradient, _fill_gradient


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
    """Compute one uncorrected two-sided t-test bracket."""
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
    expected = ttest_ind([1.0, 2.0], [10.0, 11.0, 12.0])

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
