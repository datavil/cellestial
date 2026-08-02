from collections.abc import Callable

import pytest
from anndata import AnnData
from lets_plot import aes
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec
from playwright.sync_api import Page
from plot_assertions import assert_html_renders, assert_plot_renders, assert_point_tooltip

import cellestial as cl

pytestmark = pytest.mark.browser

Plot = PlotSpec | SupPlotsSpec
PlotFactory = Callable[[AnnData, list[str], str], Plot]


def _scatter(data: AnnData, _markers: list[str], group_key: str) -> PlotSpec:
    return cl.scatter(
        data,
        mapping=aes(
            x="n_genes_by_counts",
            y="pct_counts_in_top_50_genes",
            color=group_key,
        ),
    )


def _xyplots(data: AnnData, _markers: list[str], _group_key: str) -> SupPlotsSpec:
    return cl.xyplots(
        data,
        x="n_genes_by_counts",
        y=["pct_counts_mt", "pct_counts_ribo"],
    )


def _violin(data: AnnData, _markers: list[str], group_key: str) -> PlotSpec:
    return cl.violin(data, "CD14", fill=group_key, tooltips=["n_genes_by_counts"])


def _ridge(data: AnnData, _markers: list[str], group_key: str) -> PlotSpec:
    return cl.ridge(
        data,
        key="CD14",
        group_by=group_key,
        tooltips=["n_genes_by_counts"],
    )


def _heatmap(data: AnnData, markers: list[str], group_key: str) -> PlotSpec:
    return cl.heatmap(
        data,
        keys=markers,
        group_by=group_key,
        geom="tile",
        tooltips=["value"],
    )


def _histogram(data: AnnData, _markers: list[str], group_key: str) -> PlotSpec:
    return cl.histogram(data, "n_genes_by_counts", fill=group_key, bins=20)


def _annotated_heatmap(data: AnnData, markers: list[str], group_key: str) -> SupPlotsSpec:
    return cl.annotated_heatmap(
        data,
        keys=markers[:3],
        group_by=group_key,
        row_annotations=["leiden"],
        transpose=True,
        max_rows=30,
    )


def _umaps(data: AnnData, _markers: list[str], group_key: str) -> SupPlotsSpec:
    return cl.umaps(data, [group_key, "CD14"], tooltips=["n_genes_by_counts"])


@pytest.mark.parametrize(
    ("plot_factory", "expected_type"),
    [
        pytest.param(_scatter, PlotSpec, id="scatter"),
        pytest.param(_xyplots, SupPlotsSpec, id="xyplots"),
        pytest.param(_violin, PlotSpec, id="violin"),
        pytest.param(_ridge, PlotSpec, id="ridge"),
        pytest.param(_histogram, PlotSpec, id="histogram"),
        pytest.param(_heatmap, PlotSpec, id="heatmap"),
        pytest.param(_annotated_heatmap, SupPlotsSpec, id="annotated-heatmap"),
        pytest.param(_umaps, SupPlotsSpec, id="umaps"),
    ],
)
def test_representative_plot_families_render_in_browser(
    page: Page,
    adata: AnnData,
    markers: list[str],
    group_key: str,
    plot_factory: PlotFactory,
    expected_type: type[PlotSpec] | type[SupPlotsSpec],
) -> None:
    """Representative completed plots must render without browser errors."""
    plot = plot_factory(adata, markers, group_key)
    assert_plot_renders(page, plot, expected_type)


def test_dimensional_tooltip_works_in_browser(
    page: Page,
    adata: AnnData,
    group_key: str,
) -> None:
    """A configured dimensional tooltip must appear on point hover."""
    plot = cl.umap(
        adata,
        group_key,
        tooltips=[group_key, "n_genes_by_counts"],
    )

    errors = assert_plot_renders(page, plot, PlotSpec)
    expected_groups = {str(value) for value in adata.obs[group_key].unique()}
    assert_point_tooltip(page, errors, expected_values=expected_groups)


def test_browser_assertion_detects_javascript_errors(page: Page) -> None:
    """The browser helper must fail for uncaught JavaScript exceptions."""
    broken_html = """
    <svg class="plt-container" width="100" height="100"></svg>
    <script>throw new Error("browser canary");</script>
    """

    with pytest.raises(AssertionError, match="browser canary"):
        assert_html_renders(page, broken_html)
