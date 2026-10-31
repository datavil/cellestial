from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ggrust.plot.core import PlotSpec
from ggrust.plot.subplots import SupPlotsSpec
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

if TYPE_CHECKING:
    from collections.abc import Collection

_PLOT_SELECTOR = ".ggrust-plot > svg"
# Tooltips are drawn into the overlay div next to the plot SVG (not the toolbar).
_TOOLTIP_CELL_SELECTOR = ".ggrust-plot > div:not(.ggrust-toolbar) div:not(:has(div))"
_VISIBLE_ERROR_PATTERN = re.compile(
    r"internal error|uncaught exception|error (?:building|rendering) plot",
    re.IGNORECASE,
)


def _browser_errors(page: Page) -> list[str]:
    """Capture uncaught exceptions and console errors for one browser page."""
    errors: list[str] = []

    def record_console_error(message) -> None:
        if message.type == "error":
            errors.append(f"console: {message.text}")

    page.on("pageerror", lambda error: errors.append(f"page: {error}"))
    page.on("console", record_console_error)
    return errors


def _assert_no_browser_errors(page: Page, errors: list[str]) -> None:
    """Fail for JavaScript errors or an error rendered into the page."""
    page.wait_for_timeout(50)
    body_text = page.locator("body").inner_text()
    visible_error = _VISIBLE_ERROR_PATTERN.search(body_text)
    if errors or visible_error is not None:
        details = "\n".join(errors) if errors else visible_error.group(0)
        message = f"ggrust browser rendering failed:\n{details}\n{body_text}"
        raise AssertionError(message)


def assert_html_renders(page: Page, html: str) -> list[str]:
    """Assert that HTML creates a visible ggrust SVG without browser errors."""
    errors = _browser_errors(page)
    page.set_content(html, wait_until="load")

    try:
        page.locator(_PLOT_SELECTOR).first.wait_for(state="visible", timeout=10_000)
    except PlaywrightTimeoutError as error:
        _assert_no_browser_errors(page, errors)
        body_text = page.locator("body").inner_text()
        message = f"ggrust did not render a visible SVG:\n{body_text}"
        raise AssertionError(message) from error

    _assert_no_browser_errors(page, errors)
    return errors


def assert_plot_renders(
    page: Page,
    plot: PlotSpec | SupPlotsSpec,
    expected_type: type[PlotSpec] | type[SupPlotsSpec],
) -> list[str]:
    """Assert both the Python return type and real browser rendering."""
    assert isinstance(plot, expected_type)
    # ggrust inlines its JavaScript runtime, so the HTML is already offline.
    return assert_html_renders(page, plot.to_html())


def assert_point_tooltip(
    page: Page,
    errors: list[str],
    *,
    expected_values: Collection[str],
) -> None:
    """Hover the first point and assert that its tooltip is usable."""
    point = page.locator(f"{_PLOT_SELECTOR} circle").first
    point.wait_for(state="visible")
    point.hover(force=True)

    tooltip_text = page.locator(f"{_TOOLTIP_CELL_SELECTOR}:visible")
    tooltip_text.first.wait_for(state="visible")
    rendered_text = tooltip_text.all_text_contents()
    assert any(text in expected_values for text in rendered_text), rendered_text
    _assert_no_browser_errors(page, errors)
