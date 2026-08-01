from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING

from lets_plot.plot.core import PlotSpec
from lets_plot.plot.subplots import SupPlotsSpec
from playwright.sync_api import Page
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

if TYPE_CHECKING:
    from collections.abc import Collection

_LIBRARY_SCRIPT_PATTERN = re.compile(
    r'<script[^>]+data-lets-plot-script="library"[^>]*>.*?</script>',
    re.DOTALL,
)
_PREPROCESS_ERROR_PATTERN = re.compile(r'"__error_message"\s*:\s*("(?:\\.|[^"\\])*")')
_VISIBLE_ERROR_PATTERN = re.compile(
    r"internal error|uncaught exception|error (?:building|rendering) plot",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _lets_plot_javascript() -> str:
    """Read the JavaScript bundle matching the installed Lets-Plot package."""
    return (
        files("lets_plot").joinpath("package_data", "lets-plot.min.js").read_text(encoding="utf-8")
    )


def _offline_html(plot: PlotSpec | SupPlotsSpec) -> str:
    """Generate standalone plot HTML without a CDN dependency."""
    html = plot.to_html()
    error_match = _PREPROCESS_ERROR_PATTERN.search(html)
    if error_match is not None:
        error_message = json.loads(error_match.group(1))
        message = f"Lets-Plot preprocessing failed: {error_message}"
        raise AssertionError(message)

    library_script = f"<script>{_lets_plot_javascript()}</script>"
    html, replacement_count = _LIBRARY_SCRIPT_PATTERN.subn(
        lambda _match: library_script,
        html,
        count=1,
    )
    if replacement_count != 1:
        message = "Lets-Plot HTML did not contain its JavaScript library tag"
        raise AssertionError(message)
    return html


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
        message = f"Lets-Plot browser rendering failed:\n{details}\n{body_text}"
        raise AssertionError(message)


def assert_html_renders(page: Page, html: str) -> list[str]:
    """Assert that HTML creates a visible Lets-Plot SVG without browser errors."""
    errors = _browser_errors(page)
    page.set_content(html, wait_until="load")

    try:
        page.locator("svg.plt-container").first.wait_for(state="visible", timeout=10_000)
    except PlaywrightTimeoutError as error:
        _assert_no_browser_errors(page, errors)
        body_text = page.locator("body").inner_text()
        message = f"Lets-Plot did not render a visible SVG:\n{body_text}"
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
    return assert_html_renders(page, _offline_html(plot))


def assert_point_tooltip(
    page: Page,
    errors: list[str],
    *,
    expected_values: Collection[str],
) -> None:
    """Hover the first point and assert that its tooltip is usable."""
    point = page.locator("svg.plt-container circle").first
    point.wait_for(state="visible")
    point.hover(force=True)

    tooltip_text = page.locator(".tooltip-text:visible")
    tooltip_text.first.wait_for(state="visible")
    rendered_text = tooltip_text.all_text_contents()
    assert any(text in expected_values for text in rendered_text), rendered_text
    _assert_no_browser_errors(page, errors)
