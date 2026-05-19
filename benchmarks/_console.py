"""Small console helpers for nicely formatted progress prints."""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[31m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN = "\033[36m"


def _paint(text: str, *codes: str) -> str:
    if not _USE_COLOR or not codes:
        return text
    return "".join(codes) + text + _RESET


def banner(text: str) -> None:
    """Print a big section banner."""
    line = "=" * max(len(text) + 4, 60)
    print()
    print(_paint(line, _BOLD, _CYAN))
    print(_paint(f"  {text}", _BOLD, _CYAN))
    print(_paint(line, _BOLD, _CYAN))


def section(text: str) -> None:
    """Print a smaller section header."""
    print()
    print(_paint(f"==> {text}", _BOLD, _BLUE))


def step(text: str) -> None:
    """Print an indented step line."""
    print(f"    {_paint('·', _DIM)} {text}")


def ok(text: str) -> None:
    print(f"    {_paint('✓', _GREEN)} {text}")


def warn(text: str) -> None:
    print(f"    {_paint('!', _YELLOW)} {text}")


def fail(text: str) -> None:
    print(f"    {_paint('✗', _RED)} {text}")


def info(text: str) -> None:
    print(f"    {_paint('i', _MAGENTA)} {text}")


def kv(key: str, value: object) -> None:
    """Print an indented key/value pair."""
    print(f"      {_paint(key + ':', _DIM)} {value}")


@contextmanager
def stage(label: str) -> Iterator[None]:
    """Run a block, printing start + elapsed seconds + status on exit."""
    start = time.perf_counter()
    print(f"    {_paint('»', _CYAN)} {label} ...", flush=True)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        fail(f"{label} failed after {elapsed:.2f}s: {exc.__class__.__name__}: {exc}")
        raise
    elapsed = time.perf_counter() - start
    ok(f"{label} done in {elapsed:.2f}s")


def render_seconds(value: float) -> str:
    if value < 1e-3:
        return f"{value * 1e6:.0f}µs"
    if value < 1.0:
        return f"{value * 1e3:.1f}ms"
    return f"{value:.2f}s"
