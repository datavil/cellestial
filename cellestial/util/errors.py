from __future__ import annotations

from collections.abc import Sequence
from difflib import get_close_matches
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class CellestialError(Exception):
    """
    Base class of every error raised by cellestial.

    Notes
    -----
    Each cellestial error also subclasses the matching builtin exception, so
    `except KeyError` keeps working while `except CellestialError` catches any
    error cellestial raised.

    Examples
    --------
    >>> import cellestial as cl
    >>> try:
    ...     cl.umap(data, "leidn")
    ... except cl.errors.CellestialError as error:
    ...     print(error)
    """


class _MessageKeyError(KeyError):
    """A `KeyError` whose message prints as written, not wrapped in quotes."""

    def __init__(self, message: str):
        super().__init__(message)

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


class KeyNotFoundError(_MessageKeyError, CellestialError):
    """Raised when a key is not found."""


class ConflictingLengthError(ValueError, CellestialError):
    """Raised when input lengths cannot be broadcast together."""


class UnsupportedDataTypeError(TypeError, CellestialError):
    """Raised when the supplied data object is of a type the function does not handle."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidPlotSpecError(TypeError, CellestialError):
    """Raised when a `PlotSpec` or `SupPlotsSpec` was expected but a different type was given."""

    def __init__(self, message: str):
        super().__init__(message)


class MissingAestheticError(ValueError, CellestialError):
    """Raised when a required aesthetic is absent from both arguments and the plot mapping."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidComparisonError(ValueError, CellestialError):
    """Raised when pairwise comparison input is malformed or references unknown groups."""

    def __init__(self, message: str):
        super().__init__(message)


class VariableNotFoundError(_MessageKeyError, CellestialError):
    """Raised when a requested variable (gene / feature) name is not present in the data."""


class AmbiguousVariableError(ValueError, CellestialError):
    """Raised when a variable (gene / feature) name is present in more than one modality."""

    def __init__(self, message: str):
        super().__init__(message)


class AmbiguousColumnError(ValueError, CellestialError):
    """Raised when a column name could mean more than one column (e.g. a container's and a modality's)."""

    def __init__(self, message: str):
        super().__init__(message)


class DuplicateKeysError(ValueError, CellestialError):
    """Raised when grouped keys (e.g. heatmap key groups) contain the same key in multiple groups."""

    def __init__(self, message: str):
        super().__init__(message)


class NonCategoricalColumnError(TypeError, CellestialError):
    """Raised when an operation on groups receives a column of continuous values instead of categories."""

    def __init__(self, message: str):
        super().__init__(message)


class EmptySelectionError(ValueError, CellestialError):
    """Raised when a selection of groups or keys is empty, so the result would show nothing selected."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidColorError(ValueError, CellestialError):
    """Raised when a color is not in a format the function can compute with (e.g. not a hex color)."""

    def __init__(self, message: str):
        super().__init__(message)


class DendrogramMismatchError(ValueError, CellestialError):
    """Raised when a stored dendrogram covers groups that differ from the groups present in the data."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidArgumentError(ValueError, CellestialError):
    """Raised when a required parameter is missing or a parameter value is not one of the accepted values."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidArgumentTypeError(TypeError, CellestialError):
    """Raised when a parameter receives a value of a type the function does not accept."""

    def __init__(self, message: str):
        super().__init__(message)


class UnknownParameterError(TypeError, CellestialError):
    """Raised when a keyword the function does not accept is passed, such as a renamed parameter."""

    def __init__(self, message: str):
        super().__init__(message)


class GridIndexError(IndexError, CellestialError):
    """Raised when a figure index is out of range for a grid of plots."""

    def __init__(self, message: str):
        super().__init__(message)


class IncompleteDownloadError(OSError, CellestialError):
    """Raised when a download ends before the size announced by the server."""

    def __init__(self, message: str):
        super().__init__(message)


class MissingPlotDataError(ValueError, CellestialError):
    """Raised when a plot carries no data frame to retrieve."""

    def __init__(self, message: str):
        super().__init__(message)


class CellestialWarning(UserWarning):
    """Base category for warnings raised by cellestial."""


def _unsupported_data_type(received: object, *expected: type | str) -> UnsupportedDataTypeError:
    """Build an `UnsupportedDataTypeError` naming the expected and received types."""
    # a type from an optional dependency is passed by name, so the error path never imports it.
    expected_names = " or ".join(cls if isinstance(cls, str) else cls.__name__ for cls in expected)
    received_name = type(received).__name__
    return UnsupportedDataTypeError(f"Expected {expected_names}, but received `{received_name}`.")


def _did_you_mean(keys: object, candidates: Iterable[object]) -> str:
    """
    Return a "did you mean" hint naming the closest `candidates` to each key.

    Notes
    -----
    Returns an empty string when nothing is close, so callers can append the
    result to a message unconditionally. With several keys, each hint names the
    key it belongs to. Non-string candidates are compared by their string form.
    """
    names = [str(candidate) for candidate in candidates]
    if isinstance(keys, str) or not isinstance(keys, Sequence):
        return _close_matches_sentence(str(keys), names)
    if len(keys) == 1:
        return _close_matches_sentence(str(keys[0]), names)
    hints = []
    for key in keys:
        hint = _close_matches_sentence(str(key), names)
        if hint:
            hints.append(f"For {key!r}: {hint[0].lower()}{hint[1:]}")
    return " ".join(hints)


def _close_matches_sentence(key: str, names: list[str]) -> str:
    """Return "Did you mean ...?" for the names closest to `key`, or an empty string."""
    matches = get_close_matches(key, names, n=3)
    if not matches:
        return ""
    if len(matches) == 1:
        return f"Did you mean {matches[0]!r}?"
    return f"Did you mean one of {matches!r}?"


def _ranking_not_found_message(ranking_key: str, stored_keys: list[str]) -> str:
    """Build the error message for a `ranking_key` that names no stored ranking."""
    hint = _close_matches_sentence(ranking_key, stored_keys)
    return (
        f"No precomputed ranking found under `ranking_key={ranking_key!r}`. "
        + (f"{hint} " if hint else "")
        + "Compute a differential expression ranking first, or pass the key it "
        "is stored under as `ranking_key`."
    )
