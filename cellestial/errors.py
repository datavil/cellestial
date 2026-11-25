"""
Errors and warnings raised by cellestial.

Every error subclasses `CellestialError` and the matching builtin exception, so
`except CellestialError` catches anything cellestial raised and `except KeyError`
(or `ValueError`, `TypeError`, ...) keeps working.
"""

from cellestial.util.errors import (
    AmbiguousColumnError,
    AmbiguousVariableError,
    CellestialError,
    CellestialWarning,
    ConflictingLengthError,
    DendrogramMismatchError,
    DuplicateKeysError,
    EmptySelectionError,
    GridIndexError,
    IncompleteDownloadError,
    InvalidArgumentError,
    InvalidArgumentTypeError,
    InvalidColorError,
    InvalidComparisonError,
    InvalidPlotSpecError,
    KeyNotFoundError,
    MissingAestheticError,
    MissingPlotDataError,
    NonCategoricalColumnError,
    UnknownParameterError,
    UnsupportedDataTypeError,
    UnsupportedGeometryError,
    VariableNotFoundError,
)

__all__ = [
    "AmbiguousColumnError",
    "AmbiguousVariableError",
    "CellestialError",
    "CellestialWarning",
    "ConflictingLengthError",
    "DendrogramMismatchError",
    "DuplicateKeysError",
    "EmptySelectionError",
    "GridIndexError",
    "IncompleteDownloadError",
    "InvalidArgumentError",
    "InvalidArgumentTypeError",
    "InvalidColorError",
    "InvalidComparisonError",
    "InvalidPlotSpecError",
    "KeyNotFoundError",
    "MissingAestheticError",
    "MissingPlotDataError",
    "NonCategoricalColumnError",
    "UnknownParameterError",
    "UnsupportedDataTypeError",
    "UnsupportedGeometryError",
    "VariableNotFoundError",
]
