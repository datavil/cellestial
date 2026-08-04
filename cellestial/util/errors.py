class KeyNotFoundError(Exception):
    """Raised when a key is not found."""

    def __init__(self, message: str):
        super().__init__(message)


class ConflictingLengthError(ValueError):
    """Raised when input lengths cannot be broadcast together."""


class UnsupportedDataTypeError(TypeError):
    """Raised when the supplied data object is of a type the function does not handle."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidPlotSpecError(TypeError):
    """Raised when a `PlotSpec` or `SupPlotsSpec` was expected but a different type was given."""

    def __init__(self, message: str):
        super().__init__(message)


class MissingAestheticError(ValueError):
    """Raised when a required aesthetic is absent from both arguments and the plot mapping."""

    def __init__(self, message: str):
        super().__init__(message)


class InvalidComparisonError(ValueError):
    """Raised when pairwise comparison input is malformed or references unknown groups."""

    def __init__(self, message: str):
        super().__init__(message)


class VariableNotFoundError(Exception):
    """Raised when a requested variable (gene / feature) name is not present in the data."""

    def __init__(self, message: str):
        super().__init__(message)


class AmbiguousVariableError(ValueError):
    """Raised when a variable (gene / feature) name is present in more than one modality."""

    def __init__(self, message: str):
        super().__init__(message)


class DuplicateKeysError(ValueError):
    """Raised when grouped keys (e.g. heatmap key groups) contain the same key in multiple groups."""

    def __init__(self, message: str):
        super().__init__(message)


class SpatialLibraryError(Exception):
    """Raised when spatial library metadata is missing, ambiguous, or refers to an unknown `library_id`."""

    def __init__(self, message: str):
        super().__init__(message)


class CellestialWarning(UserWarning):
    """Base category for warnings raised by cellestial."""


def _unsupported_data_type(received: object, *expected: type) -> UnsupportedDataTypeError:
    """Build an `UnsupportedDataTypeError` naming the expected and received types."""
    expected_names = " or ".join(cls.__name__ for cls in expected)
    received_name = type(received).__name__
    return UnsupportedDataTypeError(f"Expected {expected_names}, but received `{received_name}`.")
