class KeyNotFoundError(Exception):
    """Raised when a key is not found."""

    def __init__(self, message: str):
        super().__init__(message)


class ConfilictingLengthError(Exception):
    """Raised when sizes conflict."""

    def __init__(self, message: str):
        super().__init__(message)


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


class DuplicateKeysError(ValueError):
    """Raised when grouped keys (e.g. heatmap key groups) contain the same key in multiple groups."""

    def __init__(self, message: str):
        super().__init__(message)


class SpatialLibraryError(Exception):
    """Raised when spatial library metadata is missing, ambiguous, or refers to an unknown `library_id`."""

    def __init__(self, message: str):
        super().__init__(message)


class OptionalDependencyError(ImportError):
    """Raised when an optional third-party package required by a feature is not installed."""

    def __init__(self, message: str):
        super().__init__(message)
