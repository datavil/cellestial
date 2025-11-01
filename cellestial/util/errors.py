class ConflictingKeysError(Exception):
    """Raised when keys conflict."""

    def __init__(self, message: str):
        super().__init__(message)

class KeyNotFoundError(Exception):
    """Raised when a key is not found."""

    def __init__(self, message: str):
        super().__init__(message)

class ConfilictingLengthError(Exception):
    """Raised when sizes conflict."""

    def __init__(self, message: str):
        super().__init__(message)