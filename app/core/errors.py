class VectorDBConnectionError(RuntimeError):
    """Raised when the vector database cannot be reached or does not respond as expected."""


class VectorDBQueryError(RuntimeError):
    """Raised when the vector database rejects a search request."""
