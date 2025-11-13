"""Custom exceptions for pocketeer."""


class PocketeerError(Exception):
    """Base exception for all pocketeer errors."""

    pass


class ValidationError(PocketeerError):
    """Raised when input validation fails."""

    pass


class TessellationError(PocketeerError):
    """Raised when Delaunay tessellation fails."""

    pass


class GeometryError(PocketeerError):
    """Raised when geometric computations fail."""

    pass
