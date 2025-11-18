"""Pocketeer: A minimal, fpocket-style pocket finder in pure Python/Numpy."""

__version__ = "0.1.0"

# Public API
# Optional visualization function
import contextlib

from .api import find_pockets
from .core import AlphaSphere, Pocket
from .utils import (
    load_structure,
    write_individual_pocket_jsons,
    write_pockets_as_pdb,
    write_pockets_json,
    write_summary,
)

# Suppress Atomworks import messages about env variables
with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
    try:
        import atomworks  # type: ignore[import-untyped] # noqa: F401

        _ATOMWORKS_AVAILABLE = True
    except ImportError:
        _ATOMWORKS_AVAILABLE = False

__all__ = [
    "AlphaSphere",
    "Pocket",
    "__version__",
    "find_pockets",
    "load_structure",
    "write_individual_pocket_jsons",
    "write_pockets_as_pdb",
    "write_pockets_json",
    "write_summary",
]

# Add visualization function to __all__ if available
try:
    from .vis import view_pockets  # noqa: F401

    __all__.append("view_pockets")
except (ImportError, ModuleNotFoundError):
    # If vis module can't be imported, view_pockets won't be available
    pass
