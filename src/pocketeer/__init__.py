"""Pocketeer: A minimal, fpocket-style pocket finder in pure Python/Numpy."""

__version__ = "0.2.0"

# Public API

from .api import find_pockets
from .core import AlphaSphere, Pocket
from .utils import (
    load_structure,
    write_individual_pocket_jsons,
    write_pockets_as_pdb,
    write_pockets_json,
    write_summary,
)

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
