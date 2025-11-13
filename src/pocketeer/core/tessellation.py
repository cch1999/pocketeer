"""Alpha-sphere computation, filtering, and sphere-specific operations."""

import numpy as np
import numpy.typing as npt
from scipy.spatial import (
    Delaunay,
    cKDTree,  # type: ignore
)

from ..utils.constants import MIN_ATOMS_FOR_TESSELLATION
from ..utils.exceptions import TessellationError
from .geometry import circumsphere, is_sphere_empty
from .types import AlphaSphere


def compute_alpha_spheres(
    coords: npt.NDArray[np.float64],
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
) -> list[AlphaSphere]:
    """Compute alpha-spheres from Delaunay tessellation.

    Args:
        coords: (N, 3) atom coordinates
        r_min: Minimum alpha-sphere radius (Å)
        r_max: Maximum alpha-sphere radius (Å)

    Returns:
        List of valid AlphaSphere objects
    """
    if len(coords) < MIN_ATOMS_FOR_TESSELLATION:
        return []

    # Compute Delaunay tessellation
    try:
        tri = Delaunay(coords)
    except (ValueError, RuntimeError) as e:
        # Handle cases where tessellation fails (e.g., degenerate points)
        raise TessellationError(f"Delaunay tessellation failed: {e}") from e

    # Build KD-tree once for fast sphere emptiness checks
    tree = cKDTree(coords)

    spheres = []

    # Process each tetrahedron
    for sphere_id, simplex in enumerate(tri.simplices):
        tet_points = coords[simplex]
        center, radius = circumsphere(tet_points)

        # Filter by radius
        if radius < r_min or radius > r_max:
            continue

        # Check if sphere is empty (alpha-shape criterion) - use fast version
        atom_indices = set(simplex.tolist())
        if not is_sphere_empty(center, radius, tree, atom_indices):
            continue

        # Placeholder for burial status - will be computed later
        spheres.append(
            AlphaSphere(
                sphere_id=sphere_id,
                center=center,
                radius=radius,
                is_buried=True,  # will be determined in next step
                atom_indices=simplex.tolist(),
            )
        )

    return spheres


def label_polarity(
    spheres: list[AlphaSphere],
    coords: npt.NDArray[np.float64],
    polar_probe_radius: float,
) -> list[AlphaSphere]:
    """Label spheres as buried (apolar) or surface (polar).

    Modifies spheres in place.

    Args:
        spheres: list of alpha-spheres
        coords: (N, 3) atom coordinates
        polar_probe_radius: Radius to test atom contact for polarity (Å)
    """

    # Build KD-tree for fast nearest-neighbor queries
    tree = cKDTree(coords)

    # For each sphere, check if it's within probe radius of any atom
    for sphere in spheres:
        # Query for atoms within probe radius
        indices = tree.query_ball_point(sphere.center, polar_probe_radius)

        # If close to atoms beyond the defining vertices, it's surface (polar)
        defining_atoms = set(sphere.atom_indices)
        close_atoms = set(indices) - defining_atoms

        sphere.is_buried = len(close_atoms) == 0

    return spheres


def filter_surface_spheres(
    spheres: list[AlphaSphere],
) -> list[AlphaSphere]:
    """Filter to keep only buried (apolar) spheres for pocket detection.

    Args:
        spheres: list of all alpha-spheres

    Returns:
        List of buried spheres only
    """
    return [s for s in spheres if s.is_buried]
