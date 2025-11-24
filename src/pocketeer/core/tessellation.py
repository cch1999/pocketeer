"""Alpha-sphere computation, filtering, and sphere-specific operations."""

import biotite.structure as struc  # type: ignore
import numpy as np
import numpy.typing as npt
from biotite.structure import sasa  # type: ignore
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
                mean_sasa=0.0,  # will be determined in next step
                atom_indices=simplex.tolist(),
            )
        )

    return spheres


def label_polarity(
    spheres: list[AlphaSphere],
    atomarray: struc.AtomArray,
    polar_probe_radius: float,
) -> list[AlphaSphere]:
    """Calculate mean SASA values for spheres using their defining atoms.

    Uses Solvent Accessible Surface Area (SASA) to compute the mean SASA
    of the 4 atoms that define each alpha-sphere. The SASA values are stored
    in the sphere's mean_sasa attribute for later filtering.

    Modifies spheres in place.

    Args:
        spheres: list of alpha-spheres
        atomarray: Biotite AtomArray with structure data
        polar_probe_radius: Probe radius for SASA calculation (Å)
    """
    # Calculate SASA once for all atoms
    sasa_values = sasa(atomarray, probe_radius=polar_probe_radius)

    # For each sphere, compute mean SASA of its 4 defining atoms
    for sphere in spheres:
        # Get SASA values for the atoms that define this sphere
        defining_sasa = sasa_values[sphere.atom_indices]
        sphere.mean_sasa = float(np.mean(defining_sasa))

    return spheres


def filter_surface_spheres(
    spheres: list[AlphaSphere],
    sasa_threshold: float = 20.0,
) -> list[AlphaSphere]:
    """Filter to keep only buried (interior) spheres for pocket detection.

    Uses the mean SASA (solvent accessible surface area) of the 4 atoms defining
    each sphere to determine if it's buried or surface-exposed. Spheres with
    mean SASA below the threshold are considered buried and kept for pocket detection.

    Args:
        spheres: list of all alpha-spheres with computed mean_sasa values
        sasa_threshold: Threshold for mean SASA value to determine if a sphere is buried (Å²).
            Spheres with mean_sasa < sasa_threshold are kept. Typical values: 15-30 Å².

    Returns:
        List of buried spheres only (those with mean_sasa < sasa_threshold)
    """
    return [s for s in spheres if s.mean_sasa < sasa_threshold]
