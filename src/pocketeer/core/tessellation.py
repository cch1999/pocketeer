"""Alpha-sphere computation, filtering, and sphere-specific operations."""

from dataclasses import replace

import biotite.structure as struc  # type: ignore
import numpy as np
import numpy.typing as npt
from biotite.structure import sasa  # type: ignore
from biotite.structure.info import vdw_radius_protor, vdw_radius_single  # type: ignore
from scipy.spatial import (
    Delaunay,
    cKDTree,  # type: ignore
)

from ..utils.constants import MIN_ATOMS_FOR_TESSELLATION
from ..utils.exceptions import TessellationError
from .geometry import circumsphere, is_sphere_empty
from .types import AlphaSphere

_GENERIC_PROTOR_ATOM_NAMES = {"C": "C", "N": "N", "O": "O"}
_SULFUR_PROTOR_ATOM_NAMES = {"CYS": "SG", "MET": "SD"}
_DEFAULT_VDW_RADIUS = 1.7


def _sanitize_for_sasa(atomarray: struc.AtomArray) -> tuple[struc.AtomArray, np.ndarray]:
    """Return a SASA-only copy with ProtOr-compatible atom names and radii.

    Experimental structures sometimes label modified atoms as if they belonged
    to a canonical amino acid, for example ``MET:SE`` or ``THR:P``. ProtOr uses
    the residue and atom names as a strict lookup key and raises ``KeyError`` for
    these combinations. Preserve the original array for every other operation,
    but sanitize invalid names on a copy and fall back to an element radius when
    the residue has no compatible ProtOr atom.
    """
    sanitized = atomarray.copy()
    radii = np.empty(len(sanitized), dtype=float)

    for index, (raw_res_name, raw_atom_name, raw_element) in enumerate(
        zip(sanitized.res_name, sanitized.atom_name, sanitized.element, strict=True)
    ):
        res_name = str(raw_res_name).upper()
        atom_name = str(raw_atom_name).upper()
        element = str(raw_element).upper()
        try:
            radius = vdw_radius_protor(res_name, atom_name)
        except (KeyError, ValueError):
            replacement = (
                _SULFUR_PROTOR_ATOM_NAMES.get(res_name)
                if atom_name.startswith("S")
                else _GENERIC_PROTOR_ATOM_NAMES.get(atom_name[:1])
            )
            if replacement is not None:
                try:
                    radius = vdw_radius_protor(res_name, replacement)
                    sanitized.atom_name[index] = replacement
                except KeyError:
                    radius = None
            else:
                radius = None

        if radius is None:
            radius = vdw_radius_single(element)
        radii[index] = radius if radius is not None else _DEFAULT_VDW_RADIUS

    return sanitized, radii


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

    # Compute the circumspheres for all tetrahedra in one batched linear
    # algebra operation.  Calling np.linalg.solve() once per tetrahedron is
    # disproportionately expensive for the tens of thousands produced by a
    # typical protein.
    simplices = tri.simplices
    tet_points = coords[simplices]
    relative_points = tet_points[:, 1:, :] - tet_points[:, :1, :]
    matrices = 2.0 * relative_points
    right_hand_sides = np.einsum("nij,nij->ni", relative_points, relative_points)

    determinants = np.linalg.det(matrices)
    nondegenerate = determinants != 0.0
    relative_centers = np.zeros((len(simplices), 3), dtype=coords.dtype)
    try:
        relative_centers[nondegenerate] = np.linalg.solve(
            matrices[nondegenerate], right_hand_sides[nondegenerate]
        )
    except np.linalg.LinAlgError:
        # Extremely ill-conditioned input may still make a batched solve fail.
        # Preserve the previous per-tetrahedron behaviour for this rare case.
        for index in np.flatnonzero(nondegenerate):
            _, radius = circumsphere(tet_points[index])
            if radius == 0.0:
                nondegenerate[index] = False
                continue
            relative_centers[index] = np.linalg.solve(matrices[index], right_hand_sides[index])

    radii = np.linalg.norm(relative_centers, axis=1)
    candidate_indices = np.flatnonzero(nondegenerate & (radii >= r_min) & (radii <= r_max))
    centers = relative_centers + tet_points[:, 0, :]

    spheres = []

    # The remaining work is limited to radius-qualified tetrahedra.
    for sphere_id in candidate_indices:
        simplex = simplices[sphere_id]
        center = centers[sphere_id]
        radius = float(radii[sphere_id])

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
    # Calculate SASA once for all atoms. Sanitization is isolated to a copy so
    # atom indices and the annotations used by the detector remain unchanged.
    sasa_atomarray, vdw_radii = _sanitize_for_sasa(atomarray)
    sasa_values = sasa(
        sasa_atomarray,
        probe_radius=polar_probe_radius,
        vdw_radii=vdw_radii,
    )

    # For each sphere, compute mean SASA of its 4 defining atoms
    labeled_spheres = []
    for sphere in spheres:
        # Get SASA values for the atoms that define this sphere
        defining_sasa = sasa_values[sphere.atom_indices]
        mean_sasa = float(np.mean(defining_sasa))
        labeled_spheres.append(replace(sphere, mean_sasa=mean_sasa))

    return labeled_spheres


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
