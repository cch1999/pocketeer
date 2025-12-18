"""Pocket creation, scoring, and management."""

import logging

import biotite.structure as struc  # type: ignore
import numpy as np

from ..utils.constants import MAX_RADIUS_THRESHOLD, MIN_RADIUS_THRESHOLD, VOXEL_SIZE
from .geometry import compute_voxel_volume
from .types import AlphaSphere, Pocket

logger = logging.getLogger("pocketeer.pocket")


def extract_pocket_residues(
    spheres: list[AlphaSphere],
    atomarray: struc.AtomArray,
) -> list[tuple[str, int, str]]:
    """Extract unique residues associated with pocket spheres.

    Collects all atom indices from all spheres, maps them to residues
    using the AtomArray, and returns a deduplicated, sorted list of
    unique residues.

    Args:
        spheres: list of alpha-spheres in the pocket
        atomarray: Biotite AtomArray with structure data

    Returns:
        Sorted list of unique residues as (chain_id, res_id, res_name) tuples
    """
    # Collect all atom indices from all spheres
    all_atom_indices = set()
    for sphere in spheres:
        all_atom_indices.update(sphere.atom_indices)

    # Map atom indices to residues
    residue_set = set()
    for atom_idx in all_atom_indices:
        if atom_idx < len(atomarray):
            chain_id = str(atomarray.chain_id[atom_idx])
            res_id = int(atomarray.res_id[atom_idx])
            res_name = str(atomarray.res_name[atom_idx])
            residue_set.add((chain_id, res_id, res_name))

    # Return sorted list for consistent ordering
    return sorted(residue_set)


def create_residue_mask(
    residues: list[tuple[str, int, str]],
    atomarray: struc.AtomArray,
) -> np.ndarray:
    """Create a boolean mask for atoms belonging to pocket residues.

    Args:
        residues: List of residues as (chain_id, res_id, res_name) tuples
        atomarray: Biotite AtomArray with structure data

    Returns:
        Boolean numpy array where True indicates the atom belongs to a residue in the pocket
    """
    # Create string identifiers for fast vectorized lookup
    # Format: "chain_id:res_id:res_name" for efficient comparison
    residue_ids = {f"{cid}:{rid}:{rname}" for cid, rid, rname in residues}

    # Extract arrays and create identifiers for all atoms using vectorized operations
    chain_ids = atomarray.chain_id.astype(str)
    res_ids = atomarray.res_id.astype(int).astype(str)
    res_names = atomarray.res_name.astype(str)

    # Create identifiers for all atoms using vectorized string concatenation
    atom_ids = np.char.add(np.char.add(chain_ids, ":"), np.char.add(res_ids, ":"))
    atom_ids = np.char.add(atom_ids, res_names)

    # Vectorized membership check using numpy's isin (highly optimized)
    mask = np.isin(atom_ids, list(residue_ids))

    return mask


def score_pocket(
    pocket: Pocket,
) -> float:
    """Compute druggability/quality score for a pocket.

    Simple linear scoring based on size metrics.

    Args:
        pocket: pocket to score

    Returns:
        Score (higher is better)
    """
    # Simple scoring: volume + sphere count
    score = 0.0

    # Volume contribution (normalize to typical pocket size ~500 Å³)
    score += pocket.volume / 500.0

    # Sphere count contribution
    score += pocket.n_spheres / 50.0

    # Average radius (prefer moderate-sized spheres)
    avg_radius = np.mean([s.radius for s in pocket.spheres])
    if MIN_RADIUS_THRESHOLD <= avg_radius <= MAX_RADIUS_THRESHOLD:
        score += 2

    return score


def create_pocket(
    pocket_id: int,
    pocket_spheres: list[AlphaSphere],
    atomarray: struc.AtomArray,
    original_atomarray: struc.AtomArray | None = None,
) -> Pocket:
    """Create a Pocket object with computed descriptors.

    Args:
        pocket_id: unique pocket identifier
        pocket_spheres: list of spheres in this pocket
        atomarray: Filtered AtomArray to extract residue information
        original_atomarray: Original AtomArray before filtering for mask creation.
            If None, mask will be created from atomarray.

    Returns:
        Pocket object with descriptors
    """
    # Compute volume - use indices 0 to len(pocket_spheres)-1
    sphere_indices = set(range(len(pocket_spheres)))
    volume = compute_voxel_volume(sphere_indices, pocket_spheres, VOXEL_SIZE)

    # Compute centroid
    centers = np.array([sphere.center for sphere in pocket_spheres])
    centroid = centers.mean(axis=0).astype(np.float64)

    # Extract residues from filtered atomarray
    residues = extract_pocket_residues(pocket_spheres, atomarray)

    # Create residue mask for easy selection
    # Use original atomarray if provided, otherwise use filtered atomarray
    mask_atomarray = original_atomarray if original_atomarray is not None else atomarray
    mask = create_residue_mask(residues, mask_atomarray)

    # Create pocket
    pocket = Pocket(
        pocket_id=pocket_id,
        spheres=pocket_spheres,
        centroid=centroid,
        volume=volume,
        score=0.0,  # computed after creation
        residues=residues,
        mask=mask,
    )

    # Score the pocket
    pocket.score = score_pocket(pocket)

    return pocket
