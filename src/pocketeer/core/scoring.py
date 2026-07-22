"""Pocket creation, scoring, and management."""

import logging
from dataclasses import replace

import biotite.structure as struc  # type: ignore
import numpy as np

from ..utils.constants import (
    SCORE_AROMATIC_RESIDUES,
    SCORE_LR_BIAS,
    SCORE_LR_COEF,
    SCORE_LR_MEAN,
    SCORE_LR_STD,
    VOXEL_SIZE,
)
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
    atom_residue_ids: np.ndarray | None = None,
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

    if atom_residue_ids is None:
        atom_residue_ids = create_atom_residue_ids(atomarray)

    # Vectorized membership check using numpy's isin (highly optimized)
    mask = np.isin(atom_residue_ids, list(residue_ids))

    return mask


def create_atom_residue_ids(atomarray: struc.AtomArray) -> np.ndarray:
    """Create reusable residue identifiers for every atom in an array."""
    chain_ids = atomarray.chain_id.astype(str)
    res_ids = atomarray.res_id.astype(int).astype(str)
    res_names = atomarray.res_name.astype(str)
    atom_ids = np.char.add(np.char.add(chain_ids, ":"), np.char.add(res_ids, ":"))
    return np.char.add(atom_ids, res_names)


def score_pocket(
    pocket: Pocket,
) -> float:
    """Compute a druggability/ligandability score for a pocket in [0, 1].

    Uses a logistic-regression classifier trained on the P2Rank ``chen11``
    benchmark (ligand-bound proteins) to predict whether a pocket coincides with
    a real ligand-binding site. On the ``coach420`` evaluation set this ranks
    the true site first ~87% of the time when detected (ROC-AUC ~0.85), versus
    ~72% for the previous size-based heuristic.

    The six features are all derived from the ``Pocket`` itself:

    - number of lining residues
    - number of alpha-spheres
    - volume per sphere (compactness)
    - mean sphere radius
    - mean sphere SASA (buriedness)
    - fraction of aromatic lining residues

    Args:
        pocket: pocket to score

    Returns:
        Score in [0, 1] (higher is more likely to be a ligand-binding site)
    """
    radii = np.array([sphere.radius for sphere in pocket.spheres])
    sasa = np.array([sphere.mean_sasa for sphere in pocket.spheres])
    res_names = [residue[2] for residue in pocket.residues]
    frac_aromatic = (
        float(np.mean([name in SCORE_AROMATIC_RESIDUES for name in res_names]))
        if res_names
        else 0.0
    )

    features = np.array(
        [
            len(res_names),
            pocket.n_spheres,
            pocket.volume / max(pocket.n_spheres, 1),
            float(radii.mean()),
            float(sasa.mean()),
            frac_aromatic,
        ]
    )

    # Standardise, apply logistic-regression weights, squash to (0, 1).
    z = float((features - SCORE_LR_MEAN) / SCORE_LR_STD @ SCORE_LR_COEF) + SCORE_LR_BIAS
    return float(1.0 / (1.0 + np.exp(-z)))


def _create_pocket_from_components(
    pocket_id: int,
    spheres: list[AlphaSphere],
    residues: list[tuple[str, int, str]],
    mask: np.ndarray,
) -> Pocket:
    """Create a Pocket from precomputed components (internal helper).

    Used when spheres, residues, and mask are already known.
    Computes volume, centroid, and score.

    Args:
        pocket_id: unique pocket identifier
        spheres: list of spheres in this pocket
        residues: list of residues as (chain_id, res_id, res_name) tuples
        mask: boolean mask for selecting atoms in pocket residues

    Returns:
        Pocket object with computed volume, centroid, and score
    """
    # Compute volume
    sphere_indices = set(range(len(spheres)))
    volume = compute_voxel_volume(sphere_indices, spheres, VOXEL_SIZE)

    # Compute centroid
    centers = np.array([s.center for s in spheres])
    centroid = centers.mean(axis=0).astype(np.float64)

    # Create pocket
    pocket = Pocket(
        pocket_id=pocket_id,
        spheres=spheres,
        centroid=centroid,
        volume=volume,
        score=0.0,  # placeholder
        residues=residues,
        mask=mask,
    )

    # Score and return
    score = score_pocket(pocket)
    return replace(pocket, score=score)


def create_pocket(
    pocket_id: int,
    pocket_spheres: list[AlphaSphere],
    atomarray: struc.AtomArray,
    original_atomarray: struc.AtomArray | None = None,
    atom_residue_ids: np.ndarray | None = None,
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
    # Extract residues from filtered atomarray
    residues = extract_pocket_residues(pocket_spheres, atomarray)

    # Create residue mask for easy selection
    # Use original atomarray if provided, otherwise use filtered atomarray
    mask_atomarray = original_atomarray if original_atomarray is not None else atomarray
    mask = create_residue_mask(residues, mask_atomarray, atom_residue_ids)

    # Delegate to helper function
    return _create_pocket_from_components(pocket_id, pocket_spheres, residues, mask)
