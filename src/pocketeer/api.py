"""High-level API for pocket detection."""

import logging

import biotite.structure as struc  # type: ignore

from .core import (
    Pocket,
    cluster_spheres,
    compute_alpha_spheres,
    create_pocket,
    filter_surface_spheres,
    label_polarity,
)
from .utils.constants import MIN_ATOMS_FOR_TESSELLATION
from .utils.exceptions import ValidationError

logger = logging.getLogger("pocketeer.api")


def find_pockets(
    atomarray: struc.AtomArray,
    *,
    r_min: float = 3.0,
    r_max: float = 6.0,
    polar_probe_radius: float = 1.4,
    sasa_threshold: float = 20.0,
    merge_distance: float = 1.75,
    min_spheres: int = 35,
    ignore_hydrogens: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True,
) -> list[Pocket]:
    """Detect binding pockets in a protein structure.

    Main API function for pocket detection using alpha-sphere method.

    The algorithm works as follows:
        1. Compute all alpha-spheres from Delaunay tessellation
        2. Label spheres as buried (interior) or surface and filter to buried ones
        3. Cluster buried spheres into pockets using graph clustering
        4. Create Pocket objects with volume and scoring descriptors

    Args:
        atomarray: Biotite AtomArray with structure data
        r_min: Minimum alpha-sphere radius (Å). Default: 3.0
        r_max: Maximum alpha-sphere radius (Å). Default: 6.0
        polar_probe_radius: Probe radius for SASA calculation (Å). Default: 1.4
        sasa_threshold: Threshold for mean SASA value to determine if a sphere is buried (Å²).
            Spheres with mean SASA below this threshold are considered buried (interior) and
            kept for pocket detection. Higher values include more surface-exposed spheres.
            Typical range: 15-30 Å². Default: 20.0
        merge_distance: Distance threshold for merging nearby sphere clusters (Å). Default: 1.75
        min_spheres: Minimum number of spheres per pocket cluster. Default: 35
        ignore_hydrogens: Ignore hydrogen atoms (recommended). Default: True
        ignore_water: Ignore water molecules (recommended). Default: True
        ignore_hetero: Ignore hetero atoms i.e. ligands (recommended). Default: True

    Returns:
        pockets: list of detected Pocket objects, sorted by score (highest first)

    Raises:
        ValidationError: if atomarray is invalid or empty
        ValueError: if parameters are invalid

    Examples:
        >>> import pocketeer as pt
        >>> atomarray = pt.load_structure("protein.pdb")
        >>> pockets = pt.find_pockets(atomarray)

        >>> # With custom parameters
        >>> pockets = pt.find_pockets(atomarray, r_min=2.5, r_max=7.0, min_spheres=25)
    """
    # Parameter validation
    if r_min <= 0 or r_max <= r_min:
        raise ValueError(f"Invalid radius range: r_min={r_min}, r_max={r_max}")
    if polar_probe_radius <= 0:
        raise ValueError(f"polar_probe_radius must be > 0, got {polar_probe_radius}")
    if sasa_threshold <= 0:
        raise ValueError(f"sasa_threshold must be > 0, got {sasa_threshold}")

    # Input validation
    if not isinstance(atomarray, struc.AtomArray):
        raise ValidationError(f"Expected AtomArray, got {type(atomarray).__name__}")

    if len(atomarray) == 0:
        raise ValidationError("AtomArray is empty")

    # Store original atomarray for mask creation (before filtering)
    original_atomarray = atomarray

    if ignore_hydrogens:
        atomarray = atomarray[atomarray.element != "H"]

    if ignore_water:
        atomarray = atomarray[atomarray.res_name != "HOH"]

    if ignore_hetero:
        atomarray = atomarray[~atomarray.hetero]

    if len(atomarray) < MIN_ATOMS_FOR_TESSELLATION:
        logger.warning(f"Not enough atoms ({len(atomarray)}). Returning empty result.")
        return []

    # 1. Compute all alpha-spheres from Delaunay tessellation
    alpha_spheres = compute_alpha_spheres(atomarray.coord, r_min=r_min, r_max=r_max)
    logger.info(f"Computed {len(alpha_spheres)} alpha-spheres")

    # 2. Label spheres as buried or surface and filter
    alpha_spheres = label_polarity(alpha_spheres, atomarray, polar_probe_radius)
    buried_spheres = filter_surface_spheres(alpha_spheres, sasa_threshold=sasa_threshold)
    logger.info(f"Found {len(buried_spheres)} buried spheres")

    if not buried_spheres:
        logger.warning("No buried spheres found. Returning empty pocket list.")
        return []

    # 3. Cluster buried spheres into pockets
    pocket_clusters = cluster_spheres(
        buried_spheres, merge_distance=merge_distance, min_spheres=min_spheres
    )
    logger.info(f"Found {len(pocket_clusters)} clusters (potential pockets)")

    if not pocket_clusters:
        logger.warning("No clusters found. Returning empty pocket list.")
        return []

    # Create Pocket objects with descriptors
    # Pass filtered atomarray for residue extraction, original for mask creation
    pockets = [
        create_pocket(pocket_id, cluster, atomarray, original_atomarray)
        for pocket_id, cluster in enumerate(pocket_clusters)
    ]

    # Sort by score (descending)
    pockets.sort(key=lambda p: p.score, reverse=True)
    logger.info(f"Pocket detection complete. {len(pockets)} pockets found")

    return pockets
