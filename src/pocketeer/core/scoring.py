"""Pocket creation, scoring, and management."""

import logging

import numpy as np

from ..utils.constants import MAX_RADIUS_THRESHOLD, MIN_RADIUS_THRESHOLD, VOXEL_SIZE
from .geometry import compute_voxel_volume
from .types import AlphaSphere, Pocket

logger = logging.getLogger("pocketeer.pocket")


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
) -> Pocket:
    """Create a Pocket object with computed descriptors.

    Args:
        pocket_id: unique pocket identifier
        pocket_spheres: list of spheres in this pocket

    Returns:
        Pocket object with descriptors
    """
    # Compute volume - use indices 0 to len(pocket_spheres)-1
    sphere_indices = set(range(len(pocket_spheres)))
    volume = compute_voxel_volume(sphere_indices, pocket_spheres, VOXEL_SIZE)

    # Compute centroid
    centers = np.array([sphere.center for sphere in pocket_spheres])
    centroid = centers.mean(axis=0).astype(np.float64)

    # Create pocket
    pocket = Pocket(
        pocket_id=pocket_id,
        spheres=pocket_spheres,
        centroid=centroid,
        volume=volume,
        score=0.0,  # computed after creation
    )

    # Score the pocket
    pocket.score = score_pocket(pocket)

    return pocket
