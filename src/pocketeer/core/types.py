"""Core data types for pocketeer."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class AlphaSphere:
    """Represents a single alpha-sphere from Delaunay tessellation.

    Note: sphere_id is a unique identifier, NOT the index in a list.
    Use list indices for lookups, sphere_id for identification/serialization.
    """

    sphere_id: int  # unique identifier (NOT list index)
    center: npt.NDArray[np.float64]  # 3D coordinates
    radius: float
    mean_sasa: float  # Mean SASA of the sphere's defining atoms
    atom_indices: list[int]  # indices of the 4 Delaunay vertices

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sphere_id": self.sphere_id,
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "mean_sasa": float(self.mean_sasa),
            "atom_indices": self.atom_indices,
        }


@dataclass
class Pocket:
    """Represents a detected pocket (cluster of alpha-spheres)."""

    pocket_id: int
    spheres: list[AlphaSphere]  # list of spheres in pocket
    centroid: npt.NDArray[np.float64]  # geometric center
    volume: float  # estimated volume in A³
    score: float  # druggability/quality score
    residues: list[tuple[str, int, str]]  # unique residues as (chain_id, res_id, res_name) tuples
    mask: npt.NDArray[np.bool_]  # boolean mask for selecting atoms in pocket residues

    @property
    def sphere_ids(self) -> list[int]:
        """Get sphere IDs (unique identifiers) of spheres in this pocket.

        Note: These are IDs, not list indices. Use pocket.spheres to access
        the actual sphere objects.
        """
        return [sphere.sphere_id for sphere in self.spheres]

    @property
    def n_spheres(self) -> int:
        """Number of spheres in pocket."""
        return len(self.spheres)

    def __repr__(self) -> str:
        """String representation of pocket."""
        return (
            f"Pocket(pocket_id={self.pocket_id}, volume={self.volume:.1f}, "
            f"score={self.score:.2f}, n_spheres={self.n_spheres}, "
            f"n_residues={len(self.residues)})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pocket_id": self.pocket_id,
            "centroid": self.centroid.tolist(),
            "volume": float(self.volume),
            "score": float(self.score),
            "n_spheres": self.n_spheres,
            "n_residues": len(self.residues),
            "residues": [
                {"chain_id": chain_id, "res_id": res_id, "res_name": res_name}
                for chain_id, res_id, res_name in self.residues
            ],
            "spheres": [sphere.to_dict() for sphere in self.spheres],
        }
