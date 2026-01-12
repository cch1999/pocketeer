"""Pure geometric utilities for 3D computations."""

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree  # type: ignore

from ..utils.exceptions import GeometryError
from .types import AlphaSphere


def circumsphere(
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], float]:
    """Compute circumsphere of 4 points (tetrahedron).

    Args:
        points: (4, 3) array of 3D coordinates

    Returns:
        center: (3,) center coordinates
        radius: radius of circumsphere
    """
    if points.shape != (4, 3):
        raise GeometryError(f"Expected (4, 3) points, got {points.shape}")

    # Use direct formula for tetrahedron circumsphere
    # Place first point at origin
    p0 = points[0]
    p1 = points[1] - p0
    p2 = points[2] - p0
    p3 = points[3] - p0

    # Build linear system
    A = 2.0 * np.array([p1, p2, p3])
    b = np.array(
        [
            np.dot(p1, p1),
            np.dot(p2, p2),
            np.dot(p3, p3),
        ]
    )

    try:
        # Solve for center relative to p0
        center_rel = np.linalg.solve(A, b)
        center = center_rel + p0
        radius = np.linalg.norm(center_rel)
        return center, float(radius)
    except np.linalg.LinAlgError:
        # Degenerate tetrahedron
        return np.mean(points, axis=0), 0.0


def is_sphere_empty(
    center: npt.NDArray[np.float64],
    radius: float,
    tree: cKDTree,
    exclude_indices: set[int],
    tolerance: float = 1e-6,
) -> bool:
    """Fast sphere emptiness check using KD-tree radius query.

    This is much faster than is_sphere_empty for large structures because
    it only checks atoms within the sphere radius instead of all atoms.

    Args:
        center: sphere center
        radius: sphere radius
        tree: pre-built KD-tree of all atom coordinates
        exclude_indices: indices of atoms on the sphere surface (to ignore)
        tolerance: numerical tolerance for "on surface" check

    Returns:
        True if sphere is empty (only surface atoms inside)
    """
    # Query for atoms within sphere radius
    close_indices = tree.query_ball_point(center, radius)

    # Check if any non-excluded atoms are inside the sphere
    for idx in close_indices:
        if idx in exclude_indices:
            continue

        # Get exact distance to check if truly inside
        dist = np.linalg.norm(tree.data[idx] - center)
        if dist < radius - tolerance:
            return False

    return True


def bounding_box(
    coords: npt.NDArray[np.float64], padding: float = 0.0
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute axis-aligned bounding box of point set.

    Args:
        coords: (N, 3) coordinates
        padding: expand box by this amount on all sides

    Returns:
        min_corner: (3,) minimum coordinates
        max_corner: (3,) maximum coordinates
    """
    min_corner = coords.min(axis=0) - padding
    max_corner = coords.max(axis=0) + padding
    return min_corner, max_corner


def _count_voxels_in_spheres(
    x_coords: npt.NDArray[np.float64],
    y_coords: npt.NDArray[np.float64],
    z_coords: npt.NDArray[np.float64],
    centers: npt.NDArray[np.float64],
    radii: npt.NDArray[np.float64],
) -> int:
    """Count voxels inside any sphere using vectorized NumPy operations.

    Args:
        x_coords: x coordinates of voxel grid
        y_coords: y coordinates of voxel grid
        z_coords: z coordinates of voxel grid
        centers: (N_spheres, 3) sphere centers
        radii: (N_spheres,) sphere radii

    Returns:
        Count of voxels inside any sphere
    """
    grid_shape = (len(x_coords), len(y_coords), len(z_coords))
    inside_mask = np.zeros(grid_shape, dtype=bool)

    # Check each sphere
    for center, radius in zip(centers, radii, strict=True):
        # Find index ranges for this sphere's bounding box
        # This drastically reduces the number of operations per sphere
        # by only checking voxels within the sphere's bounding box
        x_mask = (x_coords >= center[0] - radius) & (x_coords <= center[0] + radius)
        y_mask = (y_coords >= center[1] - radius) & (y_coords <= center[1] + radius)
        z_mask = (z_coords >= center[2] - radius) & (z_coords <= center[2] + radius)

        # Extract small sub-grids
        lx, ly, lz = x_coords[x_mask], y_coords[y_mask], z_coords[z_mask]
        if lx.size == 0 or ly.size == 0 or lz.size == 0:
            continue

        # Broadcasting for distance check only within the bounding box
        # shape: (Nx, 1, 1) + (1, Ny, 1) + (1, 1, Nz)
        dist_sq = (
            (lx[:, None, None] - center[0]) ** 2
            + (ly[None, :, None] - center[1]) ** 2
            + (lz[None, None, :] - center[2]) ** 2
        )

        sphere_mask = dist_sq <= radius**2

        # Update the global mask using slicing
        ix = np.flatnonzero(x_mask)
        iy = np.flatnonzero(y_mask)
        iz = np.flatnonzero(z_mask)

        # Use np.ix_ to construct the update slice
        inside_mask[np.ix_(ix, iy, iz)] |= sphere_mask

    return int(inside_mask.sum())


def compute_voxel_volume(
    sphere_indices: set[int],
    spheres: list[AlphaSphere],
    voxel_size: float = 0.5,
) -> float:
    """Estimate pocket volume using voxel grid method.

    Uses vectorized NumPy operations for efficient voxel counting.

    Args:
        sphere_indices: list indices (positions) of spheres in pocket
        spheres: full list of spheres
        voxel_size: grid spacing in Ångströms

    Returns:
        Estimated volume in Å³
    """
    if not sphere_indices:
        return 0.0

    # Get all sphere centers and radii
    pocket_spheres = [spheres[idx] for idx in sphere_indices]
    centers = np.array([s.center for s in pocket_spheres], dtype=np.float64)
    radii = np.array([s.radius for s in pocket_spheres], dtype=np.float64)

    # Define bounding box
    max_radius = radii.max()
    min_corner, max_corner = bounding_box(centers, padding=max_radius)

    # Create voxel grid coordinates
    x = np.arange(min_corner[0], max_corner[0], voxel_size, dtype=np.float64)
    y = np.arange(min_corner[1], max_corner[1], voxel_size, dtype=np.float64)
    z = np.arange(min_corner[2], max_corner[2], voxel_size, dtype=np.float64)

    # Use vectorized function to count voxels
    inside_count = _count_voxels_in_spheres(x, y, z, centers, radii)

    # Volume = count * voxel_volume
    voxel_volume = voxel_size**3
    return float(inside_count) * voxel_volume
