"""Tests for geometry module."""

import numpy as np
from scipy.spatial import cKDTree

from pocketeer.core.geometry import (
    bounding_box,
    circumsphere,
    compute_voxel_volume,
    is_sphere_empty,
)
from pocketeer.core.types import AlphaSphere


def test_circumsphere_regular_tetrahedron():
    """Test circumsphere for regular tetrahedron."""
    # Regular tetrahedron with edge length ~1.63
    points = np.array(
        [
            [1, 1, 1],
            [1, -1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
        ],
        dtype=np.float64,
    )

    center, radius = circumsphere(points)

    # Center should be at origin
    assert np.allclose(center, [0, 0, 0], atol=1e-10)

    # All points should be equidistant
    distances = np.linalg.norm(points - center, axis=1)
    assert np.allclose(distances, radius, atol=1e-10)


def test_circumsphere_degenerate():
    """Test circumsphere with coplanar points."""
    # Four coplanar points
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )

    center, _ = circumsphere(points)
    # Should handle gracefully (may return degenerate sphere)
    assert center.shape == (3,)


def test_is_sphere_empty():
    """Test empty sphere detection."""

    # Create atoms at corners of cube
    coords = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # Build KD-tree for fast spatial queries
    tree = cKDTree(coords)

    # Sphere at center with small radius - should be empty
    center = np.array([0.25, 0.25, 0.25])
    radius = 0.2
    assert is_sphere_empty(center, radius, tree, set(), tolerance=1e-6)

    # Sphere at center with large radius - should contain atoms
    radius = 1.0
    assert not is_sphere_empty(center, radius, tree, set(), tolerance=1e-6)

    # Sphere that excludes its defining atoms
    center = np.array([0.5, 0.5, 0.5])
    radius = 0.87  # Just reaches corners
    assert is_sphere_empty(center, radius, tree, {0, 1, 2, 3}, tolerance=0.01)


def test_bounding_box():
    """Test bounding box computation."""
    coords = np.array(
        [
            [0, 0, 0],
            [1, 2, 3],
            [-1, -2, -3],
        ],
        dtype=np.float64,
    )

    min_corner, max_corner = bounding_box(coords)

    assert np.allclose(min_corner, [-1, -2, -3])
    assert np.allclose(max_corner, [1, 2, 3])

    # Test with padding
    min_corner, max_corner = bounding_box(coords, padding=1.0)
    assert np.allclose(min_corner, [-2, -3, -4])
    assert np.allclose(max_corner, [2, 3, 4])


def test_compute_voxel_volume_numpy():
    """Test volume calculation with vectorized NumPy operations."""
    # Create a simple test case: two overlapping spheres
    spheres = [
        AlphaSphere(
            sphere_id=0,
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[0, 1, 2, 3],
        ),
        AlphaSphere(
            sphere_id=1,
            center=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[4, 5, 6, 7],
        ),
    ]

    # Compute volume
    volume = compute_voxel_volume(set([0, 1]), spheres, voxel_size=0.5)

    # Volume should be positive
    assert volume > 0

    # Test with empty sphere set
    volume_empty = compute_voxel_volume(set(), spheres, voxel_size=0.5)
    assert volume_empty == 0.0


def test_compute_voxel_volume_consistency():
    """Test that volume calculation produces consistent results."""
    # Create a simple test case
    spheres = [
        AlphaSphere(
            sphere_id=0,
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[0, 1, 2, 3],
        ),
        AlphaSphere(
            sphere_id=1,
            center=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[4, 5, 6, 7],
        ),
    ]

    # Compute volume multiple times - should be consistent
    volume1 = compute_voxel_volume(set([0, 1]), spheres, voxel_size=0.5)
    volume2 = compute_voxel_volume(set([0, 1]), spheres, voxel_size=0.5)

    # Volumes should be identical
    assert volume1 == volume2
    assert volume1 > 0


def test_compute_voxel_volume_basic():
    """Test basic volume calculation."""
    spheres = [
        AlphaSphere(
            sphere_id=0,
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[0, 1, 2, 3],
        ),
    ]

    # Compute volume
    volume = compute_voxel_volume(set([0]), spheres, voxel_size=0.5)
    assert volume > 0


def test_compute_voxel_volume_different_voxel_sizes():
    """Test that different voxel sizes produce different but consistent results."""
    spheres = [
        AlphaSphere(
            sphere_id=0,
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[0, 1, 2, 3],
        ),
    ]

    # Smaller voxel size should give more accurate (but potentially different) volume
    volume_small = compute_voxel_volume(set([0]), spheres, voxel_size=0.25)
    volume_large = compute_voxel_volume(set([0]), spheres, voxel_size=0.5)

    # Both should be positive
    assert volume_small > 0
    assert volume_large > 0

    # Smaller voxel size typically gives more accurate volume (closer to true volume)
    # but may be slightly different due to discretization
    assert abs(volume_small - volume_large) < volume_large * 0.5  # Allow some difference
