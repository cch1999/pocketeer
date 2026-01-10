"""Tests for geometry module."""

import numpy as np
import pytest
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
    """Test volume calculation with NumPy engine."""
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

    # Compute volume with numpy engine
    volume_numpy = compute_voxel_volume(set([0, 1]), spheres, voxel_size=0.5, engine="numpy")

    # Volume should be positive
    assert volume_numpy > 0

    # Test with empty sphere set
    volume_empty = compute_voxel_volume(set(), spheres, voxel_size=0.5, engine="numpy")
    assert volume_empty == 0.0


def test_compute_voxel_volume_engines_consistency():
    """Test that NumPy and Numba engines produce consistent results."""
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

    # Compute volume with numpy engine
    volume_numpy = compute_voxel_volume(set([0, 1]), spheres, voxel_size=0.5, engine="numpy")

    # Try to compute with numba engine if available
    try:
        volume_numba = compute_voxel_volume(set([0, 1]), spheres, voxel_size=0.5, engine="numba")
        # Volumes should be very close (allowing for small numerical differences)
        assert abs(volume_numpy - volume_numba) < 1.0, (
            f"Volume mismatch: numpy={volume_numpy}, numba={volume_numba}"
        )
    except ImportError:
        # Numba not available, skip this part of the test
        pytest.skip("Numba not available")


def test_compute_voxel_volume_auto():
    """Test that auto engine selection works."""
    spheres = [
        AlphaSphere(
            sphere_id=0,
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[0, 1, 2, 3],
        ),
    ]

    # Auto should work regardless of numba availability
    volume_auto = compute_voxel_volume(set([0]), spheres, voxel_size=0.5, engine="auto")
    assert volume_auto > 0


def test_compute_voxel_volume_invalid_engine():
    """Test that invalid engine raises ValueError."""
    spheres = [
        AlphaSphere(
            sphere_id=0,
            center=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            radius=2.0,
            mean_sasa=0.0,
            atom_indices=[0, 1, 2, 3],
        ),
    ]

    with pytest.raises(ValueError, match="Invalid engine"):
        compute_voxel_volume(set([0]), spheres, voxel_size=0.5, engine="invalid")
