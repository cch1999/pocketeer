"""Tests for geometry module."""

import numpy as np
from scipy.spatial import cKDTree

from pocketeer.core.geometry import bounding_box, circumsphere, is_sphere_empty


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
