"""Tests for pocketeer visualization module."""

from pathlib import Path

import biotite.structure as struc
import numpy as np
import pytest

import pocketeer
from pocketeer import find_pockets, load_structure
from pocketeer.core.types import AlphaSphere, Pocket


def test_view_pockets_import():
    """Test that visualization function can be imported."""
    # Test conditional import
    if hasattr(pocketeer, "view_pockets"):
        assert callable(pocketeer.view_pockets)
    else:
        # If not available, that's also valid (atomworks not installed)
        pytest.skip("view_pockets not available (atomworks not installed)")


def test_view_pockets_no_pockets():
    """Test view_pockets with empty pockets list."""
    if not hasattr(pocketeer, "view_pockets"):
        pytest.skip("view_pockets not available (atomworks not installed)")

    # Create a simple structure
    atoms = struc.array(
        [
            struc.Atom([0, 0, 0], element="C"),
            struc.Atom([1, 0, 0], element="C"),
            struc.Atom([0, 1, 0], element="C"),
        ]
    )

    # Should raise ValueError for empty pockets
    with pytest.raises(ValueError, match="No pockets found"):
        pocketeer.view_pockets(atoms, [])


def test_view_pockets_invalid_input():
    """Test view_pockets with invalid input types."""
    if not hasattr(pocketeer, "view_pockets"):
        pytest.skip("view_pockets not available (atomworks not installed)")

    # Create mock pockets
    mock_pockets = [
        Pocket(
            pocket_id=1,
            spheres=[
                AlphaSphere(
                    sphere_id=0,
                    center=np.array([0, 0, 0]),
                    radius=2.0,
                    mean_sasa=10.0,
                    atom_indices=[0, 1, 2, 3],
                )
            ],
            centroid=np.array([0, 0, 0]),
            volume=100.0,
            score=1.0,
            residues=[],
            mask=np.array([False] * 4, dtype=bool),
        )
    ]

    # Test with non-AtomArray input
    with pytest.raises(TypeError, match="atomarray must be a Biotite AtomArray"):
        pocketeer.view_pockets("not_an_atomarray", mock_pockets)

    with pytest.raises(TypeError, match="atomarray must be a Biotite AtomArray"):
        pocketeer.view_pockets(np.array([1, 2, 3]), mock_pockets)


def test_view_pockets_invalid_color_scheme():
    """Test view_pockets with invalid color scheme."""
    if not hasattr(pocketeer, "view_pockets"):
        pytest.skip("view_pockets not available (atomworks not installed)")

    # Create a simple structure and pockets
    atoms = struc.array(
        [
            struc.Atom([0, 0, 0], element="C"),
            struc.Atom([1, 0, 0], element="C"),
            struc.Atom([0, 1, 0], element="C"),
        ]
    )

    mock_pockets = [
        Pocket(
            pocket_id=1,
            spheres=[
                AlphaSphere(
                    sphere_id=0,
                    center=np.array([0, 0, 0]),
                    radius=2.0,
                    mean_sasa=10.0,
                    atom_indices=[0, 1, 2, 3],
                )
            ],
            centroid=np.array([0, 0, 0]),
            volume=100.0,
            score=1.0,
            residues=[],
            mask=np.array([False] * len(atoms), dtype=bool),
        )
    ]

    # Test with invalid color scheme
    with pytest.raises(ValueError, match="Unknown color scheme"):
        pocketeer.view_pockets(atoms, mock_pockets, color_scheme="invalid")


def test_view_pockets_success():
    """Test successful visualization with real data."""
    if not hasattr(pocketeer, "view_pockets"):
        pytest.skip("view_pockets not available (atomworks not installed)")

    # Try to load a real structure and find pockets
    test_pdb = Path(__file__).parent / "data" / "6qrd.pdb"
    if not test_pdb.exists():
        pytest.skip(f"Test PDB file not found: {test_pdb}")

    try:
        atoms = load_structure(str(test_pdb))
        # Lower threshold for testing
        pockets = find_pockets(atoms, r_min=3.0, r_max=5.0, min_spheres=10)

        if pockets:  # Only test if pockets were found
            # Test different color schemes
            viewer1 = pocketeer.view_pockets(atoms, pockets, color_scheme="rainbow")
            assert viewer1 is not None

            viewer2 = pocketeer.view_pockets(atoms, pockets, color_scheme="grayscale")
            assert viewer2 is not None

            viewer3 = pocketeer.view_pockets(atoms, pockets, color_scheme="red_blue")
            assert viewer3 is not None

            # Test with different parameters
            viewer4 = pocketeer.view_pockets(atoms, pockets, sphere_opacity=0.5, sphere_scale=1.5)
            assert viewer4 is not None

        else:
            pytest.skip("No pockets found in test structure")

    except ImportError as e:
        pytest.skip(f"atomworks not available: {e}")


def test_view_pockets_color_schemes():
    """Test all supported color schemes."""
    if not hasattr(pocketeer, "view_pockets"):
        pytest.skip("view_pockets not available (atomworks not installed)")

    # Create a simple structure
    atoms = struc.array(
        [
            struc.Atom([0, 0, 0], element="C"),
            struc.Atom([1, 0, 0], element="C"),
            struc.Atom([0, 1, 0], element="C"),
        ]
    )

    mock_pockets = [
        Pocket(
            pocket_id=1,
            spheres=[
                AlphaSphere(
                    sphere_id=0,
                    center=np.array([0, 0, 0]),
                    radius=2.0,
                    mean_sasa=10.0,
                    atom_indices=[0, 1, 2, 3],
                )
            ],
            centroid=np.array([0, 0, 0]),
            volume=100.0,
            score=1.0,
            residues=[],
            mask=np.array([False] * len(atoms), dtype=bool),
        ),
        Pocket(
            pocket_id=2,
            spheres=[
                AlphaSphere(
                    sphere_id=1,
                    center=np.array([2, 0, 0]),
                    radius=1.5,
                    mean_sasa=8.0,
                    atom_indices=[4, 5, 6, 7],
                )
            ],
            centroid=np.array([2, 0, 0]),
            volume=80.0,
            score=0.8,
            residues=[],
            mask=np.array([False] * len(atoms), dtype=bool),
        ),
    ]

    # Test all color schemes
    for scheme in ["rainbow", "grayscale", "red_blue"]:
        viewer = pocketeer.view_pockets(atoms, mock_pockets, color_scheme=scheme)
        assert viewer is not None
