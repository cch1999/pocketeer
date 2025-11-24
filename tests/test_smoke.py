"""Smoke tests for pocketeer."""

import biotite.structure as struc
import pytest

import pocketeer
from pocketeer import find_pockets
from pocketeer.utils import ValidationError


def test_import():
    """Test that package imports successfully."""

    assert pocketeer.__version__ == "0.1.0"


def test_find_pockets_empty():
    """Test find_pockets with empty AtomArray."""
    # Create empty AtomArray (create a single atom then slice to empty)
    atom = struc.Atom([0, 0, 0], element="C")
    atoms = struc.array([atom])[0:0]  # Empty slice

    # Should raise ValidationError for empty AtomArray
    with pytest.raises(ValidationError, match="AtomArray is empty"):
        find_pockets(atoms)


def test_find_pockets_too_few():
    """Test find_pockets with too few atoms."""
    # Create minimal AtomArray with 3 atoms
    atoms = struc.array(
        [
            struc.Atom([0, 0, 0], element="C"),
            struc.Atom([1, 0, 0], element="C"),
            struc.Atom([0, 1, 0], element="C"),
        ]
    )

    pockets = find_pockets(atoms)

    # Too few atoms - should return empty
    assert pockets == []


def test_params_validation():
    """Test parameter validation."""
    # Create a valid structure for testing
    # Note: SASA calculation requires res_name and atom_name to be set
    atoms = struc.array(
        [
            struc.Atom([0, 0, 0], element="C", res_name="GLY", atom_name="CA"),
            struc.Atom([1, 0, 0], element="C", res_name="GLY", atom_name="CA"),
            struc.Atom([0, 1, 0], element="C", res_name="GLY", atom_name="CA"),
            struc.Atom([0, 0, 1], element="C", res_name="GLY", atom_name="CA"),
        ]
    )

    # Valid params - should work
    pockets = find_pockets(atoms, r_min=3.0, r_max=6.0)
    assert isinstance(pockets, list)

    # Invalid params - r_max < r_min
    with pytest.raises(ValueError, match="Invalid radius range"):
        find_pockets(atoms, r_min=6.0, r_max=3.0)

    # Invalid params - negative r_min
    with pytest.raises(ValueError, match="Invalid radius range"):
        find_pockets(atoms, r_min=-1.0)

    # Invalid params - negative polar_probe_radius
    with pytest.raises(ValueError, match="polar_probe_radius must be > 0"):
        find_pockets(atoms, polar_probe_radius=-1.0)

    # Invalid params - negative sasa_threshold
    with pytest.raises(ValueError, match="sasa_threshold must be > 0"):
        find_pockets(atoms, sasa_threshold=-1.0)
