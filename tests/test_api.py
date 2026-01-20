import re

import numpy as np
import pytest

from pocketeer import find_pockets, load_structure, merge_pockets, write_individual_pocket_jsons

struc = pytest.importorskip("biotite.structure")

MIN_POCKETS_FOR_MERGE = 2


@pytest.fixture
def load_test_structure(pdb_id: str = "6qrd") -> struc.AtomArray:
    """Load a test structure from the tests/data directory."""
    return load_structure(f"tests/data/{pdb_id}.pdb")


def test_find_pockets_basic(load_test_structure):
    """Test basic pocket detection."""
    pockets = find_pockets(load_test_structure)
    assert isinstance(pockets, list)
    assert len(pockets) > 0


def test_find_pockets_ignore_hydrogens(load_test_structure):
    """Test ignoring hydrogen atoms."""
    pockets_no_h = find_pockets(load_test_structure, ignore_hydrogens=True)
    pockets_with_h = find_pockets(load_test_structure, ignore_hydrogens=False)
    print(pockets_no_h)
    assert isinstance(pockets_no_h, list)
    assert isinstance(pockets_with_h, list)


def test_find_pockets_ignore_water(load_test_structure):
    """Test ignoring water molecules."""
    pockets_no_water = find_pockets(load_test_structure, ignore_water=True)
    pockets_with_water = find_pockets(load_test_structure, ignore_water=False)
    assert isinstance(pockets_no_water, list)
    assert isinstance(pockets_with_water, list)


def test_find_pockets_ignore_hetero(load_test_structure):
    """Test ignoring hetero atoms."""
    pockets_no_hetero = find_pockets(load_test_structure, ignore_hetero=True)
    pockets_with_hetero = find_pockets(load_test_structure, ignore_hetero=False)
    print(pockets_no_hetero)
    assert isinstance(pockets_no_hetero, list)
    assert isinstance(pockets_with_hetero, list)
    assert len(pockets_no_hetero) != len(pockets_with_hetero)


def test_find_pockets_sasa_threshold(load_test_structure):
    """Test that sasa_threshold filters spheres correctly."""
    # Lower threshold should filter more aggressively (fewer pockets)
    pockets_low = find_pockets(load_test_structure, sasa_threshold=10.0)
    # Higher threshold should include more spheres (more or equal pockets)
    pockets_high = find_pockets(load_test_structure, sasa_threshold=30.0)
    # Default threshold
    pockets_default = find_pockets(load_test_structure, sasa_threshold=20.0)

    assert isinstance(pockets_low, list)
    assert isinstance(pockets_high, list)
    assert isinstance(pockets_default, list)

    # Higher threshold should generally find more or equal pockets
    # (more surface-exposed spheres are included)
    assert len(pockets_high) >= len(pockets_low), (
        f"Expected higher threshold (30.0) to find >= pockets than lower (10.0), "
        f"but found {len(pockets_high)} vs {len(pockets_low)}"
    )


def test_write_individual_pocket_jsons(tmp_path, load_test_structure):
    """Pocket JSONs should use 1-based numbering."""
    pockets = find_pockets(load_test_structure)
    assert pockets, "Expected at least one detected pocket for numbering test"

    write_individual_pocket_jsons(tmp_path, pockets)

    json_dir = tmp_path / "json"
    assert json_dir.is_dir(), "Expected json output directory to be created"

    # Expected 1-based file numbers
    expected_numbers = {pocket.pocket_id + 1 for pocket in pockets}

    # Check that all expected 1-based files exist
    for pocket in pockets:
        one_based = json_dir / f"pocket_{pocket.pocket_id + 1}.json"
        assert one_based.exists(), f"Missing expected file {one_based.name}"

    # Check that all files use 1-based numbering and no extra files exist
    all_files = list(json_dir.glob("pocket_*.json"))
    assert len(all_files) == len(pockets), f"Expected {len(pockets)} files, found {len(all_files)}"

    for file_path in all_files:
        filename = file_path.name
        match = re.match(r"pocket_(\d+)\.json", filename)
        if match:
            file_number = int(match.group(1))
            # File number should be in expected 1-based set (pocket_id + 1)
            assert file_number in expected_numbers, (
                f"File {filename} uses unexpected numbering. "
                f"Expected files: {sorted(expected_numbers)}"
            )


def test_pocket_mask(load_test_structure):
    """Test that pocket mask correctly selects atoms from original atomarray."""
    original_atomarray = load_test_structure
    pockets = find_pockets(original_atomarray)

    if not pockets:
        pytest.skip("No pockets found for mask testing")

    # Test the first pocket
    pocket = pockets[0]

    # Verify mask exists and has correct type
    assert hasattr(pocket, "mask"), "Pocket should have a mask attribute"
    assert isinstance(pocket.mask, np.ndarray), "Mask should be a numpy array"
    assert pocket.mask.dtype == bool, "Mask should be boolean"
    assert len(pocket.mask) == len(original_atomarray), (
        f"Mask length ({len(pocket.mask)}) should match original atomarray length "
        f"({len(original_atomarray)})"
    )

    # Verify mask can be used to select atoms
    pocket_atoms = original_atomarray[pocket.mask]
    assert len(pocket_atoms) > 0, "Mask should select at least some atoms"


def test_merge_pockets_basic(load_test_structure):
    """Test basic pocket merging functionality."""
    pockets = find_pockets(load_test_structure)

    if len(pockets) < MIN_POCKETS_FOR_MERGE:
        pytest.skip("Need at least 2 pockets to test merging")

    # Merge first two pockets
    merged = merge_pockets(pockets[:2])

    # Verify merged pocket has correct structure
    assert merged.pocket_id == min(p.pocket_id for p in pockets[:2])
    assert len(merged.spheres) >= max(len(p.spheres) for p in pockets[:2])
    assert len(merged.residues) >= max(len(p.residues) for p in pockets[:2])
    assert merged.volume >= max(p.volume for p in pockets[:2])
    assert merged.score >= 0  # Score should be non-negative

    # Verify mask is merged correctly (logical OR)
    assert np.array_equal(merged.mask, np.any([p.mask for p in pockets[:2]], axis=0))


def test_merge_pockets_custom_id(load_test_structure):
    """Test merging with custom pocket ID."""
    pockets = find_pockets(load_test_structure)

    if len(pockets) < MIN_POCKETS_FOR_MERGE:
        pytest.skip("Need at least 2 pockets to test merging")

    custom_id = 999
    merged = merge_pockets(pockets[:2], new_pocket_id=custom_id)
    assert merged.pocket_id == custom_id


def test_merge_pockets_single(load_test_structure):
    """Test merging a single pocket returns it unchanged."""
    pockets = find_pockets(load_test_structure)

    if not pockets:
        pytest.skip("Need at least 1 pocket to test")

    merged = merge_pockets([pockets[0]])
    assert merged.pocket_id == pockets[0].pocket_id
    assert len(merged.spheres) == len(pockets[0].spheres)
    assert merged.score == pockets[0].score


def test_merge_pockets_empty_list():
    """Test that merging empty list raises ValueError."""
    with pytest.raises(ValueError, match="Cannot merge empty list"):
        merge_pockets([])


def test_merge_pockets_deduplication(load_test_structure):
    """Test that duplicate spheres are deduplicated."""
    pockets = find_pockets(load_test_structure)

    if len(pockets) < MIN_POCKETS_FOR_MERGE:
        pytest.skip("Need at least 2 pockets to test deduplication")

    # Get sphere IDs from both pockets
    sphere_ids_1 = set(pockets[0].sphere_ids)
    sphere_ids_2 = set(pockets[1].sphere_ids)

    # Merge them
    merged = merge_pockets(pockets[:2])

    # Count unique sphere IDs in merged pocket
    merged_sphere_ids = set(merged.sphere_ids)

    # Merged should have at most the union of sphere IDs (could be less if there's overlap)
    assert len(merged_sphere_ids) <= len(sphere_ids_1 | sphere_ids_2)
    assert len(merged.spheres) == len(merged_sphere_ids)  # No duplicates


def test_pocket_coords(load_test_structure):
    """Test that pocket.coords returns correctly shaped numpy array."""
    pockets = find_pockets(load_test_structure)
    if not pockets:
        pytest.skip("No pockets found for coords testing")

    pocket = pockets[0]
    coords = pocket.coords

    assert isinstance(coords, np.ndarray)
    assert coords.shape == (len(pocket.spheres), 3)
    # Check first coordinate matches first sphere's center
    assert np.allclose(coords[0], pocket.spheres[0].center)
