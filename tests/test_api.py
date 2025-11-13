import re

import pytest

from pocketeer import find_pockets, load_structure, write_individual_pocket_jsons

struc = pytest.importorskip("biotite.structure")


@pytest.fixture
def load_test_structure(pdb_id: str = "4tos_receptor") -> struc.AtomArray:
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
