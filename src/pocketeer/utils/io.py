"""I/O utilities for structure reading/writing and results export."""

import json
from pathlib import Path

import biotite.structure as struc  # type: ignore
from biotite.structure.io import mol, pdb, pdbx  # type: ignore

from ..core.types import Pocket
from .constants import ALPHABET_SIZE

FORMATS_MAP = {
    ".pdb": "pdb",
    ".ent": "pdb",  # PDB alternative extension
    ".cif": "cif",
    ".mmcif": "cif",  # mmCIF files
    ".bcif": "bcif",  # Binary CIF
    ".mol": "mol",
    ".sdf": "sdf",
}

SUPPORTED_FORMATS = list(FORMATS_MAP.keys())


def _guess_format(file_path: str) -> str:
    """Guess file format from extension.

    Args:
        file_path: path to structure file

    Returns:
        format string ('pdb', 'cif', 'bcif', 'mol', 'sdf')

    Raises:
        ValueError: if format cannot be determined
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in FORMATS_MAP:
        return FORMATS_MAP[suffix]

    raise ValueError(
        f"Cannot determine format for file '{file_path}'. "
        f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
    )


def _load_pdb_structure(structure_path: str, model: int) -> struc.AtomArray:
    """Load PDB structure."""
    pdb_file = pdb.PDBFile.read(structure_path)
    model_count = pdb.get_model_count(pdb_file)
    if model > model_count:
        raise ValueError(
            f"Model number {model} is out of range "
            f"for the input structure with {model_count} models"
        )
    return pdb_file.get_structure(model=model)


def _load_cif_structure(structure_path: str, model: int) -> struc.AtomArray:
    """Load mmCIF structure."""
    cif_file = pdbx.CIFFile.read(structure_path)
    model_count = pdbx.get_model_count(cif_file)
    if model > model_count:
        raise ValueError(
            f"Model number {model} is out of range "
            f"for the input structure with {model_count} models"
        )
    return pdbx.get_structure(cif_file, model=model)


def _load_bcif_structure(structure_path: str, model: int) -> struc.AtomArray:
    """Load BinaryCIF structure."""
    bcif_file = pdbx.BinaryCIFFile.read(structure_path)
    model_count = pdbx.get_model_count(bcif_file)
    if model > model_count:
        raise ValueError(
            f"Model number {model} is out of range "
            f"for the input structure with {model_count} models"
        )
    return pdbx.get_structure(bcif_file, model=model)


def _load_mol_structure(structure_path: str, model: int) -> struc.AtomArray:
    """Load MOL structure."""
    if model > 1:
        raise ValueError(
            f"Model number {model} is out of range for the input structure with 1 model"
        )
    mol_file = mol.MOLFile.read(structure_path)
    structure = mol_file.get_structure()
    structure.res_name[:] = mol_file.header.mol_name
    structure.atom_name[:] = struc.create_atom_names(structure)
    return structure


def _load_sdf_structure(structure_path: str, model: int) -> struc.AtomArray:
    """Load SDF structure."""
    if model > 1:
        raise ValueError(
            f"Model number {model} is out of range for the input structure with 1 model"
        )
    sd_file = mol.SDFile.read(structure_path)
    mol_name, sd_block = next(iter(sd_file.items()))
    structure = sd_block.get_structure()
    structure.res_name[:] = mol_name
    structure.atom_name[:] = struc.create_atom_names(structure)
    return structure


def load_structure(structure_path: str, model: int = 1) -> struc.AtomArray:
    """Load structure from various file formats using Biotite.

    Automatically detects file format from extension and loads the structure.
    Supports PDB, mmCIF/CIF, BinaryCIF, MOL, and SDF formats.

    Adapted from hydride:
    - https://github.com/biotite-dev/hydride/blob/main/src/hydride/cli.py

    Args:
        structure_path: path to structure file (any supported format)
        model: model number to load (default: 1, first model)

    Returns:
        AtomArray with structure data

    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if file format is unsupported or file is invalid
    """
    # Detect format from file extension
    file_format = _guess_format(structure_path)

    # Validate model number
    if model < 1:
        raise ValueError("Model number must be positive")

    # Load structure based on format
    format_loaders = {
        "pdb": _load_pdb_structure,
        "cif": _load_cif_structure,
        "bcif": _load_bcif_structure,
        "mol": _load_mol_structure,
        "sdf": _load_sdf_structure,
    }

    if file_format not in format_loaders:
        raise ValueError(f"Unsupported format: {file_format}")

    structure = format_loaders[file_format](structure_path, model)

    # Get first model if this is an AtomArrayStack (multi-model)
    if isinstance(structure, struc.AtomArrayStack):
        structure = structure[0]

    return structure


def write_pockets_as_pdb(
    output_path: str,
    pockets: list[Pocket],
) -> None:
    """Write pocket alpha-spheres as pseudo-atoms in PDB format.

    Each pocket gets a different chain ID (A, B, C, ...) for easy coloring in PyMOL.
    Each sphere is represented by its center as a carbon atom.
    B-factor encodes radius.

    Args:
        output_path: output PDB file path
        pockets: list of Pocket objects to write
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Helper function to generate chain IDs
    def get_chain_id(pocket_id: int) -> str:
        """Convert pocket ID to chain ID (A-Z, then AA-ZZ, etc.)"""
        if pocket_id < ALPHABET_SIZE:
            return chr(65 + pocket_id)  # A-Z
        elif pocket_id < ALPHABET_SIZE * 27:
            first = chr(65 + (pocket_id // 26) - 1)
            second = chr(65 + (pocket_id % 26))
            return f"{first}{second}"
        else:
            # For very large numbers, use numeric suffix
            return f"A{pocket_id}"

    with open(output_path, "w") as f:
        f.write("REMARK Alpha-spheres from pocketeer (pocket spheres only)\n")
        f.write("REMARK Each pocket has a different chain ID for PyMOL coloring\n")
        f.write("REMARK B-factor = sphere radius (Å)\n")
        f.write("REMARK Color by chain: PyMOL> spectrum chain\n")

        atom_num = 1
        for pocket in pockets:
            chain_id = get_chain_id(pocket.pocket_id)

            for sphere in pocket.spheres:
                x, y, z = sphere.center

                # PDB ATOM format with proper chain ID
                # Format: ATOM, serial, name, resName, chainID, resSeq, x, y, z,
                #         occupancy, tempFactor, element
                line = (
                    f"ATOM  {atom_num:5d}  C   SPH {chain_id:1s}{pocket.pocket_id + 1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{1.0:6.2f}{sphere.radius:6.2f}          C\n"
                )
                f.write(line)
                atom_num += 1

        f.write("END\n")


def write_pockets_json(
    output_path: str,
    pockets: list[Pocket],
) -> None:
    """Write pocket descriptors to JSON file.

    Args:
        output_path: output JSON file path
        pockets: list of pockets to write
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    data = [pocket.to_dict() for pocket in pockets]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_individual_pocket_jsons(
    output_dir: str,
    pockets: list[Pocket],
) -> None:
    """Write each pocket to a separate JSON file in a subdirectory.

    Creates a 'json/' subdirectory and writes pocket_1.json, pocket_2.json, etc.

    Args:
        output_dir: base output directory
        pockets: list of pockets to write
    """
    # Create json subdirectory
    json_dir = Path(output_dir) / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    # Write each pocket to a separate file
    for pocket in pockets:
        # Use 1-based numbering for user-friendly names
        pocket_filename = f"pocket_{pocket.pocket_id + 1}.json"
        pocket_path = json_dir / pocket_filename

        with open(pocket_path, "w") as f:
            json.dump(pocket.to_dict(), f, indent=2)


def write_summary(
    output_path: str,
    pockets: list[Pocket],
    pdb_file: str | None = None,
) -> None:
    """Write human-readable summary of detected pockets.

    Args:
        output_path: output text file path
        pockets: list of pockets
        pdb_file: input PDB filename
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("Pocketeer Pocket Detection Summary\n")
        f.write(f"{'=' * 60}\n\n")
        if pdb_file:
            f.write(f"Input: {pdb_file}\n")
        f.write(f"Found {len(pockets)} pocket(s)\n\n")

        # Sort by score
        sorted_pockets = sorted(pockets, key=lambda p: p.score, reverse=True)

        for pocket in sorted_pockets:
            f.write(f"Pocket {pocket.pocket_id + 1}:\n")
            f.write(f"  Score: {pocket.score:.2f}\n")
            f.write(f"  Volume: {pocket.volume:.1f} A³\n")
            f.write(f"  Spheres: {pocket.n_spheres}\n")
            f.write(f"  Residues: {len(pocket.residues)}\n")
            if pocket.residues:
                # Show first 10 residues, or all if fewer than 10
                MAX_RESIDUES_TO_SHOW = 10
                residue_strs = [
                    f"{res_name}{chain_id}{res_id}"
                    for chain_id, res_id, res_name in pocket.residues[:MAX_RESIDUES_TO_SHOW]
                ]
                f.write(f"    {', '.join(residue_strs)}")
                if len(pocket.residues) > MAX_RESIDUES_TO_SHOW:
                    f.write(f" ... ({len(pocket.residues) - MAX_RESIDUES_TO_SHOW} more)")
                f.write("\n")
            f.write(
                f"  Centroid: ({pocket.centroid[0]:.1f}, "
                f"{pocket.centroid[1]:.1f}, {pocket.centroid[2]:.1f})\n"
            )
            f.write("\n")
