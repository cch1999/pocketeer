"""Extract and label pocket features for a P2Rank dataset.

Usage: ``uv run python benchmarks/ml/featurize.py chen11``
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import biotite.structure as struc  # type: ignore
import numpy as np

from pocketeer import find_pockets, load_structure

CACHE = Path(__file__).parent / ".cache"
DCA_CUTOFF = 4.0
MIN_LIGAND_HEAVY_ATOMS = 6

# Unlike benchmark_ranking.py, cofactors are kept to match P2Rank's relevant-
# ligand definition. Only solvent, ions, additives, sugars, and detergents go.
IGNORE_LIG = {
    "HOH",
    "DOD",
    "WAT",
    "GOL",
    "EDO",
    "PEG",
    "PG4",
    "PGE",
    "1PE",
    "P6G",
    "MPD",
    "TRS",
    "EPE",
    "MES",
    "BTB",
    "BIS",
    "TAR",
    "TLA",
    "CIT",
    "FLC",
    "ACT",
    "ACY",
    "FMT",
    "DMS",
    "BME",
    "MRD",
    "IPA",
    "MOH",
    "EOH",
    "DOX",
    "IMD",
    "SCN",
    "AZI",
    "NO3",
    "SO4",
    "PO4",
    "SO3",
    "BO3",
    "MLI",
    "OXL",
    "NA",
    "K",
    "CL",
    "BR",
    "IOD",
    "MG",
    "CA",
    "ZN",
    "MN",
    "FE",
    "FE2",
    "CU",
    "CU1",
    "NI",
    "CO",
    "CD",
    "HG",
    "CS",
    "RB",
    "SR",
    "BA",
    "LI",
    "NAG",
    "MAN",
    "BMA",
    "FUC",
    "GAL",
    "GLC",
    "XYS",
    "SIA",
    "FUL",
    "NDG",
    "LDA",
    "LMT",
    "BOG",
    "SOG",
    "OLC",
    "OLA",
    "PLM",
    "MYR",
    "CLR",
    "D10",
    "C8E",
    "HTG",
    "F09",
    "UMQ",
    "Y01",
}
KD = {
    "ILE": 4.5,
    "VAL": 4.2,
    "LEU": 3.8,
    "PHE": 2.8,
    "CYS": 2.5,
    "MET": 1.9,
    "ALA": 1.8,
    "GLY": -0.4,
    "THR": -0.7,
    "SER": -0.8,
    "TRP": -0.9,
    "TYR": -1.3,
    "PRO": -1.6,
    "HIS": -3.2,
    "GLU": -3.5,
    "GLN": -3.5,
    "ASP": -3.5,
    "ASN": -3.5,
    "LYS": -3.9,
    "ARG": -4.5,
}
AROMATIC = {"PHE", "TRP", "TYR", "HIS"}
CHARGED = {"ASP", "GLU", "LYS", "ARG", "HIS"}


def relevant_ligands(atomarray) -> list[np.ndarray]:
    """Return coordinates for every relevant heavy-atom ligand residue."""
    hetero = atomarray[atomarray.hetero & (atomarray.element != "H")]
    if len(hetero) == 0:
        return []
    chain = hetero.chain_id.astype(str)
    res_id = hetero.res_id.astype(int).astype(str)
    res_name = hetero.res_name.astype(str)
    keys = np.char.add(np.char.add(chain, ":"), res_id)
    keys = np.char.add(np.char.add(keys, ":"), res_name)
    ligands = []
    for key in sorted(set(keys.tolist())):
        selection = keys == key
        if res_name[selection][0] in IGNORE_LIG or int(selection.sum()) < MIN_LIGAND_HEAVY_ATOMS:
            continue
        ligands.append(hetero.coord[selection])
    return ligands


def filtered(atomarray):
    """Replicate find_pockets' default hydrogen/solvent/hetero filtering."""
    atomarray = atomarray[atomarray.element != "H"]
    atomarray = atomarray[~struc.filter_solvent(atomarray)]
    return atomarray[~atomarray.hetero]


def pocket_features(pocket, elements) -> dict:
    """Compute the candidate feature set for one pocket."""
    radii = np.array([sphere.radius for sphere in pocket.spheres])
    sasa = np.array([sphere.mean_sasa for sphere in pocket.spheres])
    centers = pocket.coords
    apolar = np.array([np.mean(elements[sphere.atom_indices] == "C") for sphere in pocket.spheres])
    rgyr = float(np.sqrt(((centers - centers.mean(0)) ** 2).sum(1).mean()))
    residues = [residue[2] for residue in pocket.residues]
    kd = np.array([KD.get(name, 0.0) for name in residues]) if residues else np.array([0.0])
    return {
        "n_spheres": float(pocket.n_spheres),
        "volume": float(pocket.volume),
        "vol_per_sphere": float(pocket.volume / max(pocket.n_spheres, 1)),
        "rgyr": rgyr,
        "radius_mean": float(radii.mean()),
        "radius_std": float(radii.std()),
        "radius_min": float(radii.min()),
        "radius_max": float(radii.max()),
        "sasa_mean": float(sasa.mean()),
        "sasa_min": float(sasa.min()),
        "sasa_std": float(sasa.std()),
        "apolar_mean": float(apolar.mean()),
        "apolar_sum": float(apolar.sum()),
        "n_residues": float(len(residues)),
        "kd_mean": float(kd.mean()),
        "kd_sum": float(kd.sum()),
        "kd_max": float(kd.max()),
        "frac_aromatic": float(np.mean([name in AROMATIC for name in residues]))
        if residues
        else 0.0,
        "frac_charged": float(np.mean([name in CHARGED for name in residues])) if residues else 0.0,
    }


def process(pdb_path: Path) -> list[dict] | None:
    """Featurize every detected pocket in one structure."""
    try:
        atomarray = load_structure(str(pdb_path))
        ligands = relevant_ligands(atomarray)
        if not ligands:
            return None
        elements = filtered(atomarray).element.astype(str)
        pockets = find_pockets(atomarray)
    except Exception:  # noqa: BLE001 - skip unparseable or degenerate structures
        return None
    if not pockets:
        return None
    rows = []
    for rank, pocket in enumerate(pockets):
        dca = min(
            float(np.linalg.norm(coords[:, None, :] - pocket.coords[None, :, :], axis=2).min())
            for coords in ligands
        )
        features = pocket_features(pocket, elements)
        features.update(
            struct=pdb_path.stem,
            det_rank=rank,
            dca=dca,
            label=int(dca <= DCA_CUTOFF),
            heuristic_score=float(pocket.score),
            n_ligands=len(ligands),
        )
        rows.append(features)
    return rows


def main() -> None:
    dataset = sys.argv[1]
    pdbs = sorted((CACHE / dataset).glob("*.pdb"))
    if not pdbs:
        sys.exit(f"No structures in {CACHE / dataset}. Run fetch_datasets.py first.")
    rows = []
    n_ok = n_positive = 0
    for index, pdb_path in enumerate(pdbs):
        result = process(pdb_path)
        if result is None:
            continue
        n_ok += 1
        n_positive += sum(row["label"] for row in result)
        rows.extend(result)
        if (index + 1) % 50 == 0:
            print(f"  {index + 1}/{len(pdbs)} structures, {len(rows)} pockets", flush=True)
    out = CACHE / f"{dataset}.features.pkl"
    with out.open("wb") as file:
        pickle.dump(rows, file)
    print(f"DONE {dataset}: {n_ok} structures, {len(rows)} pockets, {n_positive} positive -> {out}")


if __name__ == "__main__":
    main()
