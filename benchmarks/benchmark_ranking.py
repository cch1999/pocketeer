"""Independent ranking-quality benchmark for pocketeer.

For every holo structure, select its largest drug-like ligand, detect pockets
with hetero atoms removed, and report the rank of the first pocket within 4 A
of that ligand. Downloads are cached under ``benchmarks/.cache``.

Run: ``uv run python benchmarks/benchmark_ranking.py [--decoy-rich-only]``
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

from pocketeer import find_pockets, load_structure

BENCH_DIR = Path(__file__).parent
CACHE_DIR = BENCH_DIR / ".cache"
DCA_CUTOFF = 4.0
MIN_LIGAND_HEAVY_ATOMS = 6

# Water, ions, common crystallisation additives, sugars, detergents, and
# prosthetic cofactors are not drug-like ligands for this sanity check.
NON_LIGAND_RESNAMES = {
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
    "HEM",
    "HEC",
    "HEB",
    "HAS",
    "FAD",
    "FMN",
    "NAD",
    "NAP",
    "NDP",
    "NAI",
    "FES",
    "SF4",
    "PLP",
    "TPP",
    "COA",
    "B12",
    "BTN",
    "PQQ",
    "MGD",
    "F43",
}


def fetch_pdb(pdb_id: str) -> Path | None:
    """Download a PDB file into the cache; return None on failure."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{pdb_id}.pdb"
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        urllib.request.urlretrieve(  # trusted RCSB host
            f"https://files.rcsb.org/download/{pdb_id}.pdb", path
        )
    except Exception:  # noqa: BLE001 - a failed case is reported by the benchmark
        path.unlink(missing_ok=True)
        return None
    return path


def detect_primary_ligand(atomarray) -> tuple[str, np.ndarray] | None:
    """Return the largest drug-like HETATM residue, deterministically."""
    hetero = atomarray[atomarray.hetero & (atomarray.element != "H")]
    if len(hetero) == 0:
        return None
    chain = hetero.chain_id.astype(str)
    res_id = hetero.res_id.astype(int).astype(str)
    res_name = hetero.res_name.astype(str)
    keys = np.char.add(np.char.add(chain, ":"), res_id)
    keys = np.char.add(np.char.add(keys, ":"), res_name)
    candidates = []
    for key in sorted(set(keys.tolist())):
        selection = keys == key
        name = str(res_name[selection][0])
        if name not in NON_LIGAND_RESNAMES and int(selection.sum()) >= MIN_LIGAND_HEAVY_ATOMS:
            candidates.append((int(selection.sum()), key, name, hetero.coord[selection]))
    if not candidates:
        return None
    _, _, name, coords = max(candidates, key=lambda item: (item[0], item[1]))
    return name, coords


def evaluate_case(pdb_id: str) -> dict:
    """Evaluate one structure and return its ranking result."""
    path = fetch_pdb(pdb_id)
    if path is None:
        return {"pdb": pdb_id, "status": "download_failed"}
    atomarray = load_structure(str(path))
    ligand = detect_primary_ligand(atomarray)
    if ligand is None:
        return {"pdb": pdb_id, "status": "no_ligand"}
    ligand_name, ligand_coords = ligand
    pockets = find_pockets(atomarray)
    if not pockets:
        return {"pdb": pdb_id, "status": "no_pockets", "ligand": ligand_name}

    dcas = np.array(
        [
            float(np.linalg.norm(ligand_coords[:, None, :] - p.coords[None, :, :], axis=2).min())
            for p in pockets
        ]
    )
    volumes = np.array([p.volume for p in pockets])
    hit_ranks = np.where(dcas <= DCA_CUTOFF)[0]
    if hit_ranks.size == 0:
        return {
            "pdb": pdb_id,
            "status": "site_not_detected",
            "ligand": ligand_name,
            "n_pockets": len(pockets),
        }
    best_rank = int(hit_ranks[0]) + 1
    true_volume = volumes[hit_ranks].max()
    larger_decoys = int(((volumes > true_volume) & (dcas > DCA_CUTOFF)).sum())
    return {
        "pdb": pdb_id,
        "status": "ok",
        "ligand": ligand_name,
        "n_pockets": len(pockets),
        "rank": best_rank,
        "top1": best_rank == 1,
        "top3": best_rank <= 3,  # noqa: PLR2004 - Top-3 is a standard cutoff
        "larger_decoys": larger_decoys,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoy-rich-only", action="store_true")
    args = parser.parse_args()
    pdb_ids = [
        line.strip()
        for line in (BENCH_DIR / "pdb_list.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    rows = []
    for pdb_id in pdb_ids:
        result = evaluate_case(pdb_id)
        rows.append(result)
        if result["status"] == "ok":
            flag = "DECOY-RICH" if result["larger_decoys"] else ""
            print(
                f"{pdb_id}  lig={result['ligand']:>4}  pockets={result['n_pockets']:>3}  "
                f"rank={result['rank']:>2}  larger_decoys={result['larger_decoys']:>2}  {flag}"
            )
        else:
            print(f"{pdb_id}  -- {result['status']}")

    ok = [row for row in rows if row["status"] == "ok"]
    pool = [row for row in ok if row["larger_decoys"] > 0] if args.decoy_rich_only else ok
    if not pool:
        print("\nNo evaluable cases.")
        return
    n = len(pool)
    top1 = sum(row["top1"] for row in pool)
    top3 = sum(row["top3"] for row in pool)
    label = "decoy-rich" if args.decoy_rich_only else "all evaluable"
    print(f"\n=== Ranking benchmark ({label}, n={n}) ===")
    print(f"Top-1: {top1}/{n} ({100 * top1 / n:.0f}%)   Top-3: {top3}/{n} ({100 * top3 / n:.0f}%)")
    print(
        "Decoy-rich cases (>=1 larger non-binding cavity): "
        f"{sum(row['larger_decoys'] > 0 for row in ok)}/{len(ok)}"
    )
    detection_misses = sum(row["status"] == "site_not_detected" for row in rows)
    if detection_misses:
        print(f"(detection misses, excluded from ranking pool: {detection_misses})")


if __name__ == "__main__":
    main()
