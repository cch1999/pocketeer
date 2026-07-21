"""Grid-search detector clustering parameters and retrain the pocket ranker.

This benchmark intentionally uses the package's current SASA implementation.
Structures that fail SASA calculation are counted and excluded, allowing atom
sanitisation to be evaluated separately.

For each merge distance, tessellation and SASA are computed once per structure
and every connected component above the smallest requested size is featurized.
Different ``min_spheres`` values then select from those cached candidates.

Usage:
    uv run python benchmarks/ml/grid_search_detector.py
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc  # type: ignore
import numpy as np
from featurize import AROMATIC, DCA_CUTOFF, relevant_ligands
from train_ranker import COMPACT6, auc, predict, rank_metrics, train_lr

from pocketeer import load_structure
from pocketeer.core.clustering import build_sphere_graph, connected_components
from pocketeer.core.geometry import compute_voxel_volume
from pocketeer.core.scoring import extract_pocket_residues
from pocketeer.core.tessellation import (
    compute_alpha_spheres,
    filter_surface_spheres,
    label_polarity,
)

CACHE = Path(__file__).parent / ".cache"
DEFAULT_MERGE_DISTANCES = (2.0, 2.25, 2.5, 2.75, 3.0)
DEFAULT_MIN_SPHERES = (30, 35, 40, 45, 50)


@dataclass
class DatasetCandidates:
    """Candidate pockets and preprocessing accounting for one dataset."""

    rows: list[dict]
    eligible: int
    sasa_errors: int
    other_errors: int
    elapsed: float


def _filtered(atomarray):
    atomarray = atomarray[atomarray.element != "H"]
    atomarray = atomarray[~struc.filter_solvent(atomarray)]
    return atomarray[~atomarray.hetero]


def _components(spheres, merge_distance: float) -> list[set[int]]:
    """Return graph components, including isolated sphere nodes."""
    graph = build_sphere_graph(spheres, merge_distance)
    components = connected_components(graph)
    covered = set().union(*components) if components else set()
    components.extend({index} for index in range(len(spheres)) if index not in covered)
    return components


def _candidate_row(struct: str, component, spheres, atomarray, ligands) -> dict:
    component_spheres = [spheres[index] for index in component]
    residues = extract_pocket_residues(component_spheres, atomarray)
    residue_names = [residue[2] for residue in residues]
    volume = compute_voxel_volume(
        set(range(len(component_spheres))), component_spheres, voxel_size=0.5
    )
    centers = np.array([sphere.center for sphere in component_spheres])
    dca = min(
        float(np.linalg.norm(coords[:, None, :] - centers[None, :, :], axis=2).min())
        for coords in ligands
    )
    radii = np.array([sphere.radius for sphere in component_spheres])
    sasa = np.array([sphere.mean_sasa for sphere in component_spheres])
    return {
        "struct": struct,
        "n_ligands": len(ligands),
        "label": int(dca <= DCA_CUTOFF),
        "dca": dca,
        "component_size": len(component),
        "n_residues": float(len(residues)),
        "n_spheres": float(len(component_spheres)),
        "vol_per_sphere": volume / len(component_spheres),
        "radius_mean": float(radii.mean()),
        "sasa_mean": float(sasa.mean()),
        "frac_aromatic": (
            float(np.mean([name in AROMATIC for name in residue_names])) if residue_names else 0.0
        ),
    }


def build_candidates(
    dataset: str, merge_distance: float, minimum_component: int
) -> DatasetCandidates:
    """Build all candidates for a dataset and one linkage distance."""
    started = time.perf_counter()
    rows = []
    eligible = sasa_errors = other_errors = 0
    pdbs = sorted((CACHE / dataset).glob("*.pdb"))
    for index, pdb_path in enumerate(pdbs, 1):
        try:
            original = load_structure(str(pdb_path))
            ligands = relevant_ligands(original)
            if not ligands:
                continue
            eligible += 1
            atomarray = _filtered(original)
            spheres = compute_alpha_spheres(atomarray.coord, r_min=3.0, r_max=6.0)
            spheres = label_polarity(spheres, atomarray, polar_probe_radius=1.4)
            spheres = filter_surface_spheres(spheres, sasa_threshold=20.0)
        except KeyError:
            sasa_errors += 1
            continue
        except Exception:  # noqa: BLE001 - account for unusable benchmark structures
            other_errors += 1
            continue

        for component in _components(spheres, merge_distance):
            if len(component) >= minimum_component:
                rows.append(_candidate_row(pdb_path.stem, component, spheres, atomarray, ligands))
        if index % 50 == 0:
            print(
                f"  {dataset} merge={merge_distance:g}: {index}/{len(pdbs)}, "
                f"{len(rows)} candidates",
                flush=True,
            )
    return DatasetCandidates(
        rows=rows,
        eligible=eligible,
        sasa_errors=sasa_errors,
        other_errors=other_errors,
        elapsed=time.perf_counter() - started,
    )


def _matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = np.array([[row[name] for name in COMPACT6] for row in rows], dtype=float)
    labels = np.array([row["label"] for row in rows], dtype=float)
    return features, labels


def _end_to_end_metrics(rows, scores, eligible):
    grouped = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        grouped[row["struct"]].append((score, row["label"]))
    detected = top1 = top3 = 0
    for items in grouped.values():
        ranked = sorted(items, reverse=True)
        labels = [label for _, label in ranked]
        if any(labels):
            detected += 1
            top1 += labels[0] == 1
            top3 += 1 in labels[:3]
    return detected, top1, top3, top1 / eligible, top3 / eligible


def evaluate_grid(merge_distance, train, test, min_spheres_values):
    print(
        "merge  min  train_pockets  test_pockets  decoys/struct  detect  "
        "AUC   Top1|det  Top3|det  Top1|all  Top3|all"
    )
    for minimum in min_spheres_values:
        train_rows = [row for row in train.rows if row["component_size"] >= minimum]
        test_rows = [row for row in test.rows if row["component_size"] >= minimum]
        train_x, train_y = _matrix(train_rows)
        test_x, test_y = _matrix(test_rows)
        weights, mean, std = train_lr(train_x, train_y)
        scores = predict(weights, mean, std, test_x)
        top1, top3, _, rank_n = rank_metrics(test_rows, scores)
        detected, _, _, end_top1, end_top3 = _end_to_end_metrics(
            test_rows, scores, test.eligible - test.sasa_errors - test.other_errors
        )
        negatives = int((test_y == 0).sum())
        usable = test.eligible - test.sasa_errors - test.other_errors
        print(
            f"{merge_distance:>5.2f} {minimum:>4} {len(train_rows):>14} "
            f"{len(test_rows):>13} {negatives / usable:>13.2f} "
            f"{detected:>3}/{usable:<3} {auc(test_y, scores):>5.3f} "
            f"{100 * top1 / rank_n:>8.1f}% {100 * top3 / rank_n:>8.1f}% "
            f"{100 * end_top1:>8.1f}% {100 * end_top3:>8.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-distances", nargs="+", type=float, default=DEFAULT_MERGE_DISTANCES)
    parser.add_argument("--min-spheres", nargs="+", type=int, default=DEFAULT_MIN_SPHERES)
    args = parser.parse_args()
    minimum_component = min(args.min_spheres)

    for merge_distance in args.merge_distances:
        print(f"\nBuilding candidates for merge_distance={merge_distance:g}")
        train = build_candidates("chen11", merge_distance, minimum_component)
        test = build_candidates("coach420", merge_distance, minimum_component)
        print(
            f"preprocess: train {train.elapsed:.1f}s, test {test.elapsed:.1f}s; "
            f"test eligible={test.eligible}, SASA errors={test.sasa_errors}, "
            f"other errors={test.other_errors}"
        )
        evaluate_grid(merge_distance, train, test, args.min_spheres)


if __name__ == "__main__":
    main()
