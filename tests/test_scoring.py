"""Tests for pocket scoring."""

import numpy as np

from pocketeer.core.scoring import score_pocket
from pocketeer.core.types import AlphaSphere, Pocket


def test_score_pocket_is_probability_and_uses_aromatic_residues():
    spheres = [
        AlphaSphere(
            sphere_id=index,
            center=np.array([float(index), 0.0, 0.0]),
            radius=3.8,
            mean_sasa=9.0,
            atom_indices=[0, 1, 2, 3],
        )
        for index in range(40)
    ]

    def pocket(residue_name: str) -> Pocket:
        return Pocket(
            pocket_id=0,
            spheres=spheres,
            centroid=np.zeros(3),
            volume=1_200.0,
            score=0.0,
            residues=[("A", index, residue_name) for index in range(1, 16)],
            mask=np.ones(4, dtype=bool),
        )

    aliphatic_score = score_pocket(pocket("ALA"))
    aromatic_score = score_pocket(pocket("PHE"))

    assert 0.0 < aliphatic_score < 1.0
    assert 0.0 < aromatic_score < 1.0
    assert aromatic_score > aliphatic_score
