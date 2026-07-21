"""Tests for alpha-sphere tessellation helpers."""

import biotite.structure as struc  # type: ignore
import numpy as np

from pocketeer.core.tessellation import _sanitize_for_sasa


def test_sanitize_for_sasa_handles_noncanonical_atom_names():
    atomarray = struc.AtomArray(6)
    atomarray.coord = np.zeros((6, 3))
    atomarray.res_name = np.array(["MET", "THR", "MET", "GLN", "LYS", "CYS"])
    atomarray.atom_name = np.array(["SE", "P", "CN", "OE", "CX", "OD"])
    atomarray.element = np.array(["SE", "P", "C", "O", "C", "O"])

    sanitized, radii = _sanitize_for_sasa(atomarray)

    # The original annotations are used by every other detector stage.
    assert atomarray.atom_name.tolist() == ["SE", "P", "CN", "OE", "CX", "OD"]
    # Match invalid names to a canonical atom with the same leading element
    # where possible. THR has no phosphorus atom, so it uses an element radius.
    assert sanitized.atom_name.tolist() == ["SD", "P", "C", "O", "C", "O"]
    assert np.all(np.isfinite(radii))
    assert np.all(radii > 0)


def test_sanitize_for_sasa_keeps_canonical_names():
    atomarray = struc.AtomArray(4)
    atomarray.coord = np.zeros((4, 3))
    atomarray.res_name = np.array(["ALA"] * 4)
    atomarray.atom_name = np.array(["N", "CA", "C", "O"])
    atomarray.element = np.array(["N", "C", "C", "O"])

    sanitized, radii = _sanitize_for_sasa(atomarray)

    assert sanitized.atom_name.tolist() == atomarray.atom_name.tolist()
    assert np.all(np.isfinite(radii))
    assert np.all(radii > 0)
