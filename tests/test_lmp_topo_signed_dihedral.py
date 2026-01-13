from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest


# Allow running tests without installing the package (src/ layout)
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))

from ase import Atoms

from simuglue.topology.features import dihedral_phis_rad
from simuglue.topology.topo import Topo
from simuglue.topology.typing import TypingOptions, type_dihedrals


def _atoms_for_signed_dihedral() -> Atoms:
    """Return 4 atoms with a non-zero, well-defined dihedral angle (~+45 deg).

    Geometry chosen so that phi != 0 and not near +/-180 to avoid sign/rounding ambiguities.
    """
    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # r0
            [1.0, 0.0, 0.0],  # r1
            [1.0, 1.0, 0.0],  # r2
            [0.0, 1.0, 1.0],  # r3
        ]
    )
    atoms = Atoms(symbols=["H", "H", "H", "H"], positions=positions)
    atoms.set_cell(np.eye(3) * 10.0)
    atoms.set_pbc((False, False, False))
    return atoms


def test_dihedral_signed_phi_flips_when_canonicalization_reverses() -> None:
    """If canonical tag order is obtained by reversing (i,j,k,l)->(l,k,j,i),
    signed phi must flip sign to remain consistent with the canonical representation.
    """
    atoms = _atoms_for_signed_dihedral()

    # Forward tag sequence will be N B N B (i=0..3)
    atoms.arrays["type"] = np.array([1, 2, 1, 2], dtype=int)
    atom_tags = {1: "N", 2: "B"}

    topo = Topo(dihedrals=[(0, 1, 2, 3)])

    opts = TypingOptions(
        diff_dihed_theta=True,
        diff_dihed_theta_abs=False,  # keep sign
        diff_dihed_theta_fmt="%.2f",
    )

    # Reference phi for the original ordering (0,1,2,3)
    phi_deg = math.degrees(dihedral_phis_rad(atoms, topo.dihedrals)[0])
    assert math.isfinite(phi_deg) and abs(phi_deg) > 1e-6

    type_dihedrals(atoms, topo, atom_tags, opts)
    assert topo.dihedral_tags is not None and len(topo.dihedral_tags) == 1

    tag = topo.dihedral_tags[0]
    tokens = tag.split()

    # Canonicalization should reverse the chemical sequence:
    # N B N B  ->  B N B N   (because 'B' < 'N')
    assert tokens[:4] == ["B", "N", "B", "N"]

    # With reversal canonicalization and signed mode, the numeric feature must flip sign
    expected = -phi_deg
    assert tokens[4] == (opts.diff_dihed_theta_fmt % expected)


def test_dihedral_signed_phi_kept_when_no_canonicalization_flip() -> None:
    """If canonical tag order does not require reversal, signed phi must be kept."""
    atoms = _atoms_for_signed_dihedral()

    # Forward tag sequence is already canonical: B N B N
    atoms.arrays["type"] = np.array([2, 1, 2, 1], dtype=int)
    atom_tags = {1: "N", 2: "B"}

    topo = Topo(dihedrals=[(0, 1, 2, 3)])

    opts = TypingOptions(
        diff_dihed_theta=True,
        diff_dihed_theta_abs=False,  # keep sign
        diff_dihed_theta_fmt="%.2f",
    )

    phi_deg = math.degrees(dihedral_phis_rad(atoms, topo.dihedrals)[0])
    assert math.isfinite(phi_deg) and abs(phi_deg) > 1e-6

    type_dihedrals(atoms, topo, atom_tags, opts)
    assert topo.dihedral_tags is not None and len(topo.dihedral_tags) == 1

    tag = topo.dihedral_tags[0]
    tokens = tag.split()

    assert tokens[:4] == ["B", "N", "B", "N"]
    assert tokens[4] == (opts.diff_dihed_theta_fmt % phi_deg)

