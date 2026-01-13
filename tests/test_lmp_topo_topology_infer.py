import numpy as np
import pytest

from ase import Atoms

from simuglue.topology.infer import (
    infer_bonds_by_distance,
    infer_angles_from_adjacency,
    infer_dihedrals_from_bonds,
)

from simuglue.topology import features as _feat


def _atoms_water_like(*, cell=10.0, pbc=True) -> Atoms:
    """Simple 3-atom molecule with two bonds and one angle.

    Geometry:
      0: O at origin
      1: H at x
      2: H at y

    Bond lengths: 0.96 Å, angle: 90°.
    """
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (0.96, 0.0, 0.0),
            (0.0, 0.96, 0.0),
        ],
        cell=[cell, cell, cell],
        pbc=[pbc, pbc, pbc],
    )
    atoms.arrays["type"] = np.array([1, 2, 2], dtype=int)
    return atoms


def _atoms_chain_4_trans(*, cell=20.0, pbc=True) -> Atoms:
    """4-atom chain with one dihedral close to 180° (trans)."""
    s = 1.0 / np.sqrt(2.0)
    atoms = Atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            (-s, +s, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0 + s, -s, 0.0),
        ],
        cell=[cell, cell, cell],
        pbc=[pbc, pbc, pbc],
    )
    atoms.arrays["type"] = np.array([1, 1, 1, 1], dtype=int)
    return atoms


def test_infer_bonds_by_distance_water_like():
    atoms = _atoms_water_like()

    topo, neighbors = infer_bonds_by_distance(
        atoms,
        rc_list=[0.96],
        drc=0.05,
        deduplicate=True,
    )

    assert topo.bonds == [(0, 1), (0, 2)]
    assert neighbors == [[1, 2], [0], [0]]

    # MIC lengths should be close to 0.96
    lengths = _feat.bond_lengths(atoms, topo.bonds)
    assert lengths is not None
    assert len(lengths) == 2
    assert all(abs(r - 0.96) < 1e-6 for r in lengths)


def test_infer_angles_from_adjacency_water_like():
    atoms = _atoms_water_like()
    topo, neighbors = infer_bonds_by_distance(atoms, rc_list=[0.96], drc=0.05)

    angles = infer_angles_from_adjacency(neighbors, sort=True)
    assert angles == [(1, 0, 2)]


def test_infer_dihedrals_from_bonds_chain_4():
    atoms = _atoms_chain_4_trans()
    topo, neighbors = infer_bonds_by_distance(atoms, rc_list=[1.0], drc=0.05)

    # Bond inference should find the chain edges only
    assert topo.bonds == [(0, 1), (1, 2), (2, 3)]

    diheds = infer_dihedrals_from_bonds(topo.bonds, neighbors, sort=True)
    assert diheds == [(0, 1, 2, 3)]


def test_infer_bonds_by_distance_negative_drc_raises():
    atoms = _atoms_water_like(pbc=False)
    with pytest.raises(ValueError):
        infer_bonds_by_distance(atoms, rc_list=[1.0], drc=-1.0)
