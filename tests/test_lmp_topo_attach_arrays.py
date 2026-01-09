import numpy as np

from ase import Atoms

from simuglue.topology.core import attach_topology_arrays_to_atoms
from simuglue.topology.infer import infer_bonds_by_distance
from simuglue.topology.typing import TypingOptions, type_bonds, type_angles


def _atoms_water_like(*, cell=10.0, pbc=True) -> Atoms:
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


def test_attach_arrays_bonds_and_angles_water_like():
    atoms = _atoms_water_like()

    topo, neighbors, _ = infer_bonds_by_distance(atoms, rc_list=[0.96], drc=0.05)
    topo.angles = [(1, 0, 2)]

    atom_tag = {1: "O", 2: "H"}
    type_bonds(atoms, topo, atom_tag, opts=TypingOptions())
    type_angles(atoms, topo, atom_tag, opts=TypingOptions())

    attach_topology_arrays_to_atoms(atoms, topo)

    # bonds: stored only on at1 (0) as "at2(type),..."
    assert atoms.arrays["bonds"].tolist() == ["1(1),2(1)", "_", "_"]

    # angles: stored only on central atom (at2=0) as "at1-at3(type),..."
    assert atoms.arrays["angles"].tolist() == ["1-2(1)", "_", "_"]

    # no dihedrals -> should not exist
    assert "dihedrals" not in atoms.arrays


def test_attach_arrays_no_bonds_does_not_create_arrays():
    atoms = _atoms_water_like()
    # empty topo
    topo, _neighbors, _ = infer_bonds_by_distance(atoms, rc_list=[], drc=0.05)
    assert topo.bonds == []

    attach_topology_arrays_to_atoms(atoms, topo)
    assert "bonds" not in atoms.arrays
    assert "angles" not in atoms.arrays
    assert "dihedrals" not in atoms.arrays
