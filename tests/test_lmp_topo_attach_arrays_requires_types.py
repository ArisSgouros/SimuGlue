import numpy as np
import pytest

from ase import Atoms

from simuglue.topology.core import attach_topology_arrays_to_atoms
from simuglue.topology.topo import Topo


def _atoms_n(n: int, *, cell: float = 10.0) -> Atoms:
    atoms = Atoms(
        symbols=["H"] * n,
        positions=[(float(i), 0.0, 0.0) for i in range(n)],
        cell=[cell, cell, cell],
        pbc=[False, False, False],
    )
    atoms.arrays["type"] = np.ones(n, dtype=int)
    return atoms


def test_attach_bonds_requires_bond_types_when_bonds_exist():
    """Bug: if bonds exist but bond_types is None, the function silently writes '_' for all atoms."""
    atoms = _atoms_n(2)
    topo = Topo(bonds=[(0, 1)])  # no bond_types

    with pytest.raises(ValueError, match="bond_types"):
        attach_topology_arrays_to_atoms(atoms, topo)


def test_attach_bonds_requires_matching_lengths():
    """Bug: zip(bond_types, bonds) silently truncates if lengths mismatch."""
    atoms = _atoms_n(3)
    topo = Topo(bonds=[(0, 1), (1, 2)], bond_types=[1])  # missing 2nd type

    with pytest.raises(ValueError, match="bond_types"):
        attach_topology_arrays_to_atoms(atoms, topo)


def test_attach_angles_requires_angle_types_when_angles_exist():
    atoms = _atoms_n(3)
    topo = Topo(
        bonds=[(0, 1), (1, 2)], bond_types=[1, 1],
        angles=[(0, 1, 2)],
    )

    with pytest.raises(ValueError, match="angle_types"):
        attach_topology_arrays_to_atoms(atoms, topo)


def test_attach_dihedrals_requires_dihedral_types_when_dihedrals_exist():
    atoms = _atoms_n(4)
    topo = Topo(
        bonds=[(0, 1), (1, 2), (2, 3)], bond_types=[1, 1, 1],
        dihedrals=[(0, 1, 2, 3)],
    )

    with pytest.raises(ValueError, match="dihedral_types"):
        attach_topology_arrays_to_atoms(atoms, topo)
