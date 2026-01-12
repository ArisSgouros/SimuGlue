import math
import numpy as np

from ase import Atoms

from simuglue.topology.infer import (
    infer_bonds_by_distance,
    infer_angles_from_adjacency,
    infer_dihedrals_from_bonds,
)
from simuglue.topology import features as feat


def _triclinic_cell() -> np.ndarray:
    # a=(10,0,0), b=(2.1,9,0), c=(1.3,0.7,8)
    return np.array(
        [
            [10.0, 0.0, 0.0],
            [2.1, 9.0, 0.0],
            [1.3, 0.7, 8.0],
        ]
    )


def test_triclinic_bond_inference_across_skewed_boundary():
    cell = _triclinic_cell()
    b = cell[1]

    r0 = np.array([1.0, 1.0, 1.0])
    # Place r1 such that the MIC vector is (-0.3, -0.1, 0)
    r1 = r0 + b + np.array([-0.3, -0.1, 0.0])

    atoms = Atoms(
        symbols=["H", "H"],
        positions=[r0, r1],
        cell=cell,
        pbc=[True, True, True],
    )
    atoms.arrays["type"] = np.array([1, 1], dtype=int)

    topo, neighbors, lengths = infer_bonds_by_distance(
        atoms,
        rc_list=[0.32],
        drc=0.05,
        deduplicate=True,
        return_lengths=True,
    )

    assert topo.bonds == [(0, 1)]
    assert neighbors == [[1], [0]]

    # MIC bond length should be ~sqrt(0.3^2 + 0.1^2)
    assert lengths is not None and len(lengths) == 1
    assert abs(lengths[0] - math.sqrt(0.1)) < 1e-6


def test_triclinic_angle_across_skewed_boundary_matches_expected_theta():
    cell = _triclinic_cell()
    b = cell[1]

    rj = np.array([1.0, 1.0, 1.0])
    ri = rj + np.array([0.3, 0.0, 0.0])
    rk = rj + b + np.array([-0.3, -0.1, 0.0])

    atoms = Atoms(
        symbols=["H", "H", "H"],
        positions=[rj, ri, rk],
        cell=cell,
        pbc=[True, True, True],
    )
    atoms.arrays["type"] = np.array([1, 1, 1], dtype=int)

    topo, neighbors, _ = infer_bonds_by_distance(
        atoms,
        rc_list=[0.30, 0.32],
        drc=0.08,
        deduplicate=True,
        return_lengths=False,
    )

    # bonds should connect j(0)-i(1) and j(0)-k(2)
    assert set(topo.bonds) == {(0, 1), (0, 2)}

    angles = infer_angles_from_adjacency(neighbors, sort=True)
    assert angles == [(1, 0, 2)]

    theta = feat.angle_thetas_deg(atoms, angles)[0]

    # expected from v1=(0.3,0,0) and v2=(-0.3,-0.1,0)
    v1 = np.array([0.3, 0.0, 0.0])
    v2 = np.array([-0.3, -0.1, 0.0])
    expected = math.degrees(math.acos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))

    assert abs(theta - expected) < 1e-3


def test_triclinic_dihedral_invariant_under_lattice_translation():
    cell = _triclinic_cell()
    b = cell[1]

    atoms = Atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            (1.0, 1.0, 1.0),
            (1.8, 1.2, 1.1),
            (2.6, 1.1, 1.4),
            (3.2, 1.6, 1.9),
        ],
        cell=cell,
        pbc=[True, True, True],
    )
    atoms.arrays["type"] = np.array([1, 1, 1, 1], dtype=int)

    dihed = (0, 1, 2, 3)
    phi1 = feat.dihedral_phis_rad(atoms, [dihed])[0]

    atoms_shift = atoms.copy()
    atoms_shift.positions[3] = atoms_shift.positions[3] + b  # same image under PBC
    phi2 = feat.dihedral_phis_rad(atoms_shift, [dihed])[0]

    # Compare modulo 2*pi
    d = math.atan2(math.sin(phi2 - phi1), math.cos(phi2 - phi1))
    assert abs(d) < 1e-8
