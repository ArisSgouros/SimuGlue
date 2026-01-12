import numpy as np
import pytest

from ase import Atoms

from simuglue.topology.infer import infer_bonds_by_distance


def _dimer(*, r: float, cell: float = 10.0, pbc: bool = False) -> Atoms:
    atoms = Atoms(
        symbols=["H", "H"],
        positions=[(0.0, 0.0, 0.0), (float(r), 0.0, 0.0)],
        cell=[cell, cell, cell],
        pbc=[pbc, pbc, pbc],
    )
    atoms.arrays["type"] = np.array([1, 1], dtype=int)
    return atoms


def test_infer_bonds_drc_zero_should_match_exact_rc():
    """Bug: with drc=0 the current implementation rejects *all* pairs.

    In infer_bonds_by_distance(), the window check uses strict inequalities:
        d2 > rmin2 and d2 < rmax2

    If drc=0, then rmin2 == rmax2 == rc^2, so the condition is impossible.
    Expected behavior: drc=0 should allow matching exactly rc (within float tolerance).
    """
    atoms = _dimer(r=1.0)
    topo, neighbors, _ = infer_bonds_by_distance(atoms, rc_list=[1.0], drc=0.0, return_lengths=False)
    assert topo.bonds == [(0, 1)]
    assert neighbors[0] == [1] and neighbors[1] == [0]


@pytest.mark.parametrize(
    "r, rc, drc",
    [
        (0.9, 1.0, 0.1),  # lower boundary
        (1.1, 1.0, 0.1),  # upper boundary
    ],
)
def test_infer_bonds_should_include_window_boundaries(r, rc, drc):
    """Bug: strict inequalities exclude the window boundaries.

    If drc is a tolerance half-width, distances at rc±drc should be accepted.
    """
    atoms = _dimer(r=r)
    topo, _, _ = infer_bonds_by_distance(atoms, rc_list=[rc], drc=drc, return_lengths=False)
    assert topo.bonds == [(0, 1)]
