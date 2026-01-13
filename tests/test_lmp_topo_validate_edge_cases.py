import pytest

from simuglue.topology.topo import Topo


def test_topo_validate_allows_zero_atoms_when_empty():
    """Bug: Topo.validate currently rejects natoms<=0 even if topology is empty.

    In practice, reading/processing an empty structure should be a no-op.
    """
    topo = Topo(bonds=[], angles=[], dihedrals=[])
    # Expected: no exception
    topo.validate(0)
