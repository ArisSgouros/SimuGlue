import importlib
import numpy as np
import pytest


def import_core():
    candidates = ["phonon.dispersion", "simuglue.phonon.dispersion"]
    for name in candidates:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    raise ModuleNotFoundError(f"Could not import core dispersion module. Tried: {candidates}")


@pytest.fixture
def core():
    return import_core()


@pytest.fixture
def ase():
    return pytest.importorskip("ase")


def test_duplicate_types_in_cell_raises(core, ase):
    from ase import Atoms

    # 1 cell, 2 atoms with the SAME type => violates "one per type per cell"
    a, b, c = 3.0, 4.0, 20.0
    pos = np.array([[0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0]], dtype=float)

    atoms = Atoms(
        symbols=["C", "C"],
        positions=pos,
        cell=[[a, 0, 0], [0, b, 0], [0, 0, c]],
        pbc=True,
    )
    atoms.set_array("type", np.array([1, 1], dtype=int))

    dynmat = np.eye(3 * len(atoms), dtype=float)
    kpoints = np.array([[0.0, 0.0, 0.0]], dtype=float)

    with pytest.raises(RuntimeError):
        core.compute_phonon_dispersion(
            atoms, dynmat, kpoints,
            cells=(1, 1, 1),
            freq_units="Thz",
        )

