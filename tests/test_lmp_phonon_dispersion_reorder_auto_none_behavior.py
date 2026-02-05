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


def make_atoms_supercell(
    *,
    Na: int,
    Nb: int,
    Nc: int,
    a: float = 3.0,
    b: float = 4.0,
    c: float = 20.0,
    basis=((1, (0.0, 0.0, 0.0)), (2, (0.5, 0.5, 0.0))),
    shuffle_atoms: bool = False,
    seed: int = 123,
):
    """
    Orthorhombic supercell with a 2-atom basis encoded by atoms.arrays["type"].
    Also sets atoms.arrays["id"] = 1..N (useful for debugging/CLI --reorder id).
    """
    from ase import Atoms

    Hprim = np.array([[a, 0.0, 0.0],
                      [0.0, b, 0.0],
                      [0.0, 0.0, c]], dtype=float)

    Hsup = np.array([[Na * a, 0.0, 0.0],
                     [0.0, Nb * b, 0.0],
                     [0.0, 0.0, Nc * c]], dtype=float)

    positions = []
    types = []
    ids = []
    atom_id = 1
    for ic in range(Nc):
        for ib in range(Nb):
            for ia in range(Na):
                R = ia * Hprim[0] + ib * Hprim[1] + ic * Hprim[2]
                for t, u in basis:
                    u = np.asarray(u, dtype=float)
                    pos = R + u[0] * Hprim[0] + u[1] * Hprim[1] + u[2] * Hprim[2]
                    positions.append(pos)
                    types.append(int(t))
                    ids.append(atom_id)
                    atom_id += 1

    positions = np.asarray(positions, dtype=float)
    types = np.asarray(types, dtype=int)
    ids = np.asarray(ids, dtype=int)

    if shuffle_atoms:
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(types))
        positions = positions[p]
        types = types[p]
        ids = ids[p]

    atoms = Atoms(symbols=["C"] * len(types), positions=positions, cell=Hsup, pbc=True)
    atoms.set_array("type", types)
    atoms.set_array("id", ids)
    return atoms


def permute_dynmat_by_atom_perm(dynmat, atom_perm):
    """
    Permute a 3N x 3N dynmat consistent with atom reordering:
      new_atoms[i] = old_atoms[atom_perm[i]]
    """
    atom_perm = np.asarray(atom_perm, dtype=int)
    natom = atom_perm.size
    dynmat = np.asarray(dynmat, float)
    assert dynmat.shape == (3 * natom, 3 * natom)

    dof = np.empty(3 * natom, dtype=int)
    dof[0::3] = 3 * atom_perm + 0
    dof[1::3] = 3 * atom_perm + 1
    dof[2::3] = 3 * atom_perm + 2
    return dynmat[np.ix_(dof, dof)]


@pytest.fixture
def core():
    return import_core()


@pytest.fixture
def ase():
    return pytest.importorskip("ase")


def test_reorder_auto_and_none_behavior(core, ase):
    """
    Demonstrate and validate reorder policies:

      1) Canonical atoms:
         - reorder='auto' and reorder='none' both succeed and agree.

      2) Scrambled atoms (atom list order changed), with dynmat permuted consistently:
         - reorder='auto' succeeds and matches the canonical reference.
         - reorder='none' rejects because atoms are not in canonical [cell][basis] order.
    """
    Na, Nb, Nc = 2, 2, 1

    atoms_canon = make_atoms_supercell(Na=Na, Nb=Nb, Nc=Nc, shuffle_atoms=False)
    natom = len(atoms_canon)
    n = 3 * natom

    # Symmetric dense dynmat so k dependence exists and mapping mistakes matter.
    rng = np.random.default_rng(7)
    A = rng.standard_normal((n, n))
    dynmat = (A + A.T) / 2.0

    kpoints = np.array([[0.0, 0.0, 0.0],
                        [0.21, 0.17, 0.0]], dtype=float)

    # --- Case 1: canonical atoms: auto and none must match
    ref_auto = core.compute_phonon_dispersion(
        atoms_canon, dynmat, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        k_format="2pi_over_a",
        reorder="auto",
    )
    ref_none = core.compute_phonon_dispersion(
        atoms_canon, dynmat, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        k_format="2pi_over_a",
        reorder="none",
    )
    assert np.allclose(ref_auto.frequencies, ref_none.frequencies, atol=1e-10, rtol=1e-10)

    # --- Case 2: scramble atom list order, permute dynmat consistently
    rng = np.random.default_rng(123)
    perm = rng.permutation(natom)

    atoms_scr = atoms_canon.copy()
    atoms_scr.positions[:] = atoms_canon.positions[perm]
    atoms_scr.arrays["type"] = atoms_canon.arrays["type"][perm]
    atoms_scr.arrays["id"] = atoms_canon.arrays["id"][perm]
    dynmat_scr = permute_dynmat_by_atom_perm(dynmat, perm)

    # (A) auto should canonicalize and recover the same result as canonical case
    got_auto = core.compute_phonon_dispersion(
        atoms_scr, dynmat_scr, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        k_format="2pi_over_a",
        reorder="auto",
    )
    assert np.allclose(ref_auto.frequencies, got_auto.frequencies, atol=1e-10, rtol=1e-10)

    # (B) none should reject scrambled atoms because tensor reshape requires canonical [cell][basis] ordering
    with pytest.raises(RuntimeError):
        core.compute_phonon_dispersion(
            atoms_scr, dynmat_scr, kpoints,
            cells=(Na, Nb, Nc),
            freq_units="Thz",
            k_format="2pi_over_a",
            reorder="none",
        )

