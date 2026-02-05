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
    Build a replicated orthorhombic supercell. Basis encoded by atoms.arrays["type"].
    Also sets atoms.arrays["id"] = 1..N (then optionally shuffles atom *order*).
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
    Apply atom permutation to a 3N x 3N dynmat when the atom list was permuted by atom_perm:
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


def test_scrambled_ids_only_has_no_effect(core, ase):
    """
    If only atoms.arrays['id'] is scrambled (but positions/types order unchanged),
    result must be identical because the algorithm is geometry-based.
    """
    atoms = make_atoms_supercell(Na=2, Nb=2, Nc=1, shuffle_atoms=False)
    natom = len(atoms)
    n = 3 * natom

    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    dynmat = (A + A.T) / 2.0  # symmetric dense => k-dependent behavior

    kpoints = np.array([[0.0, 0.0, 0.0],
                        [0.13, 0.07, 0.0]], dtype=float)

    ref = core.compute_phonon_dispersion(
        atoms, dynmat, kpoints,
        cells=(2, 2, 1),
        freq_units="Thz",
        k_format="2pi_over_a",
    )

    # scramble IDs only
    ids = atoms.arrays["id"].copy()
    rng = np.random.default_rng(1)
    rng.shuffle(ids)
    atoms2 = atoms.copy()
    atoms2.set_array("id", ids)

    got = core.compute_phonon_dispersion(
        atoms2, dynmat, kpoints,
        cells=(2, 2, 1),
        freq_units="Thz",
        k_format="2pi_over_a",
    )

    assert np.allclose(ref.frequencies, got.frequencies, atol=1e-10, rtol=1e-10)


def test_scrambled_atom_order_with_consistent_dynmat_is_invariant(core, ase):
    """
    Atom list is permuted (scrambled IDs/order), but dynmat is permuted consistently.
    The algorithm should recover a consistent canonical ordering and give identical dispersion.
    """
    atoms = make_atoms_supercell(Na=2, Nb=2, Nc=1, shuffle_atoms=False)
    natom = len(atoms)
    n = 3 * natom

    rng = np.random.default_rng(2)
    A = rng.standard_normal((n, n))
    dynmat = (A + A.T) / 2.0  # symmetric dense => k-dependent

    kpoints = np.array([[0.0, 0.0, 0.0],
                        [0.21, 0.17, 0.0]], dtype=float)

    ref = core.compute_phonon_dispersion(
        atoms, dynmat, kpoints,
        cells=(2, 2, 1),
        freq_units="Thz",
        k_format="2pi_over_a",
    )

    # permute atoms list order
    rng = np.random.default_rng(123)
    perm = rng.permutation(natom)

    atoms_scr = atoms.copy()
    atoms_scr.positions[:] = atoms.positions[perm]
    atoms_scr.arrays["type"] = atoms.arrays["type"][perm]
    atoms_scr.arrays["id"] = atoms.arrays["id"][perm]

    # IMPORTANT: permute dynmat consistently with the same atom permutation
    dynmat_scr = permute_dynmat_by_atom_perm(dynmat, perm)

    got = core.compute_phonon_dispersion(
        atoms_scr, dynmat_scr, kpoints,
        cells=(2, 2, 1),
        freq_units="Thz",
        k_format="2pi_over_a",
    )

    assert np.allclose(ref.frequencies, got.frequencies, atol=1e-10, rtol=1e-10)


def test_atoms_dynmat_mismatch_changes_result_for_k_nonzero(core, ase):
    """
    Realistic failure mode:
      - atoms list order is scrambled
      - dynmat corresponds to a different ordering
    We pick Nb=4 and a permutation NOT in the dihedral symmetry D4 so the mismatch
    cannot be "absorbed" by ring symmetries.

    Expect: at a nonzero ky, frequencies change.
    """
    from ase import Atoms

    # Supercell: Na=1, Nb=4, Nc=1 ; nbasis=1 (type=1)
    a, b, c = 3.0, 4.0, 20.0
    Na, Nb, Nc = 1, 4, 1
    Hsup = np.array([[Na * a, 0.0, 0.0],
                     [0.0, Nb * b, 0.0],
                     [0.0, 0.0, Nc * c]], dtype=float)

    # Canonical atoms in cell order y = 0, b, 2b, 3b
    pos = np.array([[0.0, 0.0 * b, 0.0],
                    [0.0, 1.0 * b, 0.0],
                    [0.0, 2.0 * b, 0.0],
                    [0.0, 3.0 * b, 0.0]], dtype=float)

    atoms = Atoms(symbols=["C"] * 4, positions=pos, cell=Hsup, pbc=True)
    atoms.set_array("type", np.array([1, 1, 1, 1], dtype=int))
    atoms.set_array("id", np.array([1, 2, 3, 4], dtype=int))

    natom = len(atoms)
    n = 3 * natom  # 12

    # Build a symmetric dynmat with a non-translation-invariant pattern in x DOF
    D = np.zeros((n, n), dtype=float)

    # distinct onsite x stiffness
    kx = [2.0, 3.5, 5.0, 8.0]
    for i, kii in enumerate(kx):
        D[3 * i + 0, 3 * i + 0] = kii

    # inter-cell x-x couplings (arbitrary, symmetric, nonuniform)
    def couple(i, j, val):
        D[3 * i + 0, 3 * j + 0] = val
        D[3 * j + 0, 3 * i + 0] = val

    couple(0, 1, 0.40)
    couple(0, 2, 0.10)
    couple(0, 3, -0.07)
    couple(1, 2, 0.33)
    couple(1, 3, 0.05)
    couple(2, 3, 0.21)

    # make y,z stable
    for i in range(natom):
        D[3 * i + 1, 3 * i + 1] = 7.0
        D[3 * i + 2, 3 * i + 2] = 9.0

    # kpoints: Gamma + nonzero ky (reciprocal coordinates)
    kpoints = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.25, 0.0],   # ky = 1/4
    ], dtype=float)

    ref = core.compute_phonon_dispersion(
        atoms, D, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        k_format="reciprocal",
    )

    # Scramble atom list order with a permutation NOT in D4 (not a shift or reflection)
    perm = np.array([0, 2, 3, 1], dtype=int)
    atoms_bad = atoms.copy()
    atoms_bad.positions[:] = atoms.positions[perm]
    atoms_bad.arrays["type"] = atoms.arrays["type"][perm]
    atoms_bad.arrays["id"] = atoms.arrays["id"][perm]

    # dynmat NOT permuted -> mismatch
    got = core.compute_phonon_dispersion(
        atoms_bad, D, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        k_format="reciprocal",
    )

    # Nonzero k should differ for this constructed mismatch
    assert not np.allclose(ref.frequencies[1], got.frequencies[1], atol=1e-7, rtol=1e-7)

