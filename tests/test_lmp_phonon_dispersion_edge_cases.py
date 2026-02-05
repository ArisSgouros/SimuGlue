import importlib
import numpy as np
import pytest


def import_core():
    """
    Core module is at src/phonon/dispersion.py.
    Depending on packaging, it might import as phonon.dispersion or simuglue.phonon.dispersion.
    """
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
    shuffle: bool = True,
    seed: int = 123,
):
    """
    Build a replicated orthorhombic supercell with a 2-atom basis encoded by atoms.arrays["type"].
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
    for ic in range(Nc):
        for ib in range(Nb):
            for ia in range(Na):
                R = ia * Hprim[0] + ib * Hprim[1] + ic * Hprim[2]
                for t, u in basis:
                    u = np.asarray(u, dtype=float)
                    pos = R + u[0] * Hprim[0] + u[1] * Hprim[1] + u[2] * Hprim[2]
                    positions.append(pos)
                    types.append(int(t))

    positions = np.asarray(positions, dtype=float)
    types = np.asarray(types, dtype=int)

    if shuffle:
        rng = np.random.default_rng(seed)
        p = rng.permutation(len(types))
        positions = positions[p]
        types = types[p]

    atoms = Atoms(symbols=["C"] * len(types), positions=positions, cell=Hsup, pbc=True)
    atoms.set_array("type", types)
    return atoms


def permute_dynmat_by_atom_perm(dynmat, atom_perm):
    """
    Permute a 3N x 3N dynmat when atom ordering changes by atom_perm:
      new_atoms[i] = old_atoms[atom_perm[i]].
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



def test_wrap_mi_even_N_range(core):
    d = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=int)
    out = core.wrap_mi(d, 4)
    # Allowed values for N=4 minimum-image with sign-preserving ties: -2,-1,0,1,2
    assert out.min() >= -2
    assert out.max() <= 2
    assert set(np.unique(out)).issubset({-2, -1, 0, 1, 2})



def test_k_to_cartesian_formats_equivalent_orthorhombic(core):
    # For orthorhombic cell, reciprocal coords (1,0,0) -> 2pi/a along x
    a, b, c = 3.0, 4.0, 20.0
    Hprim_cols = np.array([[a, 0, 0],
                           [0, b, 0],
                           [0, 0, c]], dtype=float).T  # columns

    k_frac = np.array([1.0, 0.0, 0.0], dtype=float)
    k_2pi_over_a = np.array([1.0, 0.0, 0.0], dtype=float)

    k_cart_rec = core.k_to_cartesian(k_frac, Hprim_cols, k_format="reciprocal", a_lat=a)
    k_cart_2pi = core.k_to_cartesian(k_2pi_over_a, Hprim_cols, k_format="2pi_over_a", a_lat=a)

    assert np.allclose(k_cart_rec, k_cart_2pi, atol=1e-12)


def test_compute_returns_eigenvectors_shape_and_orthonormal(core, ase):
    atoms = make_atoms_supercell(Na=2, Nb=2, Nc=1, shuffle=True)
    natom = len(atoms)
    n = 3 * natom

    dynmat = np.eye(n, dtype=float)
    kpoints = np.array([[0.0, 0.0, 0.0],
                        [0.2, 0.1, 0.0]], dtype=float)

    res = core.compute_phonon_dispersion(
        atoms, dynmat, kpoints,
        cells=(2, 2, 1),
        return_eigenvectors=True,
        freq_units="Thz",
    )

    assert res.eigenvectors is not None
    nk, nmodes, nmodes2 = res.eigenvectors.shape
    assert nk == 2
    assert nmodes == nmodes2 == 3 * res.nbasis

    V = res.eigenvectors[0]
    I = V.conj().T @ V
    assert np.allclose(I, np.eye(nmodes), atol=1e-10)


def test_hermitian_project_toggle_stability(core, ase):
    """
    If dynmat is symmetric PSD, D_k should be ~Hermitian already.
    hermitian_project on/off should not change frequencies much.
    """
    atoms = make_atoms_supercell(Na=2, Nb=2, Nc=1, shuffle=True)
    natom = len(atoms)
    n = 3 * natom

    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    dynmat = (A.T @ A) / n  # symmetric PSD
    kpoints = np.array([[0.0, 0.0, 0.0],
                        [0.13, 0.07, 0.0]], dtype=float)

    r_on = core.compute_phonon_dispersion(
        atoms, dynmat, kpoints,
        cells=(2, 2, 1),
        hermitian_project=True,
        freq_units="Thz",
    )
    r_off = core.compute_phonon_dispersion(
        atoms, dynmat, kpoints,
        cells=(2, 2, 1),
        hermitian_project=False,
        freq_units="Thz",
    )

    assert np.allclose(r_on.frequencies, r_off.frequencies, atol=1e-8, rtol=1e-8)


def test_build_cell_displacements_are_antisymmetric_for_even_N(core):
    da, db, dc = core.build_cell_displacements(4, 2, 1)

    assert np.all(da + da.T == 0), "da must be antisymmetric"
    assert np.all(db + db.T == 0), "db must be antisymmetric"
    assert np.all(dc == 0)


def test_ordering_invariance_with_nontrivial_diagonal(core, ase):
    """
    Strong test: if atoms are shuffled and dynmat is shuffled consistently,
    the returned spectrum must match the canonical spectrum.
    """
    Na, Nb, Nc = 2, 2, 1
    atoms_canonical = make_atoms_supercell(Na=Na, Nb=Nb, Nc=Nc, shuffle=False)

    natom = len(atoms_canonical)
    n = 3 * natom

    # nbasis=2 => nmodes=6. Repeat per cell so total dof is ncell*6
    nmodes = 6
    base = np.array([1.0, 2.0, 4.0, 7.0, 11.0, 16.0], dtype=float)

    ncell = Na * Nb * Nc
    assert natom == ncell * 2

    diag = np.empty(n, dtype=float)
    for icell in range(ncell):
        diag[icell * nmodes:(icell + 1) * nmodes] = base

    dynmat_canon = np.diag(diag)

    kpoints = np.array([[0.0, 0.0, 0.0],
                        [0.2, 0.1, 0.0]], dtype=float)

    ref = core.compute_phonon_dispersion(
        atoms_canonical, dynmat_canon, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
    )

    # Shuffle atoms and dynmat consistently
    rng = np.random.default_rng(123)
    perm = rng.permutation(natom)

    atoms_shuffled = atoms_canonical.copy()
    atoms_shuffled.positions[:] = atoms_canonical.positions[perm]
    atoms_shuffled.arrays["type"] = atoms_canonical.arrays["type"][perm]

    dynmat_shuffled = permute_dynmat_by_atom_perm(dynmat_canon, perm)

    got = core.compute_phonon_dispersion(
        atoms_shuffled, dynmat_shuffled, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
    )

    assert np.allclose(ref.frequencies, got.frequencies, atol=1e-12)


def test_kpoints_shape_validation(core, ase):
    atoms = make_atoms_supercell(Na=1, Nb=1, Nc=1, shuffle=False)
    dynmat = np.eye(3 * len(atoms))

    with pytest.raises(ValueError):
        core.compute_phonon_dispersion(atoms, dynmat, np.array([0.0, 0.0, 0.0]), cells=(1, 1, 1))

    with pytest.raises(ValueError):
        core.compute_phonon_dispersion(atoms, dynmat, np.zeros((5, 2)), cells=(1, 1, 1))


def test_dynmat_shape_validation(core, ase):
    atoms = make_atoms_supercell(Na=1, Nb=1, Nc=1, shuffle=False)
    kpoints = np.array([[0, 0, 0]], dtype=float)

    with pytest.raises(ValueError):
        core.compute_phonon_dispersion(atoms, np.eye(10), kpoints, cells=(1, 1, 1))


def test_bad_units_and_bad_a_lat(core, ase):
    atoms = make_atoms_supercell(Na=1, Nb=1, Nc=1, shuffle=False)
    dynmat = np.eye(3 * len(atoms))
    kpoints = np.array([[0, 0, 0]], dtype=float)

    with pytest.raises(ValueError):
        core.compute_phonon_dispersion(atoms, dynmat, kpoints, cells=(1, 1, 1), freq_units="GHz")

    with pytest.raises(ValueError):
        core.compute_phonon_dispersion(atoms, dynmat, kpoints, cells=(1, 1, 1), a_lat=0.0)

