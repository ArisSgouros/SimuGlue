import importlib
import numpy as np
import pytest

@pytest.fixture
def core():
    return importlib.import_module("simuglue.phonon.dispersion")

@pytest.fixture
def ase_atoms():
    ase = pytest.importorskip("ase")
    from ase import Atoms  # noqa

    # Primitive cell vectors (orthorhombic, simple)
    a, b, c = 3.0, 4.0, 20.0
    Na, Nb, Nc = 2, 2, 1

    # 2-atom basis, encoded by LAMMPS "type"
    # basis fractional coordinates in the primitive cell
    basis = [
        (1, np.array([0.0, 0.0, 0.0])),
        (2, np.array([0.5, 0.5, 0.0])),
    ]

    # primitive vectors (as rows for ASE)
    Hprim = np.array([[a, 0.0, 0.0],
                      [0.0, b, 0.0],
                      [0.0, 0.0, c]], dtype=float)

    # supercell vectors (rows)
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
                    pos = R + u[0] * Hprim[0] + u[1] * Hprim[1] + u[2] * Hprim[2]
                    positions.append(pos)
                    types.append(t)

    positions = np.array(positions, dtype=float)
    types = np.array(types, dtype=int)

    # Shuffle atoms so we test re-ordering logic
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(types))
    positions = positions[perm]
    types = types[perm]

    # Symbols can be anything; we mainly use arrays["type"] in your code path
    atoms = Atoms(symbols=["C"] * len(types), positions=positions, cell=Hsup, pbc=True)
    atoms.set_array("type", types)

    return atoms, (Na, Nb, Nc)


def test_wrap01(core):
    x = np.array([-1.2, -0.1, 0.0, 0.2, 1.0, 1.9])
    y = core.wrap01(x)
    assert np.all(y >= 0.0)
    assert np.all(y < 1.0)
    # specific checks
    assert np.isclose(y[0], 0.8)
    assert np.isclose(y[1], 0.9)
    assert np.isclose(y[4], 0.0)


def test_frequency_conversion_ratio(core):
    thz = core.frequency_conversion("Thz")
    cm1 = core.frequency_conversion("cm-1")
    # function uses exact THz->cm^-1 constant
    assert np.isclose(cm1 / thz, 33.35640951981521, rtol=0, atol=1e-12)


def test_build_cell_displacements_2d(core):
    da, db, dc = core.build_cell_displacements(3, 2, 1)
    assert da.shape == (6, 6)
    assert db.shape == (6, 6)
    assert dc.shape == (6, 6)
    assert np.all(dc == 0)

    # N=3: values in {-1,0,1}
    assert da.min() >= -1 and da.max() <= 1

    # N=2: values in {-1,0,1} after the even-N tie fix
    assert db.min() >= -1 and db.max() <= 1

    # Key invariants (these catch real bugs)
    assert np.all(da + da.T == 0)
    assert np.all(db + db.T == 0)


def test_infer_cell_and_basis_basic(core, ase_atoms):
    atoms, (Na, Nb, Nc) = ase_atoms
    pos = np.asarray(atoms.get_positions(), float)

    # Build cell_H in the convention used by core (columns)
    cell_H = np.asarray(atoms.cell.array, float).T

    cell, u, cell_lin = core.infer_cell_and_basis(pos, cell_H, Na, Nb, Nc)
    assert cell.shape == (len(atoms), 3)
    assert u.shape == (len(atoms), 3)
    assert cell_lin.shape == (len(atoms),)
    assert np.all(u >= 0.0) and np.all(u < 1.0)

    # cell_lin must lie in [0, ncell-1]
    ncell = Na * Nb * Nc
    assert cell_lin.min() >= 0
    assert cell_lin.max() <= ncell - 1


def test_compute_dispersion_identity_is_k_independent(core, ase_atoms):
    atoms, (Na, Nb, Nc) = ase_atoms
    natom = len(atoms)
    n = 3 * natom

    dynmat = np.eye(n, dtype=float)

    # three arbitrary k-points; result should be identical for identity dynmat
    kpoints = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.0],
        [0.33, 0.0, 0.0],
    ], dtype=float)

    res = core.compute_phonon_dispersion(
        atoms,
        dynmat,
        kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        k_format="2pi_over_a",
        imag_policy="clip",
        hermitian_project=True,
        return_eigenvectors=False,
    )

    assert res.kpoints.shape == (3, 3)
    assert res.frequencies.shape[0] == 3
    assert res.frequencies.shape[1] == 3 * res.nbasis

    # Identity dynmat => all eigenvalues 1 => all frequencies == w_conv
    w_conv = core.frequency_conversion("Thz")
    assert np.allclose(res.frequencies, w_conv, rtol=0, atol=1e-12)


def test_imag_policy_signed_vs_clip(core, ase_atoms):
    atoms, (Na, Nb, Nc) = ase_atoms
    natom = len(atoms)
    n = 3 * natom

    dynmat = -np.eye(n, dtype=float)
    kpoints = np.array([[0.0, 0.0, 0.0]], dtype=float)
    w_conv = core.frequency_conversion("Thz")

    res_clip = core.compute_phonon_dispersion(
        atoms, dynmat, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        imag_policy="clip",
    )
    assert np.allclose(res_clip.frequencies, 0.0, atol=1e-12)

    res_signed = core.compute_phonon_dispersion(
        atoms, dynmat, kpoints,
        cells=(Na, Nb, Nc),
        freq_units="Thz",
        imag_policy="signed",
    )
    assert np.allclose(res_signed.frequencies, -w_conv, atol=1e-12)


def test_bad_occupancy_raises(core):
    ase = pytest.importorskip("ase")
    from ase import Atoms  # noqa

    # Make Na=1,Nb=1,Nc=1 but give only one type => violates "one per type per cell" expectation
    a, b, c = 3.0, 4.0, 20.0
    atoms = Atoms(symbols=["C"], positions=[[0, 0, 0]], cell=[[a, 0, 0], [0, b, 0], [0, 0, c]], pbc=True)
    atoms.set_array("type", np.array([1], dtype=int))

    natom = len(atoms)
    dynmat = np.eye(3 * natom, dtype=float)
    kpoints = np.array([[0.0, 0.0, 0.0]], dtype=float)

    # nbasis inferred is 1; occupancy is 1, so this by itself is OK.
    # Force an occupancy mismatch by lying about supercell replication:
    with pytest.raises(RuntimeError):
        core.compute_phonon_dispersion(
            atoms, dynmat, kpoints,
            cells=(2, 1, 1),  # claims 2 cells along a, but we only have 1 atom total
            freq_units="Thz",
        )

