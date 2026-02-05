import importlib
from pathlib import Path

import gzip
import numpy as np
import pytest

@pytest.fixture
def cli():
    return importlib.import_module("simuglue.cli.lmp_phonon_dispersion")


@pytest.fixture
def fake_atoms():
    ase = pytest.importorskip("ase")
    from ase import Atoms  # noqa

    a, b, c = 3.0, 4.0, 20.0
    Na, Nb, Nc = 2, 2, 1

    # same 2-atom basis as in core tests
    Hprim = np.array([[a, 0.0, 0.0],
                      [0.0, b, 0.0],
                      [0.0, 0.0, c]], dtype=float)
    Hsup = np.array([[Na * a, 0.0, 0.0],
                     [0.0, Nb * b, 0.0],
                     [0.0, 0.0, Nc * c]], dtype=float)

    basis = [
        (1, np.array([0.0, 0.0, 0.0])),
        (2, np.array([0.5, 0.5, 0.0])),
    ]

    positions, types = [], []
    for ic in range(Nc):
        for ib in range(Nb):
            for ia in range(Na):
                R = ia * Hprim[0] + ib * Hprim[1] + ic * Hprim[2]
                for t, u in basis:
                    pos = R + u[0] * Hprim[0] + u[1] * Hprim[1]
                    positions.append(pos)
                    types.append(t)

    atoms = Atoms(symbols=["C"] * len(types), positions=np.array(positions), cell=Hsup, pbc=True)
    atoms.set_array("type", np.array(types, dtype=int))
    return atoms, (Na, Nb, Nc)


def test_cli_main_writes_output(cli, fake_atoms, monkeypatch, tmp_path):
    atoms, (Na, Nb, Nc) = fake_atoms

    # Monkeypatch the LAMMPS reader used by the CLI to return our atoms
    if hasattr(cli, "read_lammps_data"):
        monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: atoms)
    else:
        pytest.skip("CLI module does not expose read_lammps_data; adjust monkeypatch target.")

    natom = len(atoms)
    n = 3 * natom

    dynmat = np.eye(n, dtype=float)
    dyn_path = tmp_path / "dyn.npy"
    np.save(dyn_path, dynmat)

    kpoints = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.0],
    ], dtype=float)
    k_path = tmp_path / "k.dat"
    np.savetxt(k_path, kpoints)

    out_path = tmp_path / "disp.tsv"

    argv = [
        f"{Na},{Nb},{Nc}",
        "-s", "dummy.data",       # ignored due to monkeypatch
        "-d", str(dyn_path),
        "-k", str(k_path),
        "-o", str(out_path),
        "--output-format", "with_k",
        "--k-format", "2pi_over_a",
        "-u", "Thz",
    ]

    rc = cli.main(argv)
    assert rc == 0
    assert out_path.exists()

    # Load output; first line might be header if you write one
    data = np.loadtxt(out_path)
    # with_k: 3 k columns + nmodes
    # nbasis=2 => nmodes=6
    assert data.shape == (2, 3 + 6)


def test_load_kpoints_two_columns(cli, tmp_path):
    p = tmp_path / "k2.dat"
    np.savetxt(p, np.array([[0.0, 0.1], [0.2, 0.3]]))
    k = cli.load_kpoints(p)
    assert k.shape == (2, 3)
    assert np.allclose(k[:, 2], 0.0)


def test_load_kpoints_more_than_three_columns(cli, tmp_path):
    p = tmp_path / "k5.dat"
    np.savetxt(p, np.array([[0, 1, 2, 3, 4]], dtype=float))
    k = cli.load_kpoints(p)
    assert k.shape == (1, 3)
    assert np.allclose(k[0], [0, 1, 2])


def test_load_dynmat_npy(cli, tmp_path):
    M = np.eye(12, dtype=float)
    p = tmp_path / "dyn.npy"
    np.save(p, M)
    out = cli.load_dynmat(p, expected_size=M.size)
    assert out.shape == M.shape
    assert np.allclose(out, M)


def test_load_dynmat_npz_with_dynmat_key(cli, tmp_path):
    M = np.eye(9, dtype=float)
    p = tmp_path / "dyn.npz"
    np.savez(p, dynmat=M)
    out = cli.load_dynmat(p, expected_size=M.size)
    assert out.shape == (9, 9)
    assert np.allclose(out, M)


def test_load_dynmat_npz_first_key_fallback(cli, tmp_path):
    M = np.eye(9, dtype=float)
    p = tmp_path / "dyn.npz"
    np.savez(p, something_else=M)
    out = cli.load_dynmat(p, expected_size=M.size)
    assert out.shape == (9, 9)
    assert np.allclose(out, M)


def test_load_dynmat_ascii_gz(cli, tmp_path):
    M = np.eye(6, dtype=float)
    p = tmp_path / "dyn.txt.gz"
    with gzip.open(p, "wt") as f:
        np.savetxt(f, M)

    out = cli.load_dynmat(p, expected_size=M.size)
    assert out.shape == (6, 6)
    assert np.allclose(out, M)


def test_load_dynmat_wrong_expected_size_raises(cli, tmp_path):
    M = np.eye(6, dtype=float)
    p = tmp_path / "dyn.npy"
    np.save(p, M)
    with pytest.raises(ValueError):
        cli.load_dynmat(p, expected_size=M.size + 1)


def test_load_dynmat_non_square_raises(cli, tmp_path):
    # Create 10 numbers; expected_size=10 passes size check, then fails perfect-square check
    vec = np.arange(10, dtype=float)
    p = tmp_path / "dyn.txt"
    np.savetxt(p, vec[:, None])
    with pytest.raises(ValueError):
        cli.load_dynmat(p, expected_size=10)


def test_write_dispersion_formats(cli, tmp_path):
    k = np.array([[0.0, 0.0, 0.0],
                  [0.1, 0.2, 0.0]])
    freqs = np.ones((2, 6), dtype=float)

    out_with_k = tmp_path / "with_k.tsv"
    cli.write_dispersion(out_with_k, k, freqs, output_format="with_k")
    text = out_with_k.read_text()
    assert text.lstrip().startswith("#")  # header line
    data = np.loadtxt(out_with_k)
    assert data.shape == (2, 3 + 6)

    out_freq = tmp_path / "freq.tsv"
    cli.write_dispersion(out_freq, k, freqs, output_format="freq_only")
    text2 = out_freq.read_text()
    assert not text2.lstrip().startswith("#")
    data2 = np.loadtxt(out_freq)
    assert data2.shape == (2, 6)


def test_cli_main_invalid_cells_string(cli):
    with pytest.raises(ValueError):
        cli.main(["not,a,csv", "-s", "x", "-d", "y", "-k", "z", "-o", "out.tsv"])

