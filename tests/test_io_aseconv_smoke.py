from __future__ import annotations
import subprocess
from pathlib import Path

import pytest
from ase.io import read


def run_aseconv(tmp_path, args):
    """Helper: run `sgl io aseconv` in tmp dir."""
    # If your `sgl` is not on PATH in CI, adjust to call the module directly.
    result = subprocess.run(
        ["sgl", "io", "aseconv", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"aseconv failed:\nCMD: sgl io aseconv {' '.join(args)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def test_extxyz_roundtrip(tmp_path):
    """extxyz -> extxyz sanity check."""
    in_xyz = tmp_path / "in.xyz"
    out_xyz = tmp_path / "out.xyz"

    # minimal extxyz: 2 atoms + cell
    in_xyz.write_text(
        "2\n"
        "Lattice=\"5 0 0 0 5 0 0 0 5\" Properties=species:S:1:pos:R:3\n"
        "H 0.0 0.0 0.0\n"
        "He 1.0 1.0 1.0\n",
        encoding="utf-8",
    )

    run_aseconv(tmp_path, [
        "-i", str(in_xyz),
        "-o", str(out_xyz),
    ])

    a_in = read(in_xyz)
    a_out = read(out_xyz)

    assert len(a_in) == len(a_out)
    assert a_in.get_chemical_symbols() == a_out.get_chemical_symbols()
    assert a_in.get_cell().volume == pytest.approx(a_out.get_cell().volume)


def test_extxyz_to_lammps_data_and_back(tmp_path):
    """extxyz -> lammps-data -> extxyz: preserves natoms & symbols."""
    in_xyz = tmp_path / "in.xyz"
    data_file = tmp_path / "out.data"
    back_xyz = tmp_path / "back.xyz"

    in_xyz.write_text(
        "4\n"
        "Lattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3\n"
        "Bi 0 0 0\n"
        "Bi 1 0 0\n"
        "Se 0 1 0\n"
        "Se 0 0 1\n",
        encoding="utf-8",
    )

    # extxyz -> lammps-data (use atomic style for simpler test data)
    run_aseconv(tmp_path, [
        "-i", str(in_xyz),
        "--iformat", "extxyz",
        "-o", str(data_file),
        "--oformat", "lammps-data",
        "--lammps-style", "atomic",
        "--overwrite",
    ])

    assert data_file.exists()

    # lammps-data -> extxyz
    run_aseconv(tmp_path, [
        "-i", str(data_file),
        "--iformat", "lammps-data",
        "--lammps-style", "atomic",
        "-o", str(back_xyz),
        "--oformat", "extxyz",
        "--overwrite",
    ])

    a_in = read(in_xyz)
    a_back = read(back_xyz)

    assert len(a_in) == len(a_back)
    assert sorted(a_in.get_chemical_symbols()) == sorted(a_back.get_chemical_symbols())


def test_lammps_dump_to_extxyz(tmp_path):
    """lammps-dump-text -> extxyz basic check."""
    dump = tmp_path / "in.dump"
    out_xyz = tmp_path / "out.xyz"

    dump.write_text(
        "ITEM: TIMESTEP\n"
        "0\n"
        "ITEM: NUMBER OF ATOMS\n"
        "2\n"
        "ITEM: BOX BOUNDS pp pp pp\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "0.0 10.0\n"
        "ITEM: ATOMS id type x y z\n"
        "1 1 0.0 0.0 0.0\n"
        "2 1 1.0 0.0 0.0\n",
        encoding="utf-8",
    )

    run_aseconv(tmp_path, [
        "-i", str(dump),
        "--iformat", "lammps-dump-text",
        "-o", str(out_xyz),
        "--oformat", "extxyz",
    ])

    a = read(out_xyz, format="extxyz")
    assert len(a) == 2
    xs = a.get_positions()[:, 0]
    assert xs.min() == pytest.approx(0.0)
    assert xs.max() == pytest.approx(1.0)

