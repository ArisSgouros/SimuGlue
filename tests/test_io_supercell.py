# tests/test_io_supercell.py
from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np
from ase import Atoms
from ase.io import read, write

from simuglue.cli.io_supercell import main  # noqa: F401


def _make_two_frame_extxyz(path: Path) -> None:
    """
    Create a 2-frame extxyz with different natoms and different cells
    so we can test frame selection and cell scaling.
    """
    # Frame 0: 2 atoms, orthorhombic cell
    a0 = Atoms(
        symbols=["H", "He"],
        positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[2.0, 3.0, 4.0],
        pbc=True,
    )

    # Frame 1: 3 atoms, different cell
    a1 = Atoms(
        symbols=["H", "He", "Li"],
        positions=[[0.0, 0.0, 0.0], [0.4, 0.2, 0.1], [1.1, 0.1, 0.3]],
        cell=[5.0, 6.0, 7.0],
        pbc=True,
    )

    write(path, [a0, a1], format="extxyz")


def _diag_cell_lengths(atoms: Atoms) -> tuple[float, float, float]:
    """
    For orthorhombic cells, return (a, b, c) = diagonal lengths.
    """
    cell = atoms.cell.array
    return float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])


def test_supercell_default_first_frame_only(tmp_path: Path):
    in_file = tmp_path / "in.extxyz"
    out_file = tmp_path / "out.extxyz"
    _make_two_frame_extxyz(in_file)

    # Default: first frame only, replicate 2x1x1
    rc = main(["-i", str(in_file), "-o", str(out_file), "--repl", "2", "1", "1", "--overwrite"])
    assert rc == 0

    out = read(out_file, format="extxyz", index=":")
    # Should output a single frame (first frame only)
    assert isinstance(out, list)
    assert len(out) == 1

    out0 = out[0]
    assert len(out0) == 2 * 2  # original natoms 2, replicated by 2 along a
    a, b, c = _diag_cell_lengths(out0)
    assert a == pytest.approx(2.0 * 2)
    assert b == pytest.approx(3.0)
    assert c == pytest.approx(4.0)


def test_supercell_frames_all(tmp_path: Path):
    in_file = tmp_path / "in.extxyz"
    out_file = tmp_path / "out.extxyz"
    _make_two_frame_extxyz(in_file)

    rc = main(["-i", str(in_file), "-o", str(out_file), "--repl", "1", "2", "3", "--frames", "all", "--overwrite"])
    assert rc == 0

    out = read(out_file, format="extxyz", index=":")
    assert isinstance(out, list)
    assert len(out) == 2  # two frames preserved

    out0, out1 = out

    # Frame 0: natoms 2, replicate 1x2x3 => factor 6
    assert len(out0) == 2 * 6
    a, b, c = _diag_cell_lengths(out0)
    assert a == pytest.approx(2.0)
    assert b == pytest.approx(3.0 * 2)
    assert c == pytest.approx(4.0 * 3)

    # Frame 1: natoms 3, replicate 1x2x3 => factor 6
    assert len(out1) == 3 * 6
    a, b, c = _diag_cell_lengths(out1)
    assert a == pytest.approx(5.0)
    assert b == pytest.approx(6.0 * 2)
    assert c == pytest.approx(7.0 * 3)


def test_supercell_frames_index(tmp_path: Path):
    in_file = tmp_path / "in.extxyz"
    out_file = tmp_path / "out.extxyz"
    _make_two_frame_extxyz(in_file)

    # Select second frame only (index 1), replicate 3x1x1
    rc = main(["-i", str(in_file), "-o", str(out_file), "--repl", "3", "1", "1", "--frames", "1", "--overwrite"])
    assert rc == 0

    out = read(out_file, format="extxyz", index=":")
    assert isinstance(out, list)
    assert len(out) == 1

    out1 = out[0]
    # second frame had 3 atoms, factor 3
    assert len(out1) == 3 * 3
    a, b, c = _diag_cell_lengths(out1)
    assert a == pytest.approx(5.0 * 3)
    assert b == pytest.approx(6.0)
    assert c == pytest.approx(7.0)


def test_supercell_repl_must_be_positive(tmp_path: Path):
    in_file = tmp_path / "in.extxyz"
    out_file = tmp_path / "out.extxyz"
    _make_two_frame_extxyz(in_file)

    # Expect non-zero exit due to invalid repl (0 is not allowed)
    with pytest.raises(SystemExit) as excinfo:
        main(["-i", str(in_file), "-o", str(out_file), "--repl", "0", "1", "1", "--overwrite"])
    assert excinfo.value.code == 1

