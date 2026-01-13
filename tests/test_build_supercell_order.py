# tests/test_build_supercell_order.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

from simuglue.cli.build_supercell import main


def _make_single_atom_extxyz(path: Path) -> None:
    a0 = Atoms(
        symbols=["H"],
        positions=[[0.0, 0.0, 0.0]],
        cell=[2.0, 3.0, 4.0],  # orthorhombic; makes expected shifts unambiguous
        pbc=True,
    )
    write(path, a0, format="extxyz")


def _read_single_frame(path: Path) -> Atoms:
    frames = read(path, format="extxyz", index=":")
    assert isinstance(frames, list)
    assert len(frames) == 1
    return frames[0]


def _sorted_positions(atoms: Atoms) -> np.ndarray:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    # primary sort by x, then y, then z
    idx = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
    return pos[idx]


def _diag_cell_lengths(atoms: Atoms) -> tuple[float, float, float]:
    cell = atoms.cell.array
    return float(cell[0, 0]), float(cell[1, 1]), float(cell[2, 2])


def test_supercell_repl_order_affects_atom_order(tmp_path: Path):
    in_file = tmp_path / "in.extxyz"
    _make_single_atom_extxyz(in_file)

    out_default = tmp_path / "out_default.extxyz"
    out_abc = tmp_path / "out_abc.extxyz"
    out_bac = tmp_path / "out_bac.extxyz"

    # Replicate 2x2x1
    base_args = ["-i", str(in_file), "--repl", "2", "2", "1", "--overwrite"]

    # 1) default (no --order)
    rc = main([*base_args, "-o", str(out_default)])
    assert rc == 0

    # 2) explicit abc should match default semantics
    rc = main([*base_args, "-o", str(out_abc), "--order", "abc"])
    assert rc == 0

    # 3) different order should change ordering
    rc = main([*base_args, "-o", str(out_bac), "--order", "bac"])
    assert rc == 0

    a_def = _read_single_frame(out_default)
    a_abc = _read_single_frame(out_abc)
    a_bac = _read_single_frame(out_bac)

    # Same sizes
    assert len(a_def) == len(a_abc) == len(a_bac) == 1 * 2 * 2 * 1

    # Same cell scaling
    a, b, c = _diag_cell_lengths(a_def)
    assert a == pytest.approx(2.0 * 2)
    assert b == pytest.approx(3.0 * 2)
    assert c == pytest.approx(4.0 * 1)
    assert _diag_cell_lengths(a_abc) == pytest.approx((a, b, c))
    assert _diag_cell_lengths(a_bac) == pytest.approx((a, b, c))

    # Default and explicit abc should be identical ordering
    assert np.allclose(a_def.get_positions(), a_abc.get_positions())

    # Geometry should be identical as a set (ordering ignored)
    assert np.allclose(_sorted_positions(a_abc), _sorted_positions(a_bac))

    # But ordering should differ
    assert not np.allclose(a_abc.get_positions(), a_bac.get_positions())

    # Strong, order-specific check:
    # For a single atom at the origin, the first incremental shift should follow the
    # fastest-varying axis (first in --order).
    pos_abc = np.asarray(a_abc.get_positions(), float)
    pos_bac = np.asarray(a_bac.get_positions(), float)

    d_abc = pos_abc[1] - pos_abc[0]
    d_bac = pos_bac[1] - pos_bac[0]

    assert np.allclose(d_abc, [2.0, 0.0, 0.0])  # 'a' fastest -> +a
    assert np.allclose(d_bac, [0.0, 3.0, 0.0])  # 'b' fastest -> +b

