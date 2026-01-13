# test_build_supercell_incr_array.py
from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms

# Adjust this import to your actual module path/file name.
from simuglue.cli.build_supercell import _repl_ordered, _apply_increment_array_posthoc


def test_incr_array_z_blocks_multiatom_abc():
    """
    Replicate 1x1x3 and increment only along z (c).
    Expect each z-slab to get +0, +1, +2 added to the per-atom array.
    """
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]],
        cell=[1.0, 1.0, 1.0],
        pbc=True,
    )
    atoms.arrays["mol-id"] = np.array([10, 20], dtype=np.int64)

    na, nb, nc = 1, 1, 3
    order = "abc"

    out = _repl_ordered(atoms, na, nb, nc, order=order)
    _apply_increment_array_posthoc(
        out,
        base_n=len(atoms),
        na=na, nb=nb, nc=nc,
        order=order,
        array_name="mol-id",
        inc_x=False, inc_y=False, inc_z=True,
    )

    expected = np.array([10, 20, 11, 21, 12, 22], dtype=np.int64)
    np.testing.assert_array_equal(out.arrays["mol-id"], expected)


def test_incr_array_respects_order_for_x_increment():
    """
    Use a single atom so the output array equals the per-image offsets.
    Replicate 2x2x1 and increment only along x (a).
    Order 'abc' should give ia sequence [0,1,0,1]
    Order 'bac' should give ia sequence [0,0,1,1]
    """
    atoms = Atoms(
        numbers=[14],
        positions=[[0.1, 0.2, 0.3]],
        cell=[1.0, 1.0, 1.0],
        pbc=True,
    )
    atoms.arrays["mol-id"] = np.array([0], dtype=np.int64)

    na, nb, nc = 2, 2, 1

    # order = abc
    out_abc = _repl_ordered(atoms, na, nb, nc, order="abc")
    _apply_increment_array_posthoc(
        out_abc,
        base_n=len(atoms),
        na=na, nb=nb, nc=nc,
        order="abc",
        array_name="mol-id",
        inc_x=True, inc_y=False, inc_z=False,
    )
    np.testing.assert_array_equal(out_abc.arrays["mol-id"], np.array([0, 1, 0, 1], dtype=np.int64))

    # order = bac
    out_bac = _repl_ordered(atoms, na, nb, nc, order="bac")
    _apply_increment_array_posthoc(
        out_bac,
        base_n=len(atoms),
        na=na, nb=nb, nc=nc,
        order="bac",
        array_name="mol-id",
        inc_x=True, inc_y=False, inc_z=False,
    )
    np.testing.assert_array_equal(out_bac.arrays["mol-id"], np.array([0, 0, 1, 1], dtype=np.int64))


def test_incr_array_missing_raises():
    atoms = Atoms(
        numbers=[1],
        positions=[[0.0, 0.0, 0.0]],
        cell=[1.0, 1.0, 1.0],
        pbc=True,
    )

    na, nb, nc = 1, 1, 2
    out = _repl_ordered(atoms, na, nb, nc, order="abc")

    with pytest.raises(ValueError, match="not found"):
        _apply_increment_array_posthoc(
            out,
            base_n=len(atoms),
            na=na, nb=nb, nc=nc,
            order="abc",
            array_name="mol-id",
            inc_x=False, inc_y=False, inc_z=True,
        )


def test_incr_array_non_1d_raises():
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.2]],
        cell=[1.0, 1.0, 1.0],
        pbc=True,
    )
    # 2D per-atom array (N,2) should be rejected by the increment helper
    atoms.arrays["mol-id"] = np.array([[1, 2], [3, 4]], dtype=np.int64)

    na, nb, nc = 1, 1, 2
    out = _repl_ordered(atoms, na, nb, nc, order="abc")

    with pytest.raises(ValueError, match="must be 1D"):
        _apply_increment_array_posthoc(
            out,
            base_n=len(atoms),
            na=na, nb=nb, nc=nc,
            order="abc",
            array_name="mol-id",
            inc_x=False, inc_y=False, inc_z=True,
        )

