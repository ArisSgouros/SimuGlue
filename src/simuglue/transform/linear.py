#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
from ase import Atoms
from typing import Sequence

def apply_transform(atoms_ref, F: np.ndarray, scale_atoms=True):
    """
    Affine map using F with ASE's row-vector cell convention:
    C_new = C_old @ F.T, positions remapped consistently.
    """
    atoms_transformed = atoms_ref.copy()
    C = atoms_transformed.get_cell().array  # rows = a,b,c
    C_new = C @ F.T                         # right-multiply by F^T
    atoms_transformed.set_cell(C_new, scale_atoms=scale_atoms)
    return atoms_transformed


def voigt_to_cart(v: Sequence[float]) -> np.ndarray:
    """
    Convert a 6×1 array [xx, yy, zz, yz, xz, xy] to a 3×3 matrix:
        [[xx, xy, xz],
         [0 , yy, yz],
         [0 , 0 , zz]]
    """
    if len(v) != 6:
        raise ValueError("Expected 6 elements: [xx, yy, zz, yz, xz, xy].")
    xx, yy, zz, yz, xz, xy = map(float, v)
    return np.array([
        [xx, xy, xz],
        [0.0, yy, yz],
        [0.0, 0.0, zz],
    ], dtype=float)
