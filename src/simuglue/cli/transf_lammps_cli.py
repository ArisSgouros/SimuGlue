#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from ase import Atoms
from simuglue.transform.linear import apply_transform
from simuglue.mechanics.voigt import voigt_to_cart
from simuglue.io.util_ase_lammps import read_lammps, write_lammps

def _parse_transformer(s: str) -> np.ndarray:
    """
    Minimal parser.

    - If voigt=True: expect 6 numbers: 'xx yy zz yz xz xy'.
    - Else: expect exactly 3 rows separated by ';', each with 3 numbers:
            'a b c; d e f; g h i'.
    """
    s = s.strip()
    strain = [float(x) for x in s.replace(",", " ").split()]
    if len(strain) != 6:
        raise ValueError("With --voigt, provide exactly 6 numbers: 'xx yy zz yz xz xy'.")

    # convert strain to linear transformer
    I_voigt = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    F = strain + I_voigt

    return voigt_to_cart(F)

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply a linear deformation gradient to a lammps data file.")
    p.add_argument("data_in", help="Input lammps data file.")
    p.add_argument("strain", help="Voigt notation: 'xx yy zz yz xz xy'.")
    p.add_argument("--output", "-o", default="o.lammps", help="Output lammps file.")
    return p

def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    data_in_path = Path(args.data_in)
    if not data_in_path.exists():
        raise FileNotFoundError(f"XYZ not found: {data_in_path}")

    atoms_ref = read_lammps(data_in_path, style="atomic")

    F = _parse_transformer(args.strain)
    if F.shape != (3, 3):
        raise ValueError(f"Transformer must be 3x3, got {F.shape}")

    atoms_def = apply_transform(atoms_ref, F)
    write_lammps(atoms_def, f"{args.output}", style="atomic", units="metal", force_skew=True)

if __name__ == "__main__":
    main()
