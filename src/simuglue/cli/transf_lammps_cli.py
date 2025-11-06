#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from ase import Atoms
from simuglue.transform.linear import apply_transform
from simuglue.io.util_ase_lammps import read_lammps, write_lammps
from simuglue.cli._transf_util import _parse_3x3

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply a linear deformation gradient to a lammps data file.")
    p.add_argument("data_in", help="Input lammps data file.")
    p.add_argument("--F", required=True, help="Deformation gradient: 'xx xy xz; yx yy yz; zx zy zz' (semicolon-separated rows)")
    p.add_argument("--output", "-o", default="o.lammps", help="Output lammps file.")
    return p

def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    data_in_path = Path(args.data_in)
    if not data_in_path.exists():
        raise FileNotFoundError(f"XYZ not found: {data_in_path}")

    F = _parse_3x3(args.F)
    if F.shape != (3, 3):
        raise ValueError(f"Transformer must be 3x3, got {F.shape}")

    atoms_ref = read_lammps(data_in_path, style="atomic")
    atoms_def = apply_transform(atoms_ref, F)
    write_lammps(atoms_def, f"{args.output}", style="atomic", units="metal", force_skew=True)

if __name__ == "__main__":
    main()
