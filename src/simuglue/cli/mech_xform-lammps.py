#!/usr/bin/env python3
from __future__ import annotations
import sys
import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from simuglue.transform.linear import apply_transform
from simuglue.io.matrix_3x3 import parse_3x3


# ---------- IO helpers ----------
def _read_lmp(src: str | Path, style: str, units: str) -> Atoms:
    with ExitStack() as stack:
        if src == "-":
            fh = sys.stdin
        else:
            path = Path(src)
            if not path.exists():
                raise FileNotFoundError(f"Input not found: {path}")
            fh = stack.enter_context(path.open("r"))
        return read_lammps_data(fh, style=style, units=units)

def _write_lmp(dst: str | Path, atoms: Atoms, atom_style: str, units: str, force_skew: bool) -> None:
    with ExitStack() as stack:
        if dst == "-":
            fh = sys.stdout
        else:
            fh = stack.enter_context(Path(dst).open("w"))
        write_lammps_data(fh, atoms, atom_style=atom_style, units=units, force_skew=force_skew)


# ---------- CLI ----------
def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Apply a linear deformation gradient to a LAMMPS data file. Use '-' for stdin/stdout."
    )
    p.add_argument("-i", "--input", default="-",
                   help="Input LAMMPS data path or '-' for stdin (default: '-')")
    p.add_argument(
        "--F",
        required=True,
        help="Deformation gradient as 'xx xy xz; yx yy yz; zx zy zz' (semicolon-separated rows).",
    )
    p.add_argument(
        "--output", "-o",
        default="-",
        help="Output LAMMPS data path or '-' for stdout (default: '-')",
    )
    p.add_argument(
        "--style",
        default="atomic",
        help="LAMMPS atom style for read/write (default: atomic).",
    )
    p.add_argument(
        "--units",
        default="metal",
        help="LAMMPS units for read/write (default: metal).",
    )
    p.add_argument(
        "--force-skew",
        action="store_true",
        help="Force triclinic (skew) cell in writer.",
    )
    return p


def main(argv: Optional[list[str]] = None, prog: str | None = None) -> int:
    args = build_parser(prog=prog).parse_args(argv)

    # Parse F and validate
    F = parse_3x3(args.F)
    if F.shape != (3, 3):
        raise ValueError(f"Transformer must be 3x3, got {F.shape}")

    # Read → Transform → Write
    atoms_ref = _read_lmp(args.input, style=args.style, units=args.units)
    atoms_def = apply_transform(atoms_ref, F)
    _write_lmp(args.output, atoms_def, atom_style=args.style, units=args.units, force_skew=args.force_skew)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

