#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
from ase import Atoms
from ase.io import read
from simuglue.quantum_espresso.build_pwi import build_pwi_from_header
from simuglue.cli._xyz_io import _iter_frames

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate Quantum ESPRESSO input file(s) by combining a header "
            "with CELL_PARAMETERS and ATOMIC_POSITIONS read from an extxyz file."
        )
    )
    p.add_argument("--header", required=True, help="QE header file (without CELL_PARAMETERS/ATOMIC_POSITIONS).")
    p.add_argument("--xyz", required=True, help="Input extxyz trajectory or structure.")
    p.add_argument(
        "--frames",
        default=None,
        help="Frame index (int) or 'all'. Default: first frame only."
    )
    p.add_argument("--outdir", default=".", help="Directory for output .in files. Default: current directory.")
    p.add_argument("-o", "--output", help="Path to output (default: stem of xyz file).")
    return p

def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    header_path = Path(args.header)
    xyz_path = Path(args.xyz)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not header_path.exists():
        raise FileNotFoundError(f"Header not found: {header_path}")
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")

    # Read header text once (passed to builder below)
    header_text = header_path.read_text(encoding="utf-8")

    # Normalize frames argument
    if args.frames is None:
        frames_arg: int | str | None = None
    else:
        s = str(args.frames).strip().lower()
        frames_arg = "all" if s == "all" else int(s)

    generated = 0
    for i, atoms in _iter_frames(xyz_path, frames_arg):
        # Decide base filename
        outfile = args.output if args.output else xyz_path.stem
        if frames_arg:
            outfile += f"_{i:05d}.in"
        outpath = outdir / outfile

        qe_text = build_pwi_from_header(header_text, atoms)
        outpath.write_text(qe_text, encoding="utf-8")
        generated += 1

    # Friendly summary
    if generated == 1:
        print(f"Generated 1 QE input in {outdir.resolve()}")
    else:
        print(f"Generated {generated} QE inputs in {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
