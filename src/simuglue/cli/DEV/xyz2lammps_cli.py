#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read, write
from simuglue.transform.linear import apply_transform
from simuglue.cli._xyz_io import _iter_frames

def _parse_frames_arg(frames_arg: str | None) -> str | int | None:
    if frames_arg is None:
        return None
    s = frames_arg.strip().lower()
    return "all" if s == "all" else int(s)

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply a linear deformation gradient to an extxyz (single or multi-frame).")
    p.add_argument("xyz", help="Input extxyz file.")
    p.add_argument("--frames", default=None, help="Frame index (int) or 'all'. Default: first frame only.")
    p.add_argument("--output", "-o", default="o.xyz", help="Output extxyz file (single or multi-frame).")
    return p

def write_lammps(
    atoms: Atoms,
    output_file: str,
    style: str = "atomic",
    units: str = "metal",
    force_skew: bool = False,
    write_velocities: bool = True,
) -> Path:
    """
    Write a LAMMPS data file.

    Parameters
    ----------
    atoms : Atoms
        Structure to write.
    output_file : str
        Path to output .data
    style : str
        LAMMPS atom style ('atomic', 'charge', 'molecular', 'full').
    units : str
        LAMMPS units ('metal', 'real', 'si', 'lj', 'cgs', 'electron').
    force_skew : bool
        Allow triclinic (skewed) cell output if present.
    write_velocities : bool
        Include velocities if present.

    Returns
    -------
    Path
    """
    out_path = Path(output_file)
    has_vel = atoms.get_velocities() is not None

    kw = {
        "format": "lammps-data",
        "atom_style": style,
        "units": units,
        "force_skew": force_skew,
    }

    if has_vel and write_velocities:
        try:
            write(str(out_path), atoms, **kw, velocities=True)
        except TypeError:
            # Older ASE versions may not accept 'velocities' kwarg
            write(str(out_path), atoms, **kw)
    else:
        write(str(out_path), atoms, **kw)

    return out_path

def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    xyz_path = Path(args.xyz)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")

    frames_sel = _parse_frames_arg(args.frames)

    out_frames: List[Atoms] = []
    for _, atoms_ref in _iter_frames(xyz_path, frames_sel):
        out_frames.append(atoms_ref)

    # Single output file (multi-frame if >1)
    #write(args.output, out_frames if len(out_frames) > 1 else out_frames[0], format="extxyz")
    print(f"Wrote {len(out_frames)} frame(s) to {Path(args.output).resolve()}")
    write_lammps(out_frames[0], f"{args.output}", style="atomic", units="metal", force_skew=True)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
