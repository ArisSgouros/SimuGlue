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
from simuglue.cli._transf_util import _parse_3x3

def _parse_frames_arg(frames_arg: str | None) -> str | int | None:
    if frames_arg is None:
        return None
    s = frames_arg.strip().lower()
    return "all" if s == "all" else int(s)

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply a linear deformation gradient to an extxyz (single or multi-frame).")
    p.add_argument("xyz", help="Input extxyz file.")
    p.add_argument("--F", help="Deformation gradient: 'xx xy xz; yx yy yz; zx zy zz' (semicolon-separated rows)")
    p.add_argument("--frames", default=None, help="Frame index (int) or 'all'. Default: first frame only.")
    p.add_argument("--output", "-o", default="o.xyz", help="Output extxyz file (single or multi-frame).")
    return p


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    xyz_path = Path(args.xyz)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")

    F = _parse_3x3(args.F)
    if F.shape != (3, 3):
        raise ValueError(f"Transformer must be 3x3, got {F.shape}")

    frames_sel = _parse_frames_arg(args.frames)

    out_frames: List[Atoms] = []
    for _, atoms_ref in _iter_frames(xyz_path, frames_sel):
        atoms_def = apply_transform(atoms_ref, F)
        out_frames.append(atoms_def)

    # Single output file (multi-frame if >1)
    write(args.output, out_frames if len(out_frames) > 1 else out_frames[0], format="extxyz")
    print(f"Wrote {len(out_frames)} frame(s) to {Path(args.output).resolve()}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
