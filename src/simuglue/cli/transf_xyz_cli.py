#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
from ase import Atoms
from ase.io import read, write
from simuglue.transform.linear import apply_transform
from simuglue.mechanics.voigt import voigt_to_cart
from simuglue.cli._xyz_io import _iter_frames

def _parse_strain(s: str, voigt: bool) -> np.ndarray:
    """
    Minimal parser.

    - If voigt=True: expect 6 numbers: 'xx yy zz yz xz xy'.
    - Else: expect exactly 3 rows separated by ';', each with 3 numbers:
            'a b c; d e f; g h i'.
    """
    s = s.strip()
    if voigt:
        strain = [float(x) for x in s.replace(",", " ").split()]
        if len(strain) != 6:
            raise ValueError("With --voigt, provide exactly 6 numbers: 'xx yy zz yz xz xy'.")

        # convert strain to linear transformer
        I_voigt = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        F = strain + I_voigt

        return voigt_to_cart(F)

    # Non-voigt: strictly 'a b c; d e f; g h i'
    rows = [r.strip() for r in s.split(";") if r.strip()]
    if len(rows) != 3:
        raise ValueError("Provide exactly 3 rows separated by ';', e.g. '1 0 0; 0 1 0; 0 0 1'.")
    mat: list[float] = []
    for r in rows:
        parts = [float(x) for x in r.replace(",", " ").split()]
        if len(parts) != 3:
            raise ValueError("Each row must contain exactly 3 numbers.")
        mat.extend(parts)

    strain = np.asarray(mat, dtype=float).reshape(3, 3)
    F = strain + np.eye(3)
    return F

def _parse_frames_arg(frames_arg: str | None) -> str | int | None:
    if frames_arg is None:
        return None
    s = frames_arg.strip().lower()
    return "all" if s == "all" else int(s)

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply a linear deformation gradient to an extxyz (single or multi-frame).")
    p.add_argument("xyz", help="Input extxyz file.")
    p.add_argument("strain", help="Either 'a b c; d e f; g h i' (semicolon-separated rows) or 6 nums with --voigt.")
    p.add_argument("--frames", default=None, help="Frame index (int) or 'all'. Default: first frame only.")
    p.add_argument("--output", "-o", default="o.xyz", help="Output extxyz file (single or multi-frame).")
    p.add_argument("--voigt", action="store_true",
                   help="Interpret --strain tensor in Voigt notation [xx yy zz yz xz xy].")
    return p

def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    xyz_path = Path(args.xyz)
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")

    F = _parse_strain(args.strain, voigt=args.voigt)
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
