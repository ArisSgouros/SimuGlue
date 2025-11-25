#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.io import read, write

from simuglue.transform.linear import apply_transform
from simuglue.io.matrix_3x3 import parse_3x3

def _parse_frames_arg(frames_arg: str | None) -> str | int | None:
    """
    Parse --frames:
      - None     -> use first frame only
      - 'all'    -> all frames
      - '<int>'  -> that frame index (0-based)
    """
    if frames_arg is None:
        return None
    s = frames_arg.strip().lower()
    return "all" if s == "all" else int(s)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl mech deform-xyz",
        description=(
            "Apply a linear deformation gradient F to an extxyz structure "
            "(single or multi-frame). "
            "Reads/writes extxyz from file or stdin/stdout."
        ),
    )

    p.add_argument(
        "-i",
        "--input",
        default="-",
        help="Input extxyz file or '-' for stdin (default: '-').",
    )

    p.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output extxyz file or '-' for stdout (default: '-').",
    )

    p.add_argument(
        "--F",
        required=True,
        help="Deformation gradient as "
             "'F11 F12 F13; F21 F22 F23; F31 F32 F33'.",
    )

    p.add_argument(
        "--frames",
        default=None,
        help="Frame index (0-based int) or 'all'. "
             "Default: first frame only.",
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file (only when output is not '-').",
    )

    return p


def _select_frames_from_list(
    frames: List[Atoms],
    frames_sel: str | int | None,
) -> List[Atoms]:
    """Apply --frames selection to a list of Atoms."""
    if not frames:
        return []

    if frames_sel is None:
        # default: first frame only
        return [frames[0]]

    if frames_sel == "all":
        return frames

    # integer index
    idx = int(frames_sel)
    return [frames[idx]]


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    # Parse F
    try:
        F = parse_3x3(args.F)
    except Exception as exc:
        parser.exit(status=1, message=f"Error parsing F: {exc}\n")

    if F.shape != (3, 3):
        parser.exit(
            status=1,
            message=f"Error: deformation gradient must be 3x3, got {F.shape}\n",
        )

    frames_sel = _parse_frames_arg(args.frames)

    # -------- 1) read input frames (all) --------
    # Decide source (stdin vs file)
    if args.input == "-":
        src = sys.stdin
    else:
        in_path = Path(args.input)
        if not in_path.is_file():
            parser.exit(status=1, message=f"Input file not found: {in_path}\n")
        src = in_path

    # ASE read call
    try:
        images = read(src, format="extxyz", index=":")
    except Exception as exc:
        parser.exit(status=1, message=f"Failed to read extxyz: {exc}\n")

    # Normalize to list[Atoms]
    if isinstance(images, Atoms):
        all_frames: List[Atoms] = [images]
    else:
        all_frames = list(images)

    # -------- 2) apply selection --------
    in_frames = _select_frames_from_list(all_frames, frames_sel)

    if not in_frames:
        parser.exit(status=1, message="No frames selected or read; nothing to do.\n")

    # -------- 3) deform atoms --------
    out_frames: List[Atoms] = [apply_transform(atoms_ref, F) for atoms_ref in in_frames]

    # -------- 4) write output --------
    if args.output == "-":
        out_dst = sys.stdout
    else:
        out_path = Path(args.output)
        if out_path.exists() and not args.overwrite:
            parser.exit(status=1, message=f"Refusing to overwrite existing file: {out_path}\n")
        out_dst = out_path

    write(out_dst, out_frames, format="extxyz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

