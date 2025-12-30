#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

from ase import Atoms
from ase.io import read, write

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
        prog=prog or "sgl build supercell",
        description=(
            "Construct a supercell by replicating an atomic structure along the cell "
            "vectors a, b, and c. Supports single- or multi-frame extxyz input and "
            "reads/writes from files or standard input/output."
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
        "--repl",
        nargs=3,
        type=int,
        metavar=("NA", "NB", "NC"),
        default=(1, 1, 1),
        help=(
            "Replicate along cell vectors a, b, c "
            "(NA NB NC; default: 1 1 1)."
        ),
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

    na, nb, nc = args.repl
    if na < 1 or nb < 1 or nc < 1:
        parser.exit(
            status=1,
            message="Error: --repl values must be positive integers ≥ 1\n",
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

    # -------- 3) replicate atoms --------
    out_frames: List[Atoms] = [atoms_ref.repeat((na, nb, nc)) for atoms_ref in in_frames]

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

