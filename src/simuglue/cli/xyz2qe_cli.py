#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
from ase import Atoms
from ase.io import read
from simuglue.quantum_espresso.build_pwi import build_pwi_from_header


def _iter_frames(xyz_path: Path, frames: str | int | None) -> Iterable[tuple[int, Atoms]]:
    """
    Yield (frame_index, Atoms) pairs.
    frames:
      - None  -> first frame only (index 0)
      - int   -> that exact frame index
      - "all" -> all frames
    """
    if frames is None:
        yield 0, read(str(xyz_path), format="extxyz", index=0)
        return

    if isinstance(frames, int):
        yield frames, read(str(xyz_path), format="extxyz", index=frames)
        return

    if isinstance(frames, str) and frames.lower() == "all":
        objs = read(str(xyz_path), format="extxyz", index=":")
        # ASE may return a single Atoms or a list; normalize to list
        if isinstance(objs, Atoms):
            yield 0, objs
        else:
            for i, at in enumerate(objs):
                yield i, at
        return

    raise ValueError("frames must be None, an integer, or the string 'all'.")


def main():
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
    p.add_argument(
        "--prefix",
        default=None,
        help="Optional prefix for output file names. If omitted, the xyz stem is used.",
    )
    args = p.parse_args()

    header = Path(args.header)
    xyz_path = Path(args.xyz)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not header.exists():
        raise FileNotFoundError(f"Header not found: {header}")
    if not xyz_path.exists():
        raise FileNotFoundError(f"XYZ not found: {xyz_path}")

    # Normalize frames argument
    frames_arg: int | str | None
    if args.frames is None:
        frames_arg = None
    else:
        s = str(args.frames).strip().lower()
        frames_arg = "all" if s == "all" else int(s)

    generated = 0
    for i, atoms in _iter_frames(xyz_path, frames_arg):
        # Decide base filename
        base = args.prefix if args.prefix else xyz_path.stem
        if frames_arg: base += f"_{i:05d}"
        infile = outdir / f"{base}.in"
        outfile = outdir / f"{base}.out"

        qe_text = build_pwi_from_header(header, atoms)
        infile.write_text(qe_text, encoding="utf-8")
        generated += 1

    # Friendly summary
    if generated == 1:
        print(f"Generated 1 QE input in {outdir.resolve()}")
    else:
        print(f"Generated {generated} QE inputs in {outdir.resolve()}")


if __name__ == "__main__":
    main()

