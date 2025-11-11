#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Sequence

from simuglue.quantum_espresso.pwi_update import update_qe_input
from simuglue.cli._xyz_io import _iter_frames


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Generate Quantum ESPRESSO input file(s) by combining an existing .pwi "
            "with CELL_PARAMETERS and ATOMIC_POSITIONS from an extxyz file."
        ),
    )
    p.add_argument("--pwi", required=True, help="Template QE input (.pwi) file.")
    p.add_argument("--xyz", required=True, help="Input extxyz trajectory or structure.")
    p.add_argument("--prefix", help="Override QE prefix in &control / input.")
    p.add_argument("--outdir", help="Override QE outdir in &control / input.")
    p.add_argument(
        "--frames",
        default=None,
        help="Frame index (int) or 'all'. Default: first frame only.",
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=".",
        help="Directory for output .in files (default: current directory).",
    )
    p.add_argument(
        "-o",
        "--output",
        help=(
            "Base name for output file(s). "
            "If multiple frames: files are suffixed with _00000.in, etc. "
            "If single frame: uses BASE.in."
        ),
    )
    return p


def _parse_frames_arg(raw: str | None):
    if raw is None:
        return None  # first frame only
    s = str(raw).strip().lower()
    if s == "all":
        return "all"
    return int(s)


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    pwi_path = Path(args.pwi)
    xyz_path = Path(args.xyz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pwi_path.is_file():
        parser.error(f"PWI file not found: {pwi_path}")
    if not xyz_path.is_file():
        parser.error(f"XYZ file not found: {xyz_path}")

    pwi_text = pwi_path.read_text(encoding="utf-8")
    frames_arg = _parse_frames_arg(args.frames)

    # Determine base name (without extension)
    if args.output:
        base = Path(args.output).stem
    else:
        base = xyz_path.stem

    generated = 0

    for i, atoms in _iter_frames(xyz_path, frames_arg):
        # Decide output filename based on frames selection
        if frames_arg == "all":
            fname = f"{base}_{i:05d}.in"
        elif isinstance(frames_arg, int):
            # Specific frame only
            fname = f"{base}_{i:05d}.in"
        else:
            # Default: first frame only
            fname = f"{base}.in"

        outpath = output_dir / fname

        qe_text = update_qe_input(
            pwi_text,
            cell=atoms.get_cell().array,
            positions=atoms.get_positions(),
            symbols=atoms.get_chemical_symbols(),
            prefix=args.prefix,
            outdir=args.outdir,
        )

        outpath.write_text(qe_text, encoding="utf-8")
        generated += 1

    # Summary
    if generated == 0:
        parser.error("No frames were written (check --frames argument).")
    elif generated == 1:
        print(f"Generated 1 QE input in {output_dir.resolve()}")
    else:
        print(f"Generated {generated} QE inputs in {output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

