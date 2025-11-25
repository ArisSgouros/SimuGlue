#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

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
        help="Directory for output .in files (default: current directory). "
             "Ignored when writing to stdout.",
    )
    p.add_argument(
        "-o",
        "--output",
        help=(
            "Base name for output file(s), or '-' for stdout (single-frame only). "
            "If multiple frames: files are suffixed with _00000.in, etc. "
            "If single frame and no -o given: write to stdout."
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

    if not pwi_path.is_file():
        parser.error(f"PWI file not found: {pwi_path}")
    if not xyz_path.is_file():
        parser.error(f"XYZ file not found: {xyz_path}")

    pwi_text = pwi_path.read_text(encoding="utf-8")
    frames_arg = _parse_frames_arg(args.frames)

    # Decide stdout vs files
    stdout_mode = False
    if frames_arg is None:
        # Single-frame mode
        if args.output is None or args.output == "-":
            stdout_mode = True
    else:
        # Multi-frame mode
        if args.output == "-":
            parser.error("Cannot use '-' (stdout) with multiple frames; "
                         "please provide a file base name instead.")

    # Prepare output directory only if we are writing files
    if not stdout_mode:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None  # not used

    generated = 0

    # Determine base name only if writing to files
    if not stdout_mode:
        if args.output:
            base = Path(args.output).stem
        else:
            base = xyz_path.stem
    else:
        base = None  # not used

    for i, atoms in _iter_frames(xyz_path, frames_arg):
        qe_text = update_qe_input(
            pwi_text,
            cell=atoms.get_cell().array,
            positions=atoms.get_positions(),
            symbols=atoms.get_chemical_symbols(),
            prefix=args.prefix,
            outdir=args.outdir,
        )

        if stdout_mode:
            # Single frame only by construction
            sys.stdout.write(qe_text)
            if not qe_text.endswith("\n"):
                sys.stdout.write("\n")
        else:
            # File naming depends on frames selection
            if frames_arg == "all":
                fname = f"{base}_{i:05d}.in"
            elif isinstance(frames_arg, int):
                fname = f"{base}_{i:05d}.in"
            else:
                # Default single-frame-to-file (if stdout_mode=False by explicit -o)
                fname = f"{base}.in"

            outpath = output_dir / fname
            outpath.write_text(qe_text, encoding="utf-8")

        generated += 1

    if generated == 0:
        parser.error("No frames were written (check --frames argument).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

