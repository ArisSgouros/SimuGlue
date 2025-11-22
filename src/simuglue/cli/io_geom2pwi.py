#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

from simuglue.quantum_espresso.pwi_update import update_qe_input
from simuglue.io.aseconv import (
    _infer_format_from_suffix,
    _read_atoms,
    SUPPORTED_INPUTS,
)


def _parse_iopts(arg: str | None) -> Dict[str, Dict[str, Any]]:
    """
    Parse --iopts string of the form:

        "fmt.key=val,otherfmt.otherkey=val2,..."

    into:

        {
            "fmt": {"key": val},
            "otherfmt": {"otherkey": val2},
        }

    Values:
        - 'true'/'false' (case-insensitive) -> bool
        - int if possible
        - float if possible
        - otherwise left as string
    """
    opts: Dict[str, Dict[str, Any]] = {}
    if not arg:
        return opts

    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue

        if "=" not in item:
            raise ValueError(f"Invalid --iopts entry (missing '='): {item!r}")

        lhs, val_str = item.split("=", 1)
        lhs = lhs.strip()
        val_str = val_str.strip()

        if "." not in lhs:
            raise ValueError(
                f"Invalid --iopts entry (expected 'fmt.key=val'): {item!r}"
            )

        fmt, key = lhs.split(".", 1)
        fmt = fmt.strip()
        key = key.strip()
        if not fmt or not key:
            raise ValueError(
                f"Invalid --iopts entry (empty fmt/key): {item!r}"
            )

        # Parse value
        low = val_str.lower()
        if low in {"true", "false"}:
            v: Any = (low == "true")
        else:
            try:
                v = int(val_str)
            except ValueError:
                try:
                    v = float(val_str)
                except ValueError:
                    v = val_str  # keep as string

        opts.setdefault(fmt, {})[key] = v

    return opts


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Generate Quantum ESPRESSO input file(s) by combining an existing .pwi "
            "with CELL_PARAMETERS and ATOMIC_POSITIONS from an arbitrary structure "
            "file readable via ASE (extxyz, lammps-data, traj, etc.)."
        ),
    )
    p.add_argument(
        "--pwi",
        required=True,
        help="Template QE input (.pwi/.in) file.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input structure file (extxyz, lammps-data, traj, ...).",
    )
    p.add_argument(
        "--iformat",
        default="auto",
        choices=["auto", *sorted(SUPPORTED_INPUTS)],
        help=(
            "Input structure format. 'auto' infers from extension "
            "(for non-stdin inputs)."
        ),
    )
    p.add_argument(
        "--iopts",
        metavar="OPTS",
        help=(
            "Format-specific READ options in 'fmt.key=val,...' form. "
            "Example for LAMMPS data: "
            "'lammps-data.style=full,lammps-data.units=metal'."
        ),
    )
    p.add_argument(
        "--prefix",
        help="Override QE prefix in &control / input.",
    )
    p.add_argument(
        "--outdir",
        help="Override QE outdir in &control / input.",
    )
    p.add_argument(
        "--frames",
        default=None,
        help=(
            "Frame selection in a simple form: integer index (e.g. '0') "
            "for a single frame, or 'all' for all frames. "
            "Default: first frame only."
        ),
    )
    p.add_argument(
        "--output-dir",
        dest="output_dir",
        default=".",
        help=(
            "Directory for output .in files (default: current directory). "
            "Ignored when writing to stdout."
        ),
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


def _frames_to_index_arg(raw: str | None) -> str:
    """
    Map our simple --frames semantics into the index argument for _read_atoms:

        None      -> "0"   (first frame only)
        "all"     -> ":"   (all frames)
        "<other>" -> "<other>" (passed directly, you can later extend to ASE syntax)
    """
    if raw is None:
        return "0"
    s = raw.strip().lower()
    if s == "all":
        return ":"
    return raw


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    pwi_path = Path(args.pwi)
    if not pwi_path.is_file():
        parser.error(f"PWI file not found: {pwi_path}")

    input_path = Path(args.input)
    if not input_path.is_file():
        parser.error(f"Input structure file not found: {input_path}")

    # Resolve input format
    iformat = args.iformat
    if iformat == "auto":
        inferred = _infer_format_from_suffix(input_path)
        if inferred is None:
            parser.error(
                "Could not infer input format from file extension; "
                "please specify --iformat."
            )
        iformat = inferred

    if iformat not in SUPPORTED_INPUTS:
        parser.error(f"Unsupported input format for structure file: {iformat}")

    # Parse format-specific read options
    try:
        read_opts = _parse_iopts(args.iopts)
    except ValueError as e:
        parser.error(str(e))

    pwi_text = pwi_path.read_text(encoding="utf-8")
    index_arg = _frames_to_index_arg(args.frames)

    # Read atoms using shared aseconv machinery
    atoms_list = _read_atoms(
        src=str(input_path),
        fmt=iformat,
        frames=index_arg,
        options=read_opts,
    )
    nframes = len(atoms_list)

    if nframes == 0:
        parser.error("No frames were read from input structure file.")

    # Decide stdout vs files
    stdout_mode = False
    if nframes == 1:
        # Single-frame mode
        if args.output is None or args.output == "-":
            stdout_mode = True
    else:
        # Multi-frame mode
        if args.output == "-":
            parser.error(
                "Cannot use '-' (stdout) with multiple frames; "
                "please provide a file base name instead."
            )

    # Prepare output directory if we are writing files
    if not stdout_mode:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = None  # not used

    # Determine base name for file outputs
    if not stdout_mode:
        if args.output:
            base = Path(args.output).stem
        else:
            base = input_path.stem
    else:
        base = None  # not used

    generated = 0

    for i, atoms in enumerate(atoms_list):
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
            # File naming: if multiple frames, suffix with _00000; else simple .in
            if nframes > 1:
                fname = f"{base}_{i:05d}.in"
            else:
                fname = f"{base}.in"

            outpath = output_dir / fname
            outpath.write_text(qe_text, encoding="utf-8")

        generated += 1

    if generated == 0:
        parser.error("No frames were written (check --frames argument).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

