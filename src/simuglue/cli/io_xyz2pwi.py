#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

from ase import Atoms
from ase.io import read

from simuglue.quantum_espresso.pwi_update import update_qe_input


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl io xyz2pwi",
        description=(
            "Generate Quantum ESPRESSO input text by combining an existing .pwi "
            "with CELL_PARAMETERS and ATOMIC_POSITIONS from an extxyz file. "
            "If multiple frames are selected, all resulting inputs are concatenated "
            "into a single stream/file."
        ),
    )
    p.add_argument(
        "--pwi",
        required=True,
        help="Template QE input (.pwi/.in) file.",
    )
    p.add_argument(
        "--xyz",
        required=True,
        help="Input extxyz trajectory or structure.",
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
        help="Frame index (0-based int) or 'all'. Default: first frame only.",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output file path, or '-' for stdout. "
            "Default: stdout."
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return p


def _parse_frames_arg(raw: str | None) -> str | int | None:
    """
    Parse --frames:
      - None     -> use first frame only
      - 'all'    -> all frames
      - '<int>'  -> that frame index (0-based)
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "all":
        return "all"
    return int(s)


def _read_all_frames_extxyz(
    xyz_arg: str,
    parser: argparse.ArgumentParser,
) -> List[Atoms]:
    """
    Read *all* frames as extxyz from a file (no stdin here),
    and return them as a list of Atoms.
    """
    xyz_path = Path(xyz_arg)
    if not xyz_path.is_file():
        parser.error(f"XYZ file not found: {xyz_path}")

    try:
        images = read(xyz_path, format="extxyz", index=":")
    except Exception as exc:
        parser.error(f"Failed to read extxyz from {xyz_path}: {exc}")

    if isinstance(images, Atoms):
        frames: List[Atoms] = [images]
    else:
        frames = list(images)

    return frames


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

    idx = int(frames_sel)
    return [frames[idx]]


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    # 1) read pwi input
    pwi_path = Path(args.pwi)
    if not pwi_path.is_file():
        parser.error(f"PWI file not found: {pwi_path}")
    pwi_text = pwi_path.read_text(encoding="utf-8")

    # 2) read atoms from xyz (all frames)
    all_frames = _read_all_frames_extxyz(args.xyz, parser)

    # 3) apply selection
    frames_sel = _parse_frames_arg(args.frames)
    sel_frames = _select_frames_from_list(all_frames, frames_sel)
    if not sel_frames:
        parser.error("No frames were selected or read; check --frames argument.")

    # 4) generate new qe_text for each selected frame
    qe_texts: list[str] = []
    for atoms in sel_frames:
        qe_text = update_qe_input(
            pwi_text,
            cell=atoms.get_cell().array,
            positions=atoms.get_positions(),
            symbols=atoms.get_chemical_symbols(),
            prefix=args.prefix,
            outdir=args.outdir,
        )
        qe_texts.append(qe_text)

    # Concatenate if multiple frames.
    if len(qe_texts) == 1:
        # Single-frame: don't mess with whitespace, keep whatever update_qe_input returned.
        combined_qe = qe_texts[0]
    else:
        # Multi-frame: normalize trailing newlines and separate with a blank line.
        cleaned = [txt.rstrip("\n") for txt in qe_texts]
        combined_qe = "\n\n".join(cleaned) + "\n"

    # 5) write to file/stdout
    if args.output is None or args.output == "-":
        sys.stdout.write(combined_qe)
    else:
        out_path = Path(args.output)
        if out_path.exists() and not args.overwrite:
            parser.error(f"Refusing to overwrite existing file: {out_path}")
        out_path.write_text(combined_qe, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

