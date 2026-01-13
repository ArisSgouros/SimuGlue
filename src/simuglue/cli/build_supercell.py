#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List

from ase import Atoms
from ase.io import read, write

from simuglue.cli._parser import _parse_bool01

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

def _repl_ordered(atoms: Atoms, na: int, nb: int, nc: int, order: str = "abc") -> Atoms:
    if sorted(order) != ["a", "b", "c"]:
        raise ValueError(f"order must be a permutation of 'abc', got {order!r}")

    reps = {"a": na, "b": nb, "c": nc}
    axis = {"a": 0, "b": 1, "c": 2}

    out = atoms
    for ch in order:  # first = fastest varying
        r = [1, 1, 1]
        r[axis[ch]] = reps[ch]
        out = out.repeat(tuple(r))
    return out

import numpy as np

def _replica_indices_like_repl_ordered(
    n0: int, na: int, nb: int, nc: int, order: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-atom replica indices (ia, ib, ic) of length n0*na*nb*nc
    in the exact ordering produced by `_repl_ordered()` which does
    sequential Atoms.repeat() calls in `order` (first = fastest varying).
    """
    reps = {"a": int(na), "b": int(nb), "c": int(nc)}
    if sorted(order) != ["a", "b", "c"]:
        raise ValueError(f"order must be a permutation of 'abc', got {order!r}")

    ia = np.zeros(n0, dtype=np.int64)
    ib = np.zeros(n0, dtype=np.int64)
    ic = np.zeros(n0, dtype=np.int64)

    for ch in order:
        m = reps[ch]
        if m == 1:
            continue

        if ch == "a":
            ia = np.concatenate([ia + j for j in range(m)])
            ib = np.tile(ib, m)
            ic = np.tile(ic, m)
        elif ch == "b":
            ib = np.concatenate([ib + j for j in range(m)])
            ia = np.tile(ia, m)
            ic = np.tile(ic, m)
        else:  # "c"
            ic = np.concatenate([ic + j for j in range(m)])
            ia = np.tile(ia, m)
            ib = np.tile(ib, m)

    return ia, ib, ic

def _apply_increment_array_posthoc(
    out: Atoms,
    *,
    base_n: int,
    na: int,
    nb: int,
    nc: int,
    order: str,
    array_name: str,
    inc_x: bool,
    inc_y: bool,
    inc_z: bool,
) -> None:
    if array_name not in out.arrays:
        raise ValueError(f"Array {array_name!r} not found in atoms.arrays")

    arr = out.arrays[array_name]
    if arr.ndim != 1:
        raise ValueError(f"Array {array_name!r} must be 1D, got shape {arr.shape}")

    ia, ib, ic = _replica_indices_like_repl_ordered(base_n, na, nb, nc, order)
    offsets = (ia if inc_x else 0) + (ib if inc_y else 0) + (ic if inc_z else 0)

    if len(offsets) != len(out):
        raise RuntimeError(f"Internal mismatch: offsets={len(offsets)} vs atoms={len(out)}")

    # Require integer-like
    if np.issubdtype(arr.dtype, np.integer):
        out.arrays[array_name] = arr + offsets.astype(arr.dtype, copy=False)
    else:
        # allow float that is actually integer-like
        if np.allclose(arr, np.round(arr)):
            arr_i = np.round(arr).astype(np.int64)
            out.arrays[array_name] = (arr_i + offsets).astype(np.int64, copy=False)
        else:
            raise ValueError(f"Array {array_name!r} must be integer-like to increment (dtype={arr.dtype})")



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
        "--order",
        default="abc",
        choices=("abc", "acb", "bac", "bca", "cab", "cba"),
        help=(
            "Replication loop order for generating images. "
            "Uses a,b,c for the three cell vectors. Default: abc. "
            "Example: --order bac replicates b fastest, then a, then c."
        ),
    )

    p.add_argument(
        "--frames",
        default=None,
        help="Frame index (0-based int) or 'all'. "
             "Default: first frame only.",
    )

    p.add_argument(
        "--incr-array",
        nargs=4,
        metavar=("NAME", "X", "Y", "Z"),
        default=None,
        help=(
            "Increment a 1D integer-like per-atom array during replication. "
            "X/Y/Z are booleans (0/1/true/false) mapped to a/b/c respectively. "
            "Offset added is (ia if X) + (ib if Y) + (ic if Z), where (ia,ib,ic) "
            "are the replica indices. Example: --incr-array mol-id 0 0 1."
        ),
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

    order = args.order

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
    out_frames: List[Atoms] = []

    for atoms_ref in in_frames:
        out = _repl_ordered(atoms_ref, na, nb, nc, order=order)

        if args.incr_array is not None:
            name, sx, sy, sz = args.incr_array
            incx = _parse_bool01(sx)
            incy = _parse_bool01(sy)
            incz = _parse_bool01(sz)

            _apply_increment_array_posthoc(
                out,
                base_n=len(atoms_ref),
                na=na, nb=nb, nc=nc,
                order=order,
                array_name=name,
                inc_x=incx, inc_y=incy, inc_z=incz,
            )

        out_frames.append(out)
    # -------- 3) replicate atoms --------
    #out_frames: List[Atoms] = [_repl_ordered(atoms_ref, na, nb, nc, order=order) for atoms_ref in in_frames]


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

