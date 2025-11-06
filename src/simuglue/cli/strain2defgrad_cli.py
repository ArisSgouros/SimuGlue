#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from simuglue.mechanics.kinematics import defgrad_from_strain
import numpy as np

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Convert a strain tensor to a deformation gradient (F).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--E",
        dest="E_str",
        help="Strain tensor E (3×3) as 'exx exy exz; eyx eyy eyz; ezx ezy ezz' "
             "or 9 space-separated values.",
    )
    g.add_argument(
        "--E-voigt",
        dest="E_voigt_str",
        help="Strain tensor in Voigt 6 as 'exx eyy ezz gyz gxz gxy'. "
             "Assumes symmetric convention (no ×2 on shear).",
    )
    p.add_argument(
        "--measure",
        choices=("engineering", "green-lagrange", "hencky"),
        default="engineering",
        help="Strain measure used to compute F from E.",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Optional JSON output file. If omitted, only prints to screen.",
    )
    p.add_argument(
        "--print",
        dest="print_3x3",
        action="store_true",
        help="Also print F to stdout as 'a b c; d e f; g h i'.",
    )
    p.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Decimal digits in JSON (apply rounding before writing).",
    )
    return p

# ---------- helpers ----------
def _parse_3x3(s: str) -> np.ndarray:
    vals = [float(x) for x in s.replace(";", " ").split()]
    if len(vals) != 9:
        raise ValueError("3×3 input needs 9 numbers")
    M = np.array(vals, float).reshape(3, 3)
    return 0.5 * (M + M.T)  # enforce symmetry

def _parse_voigt6_to_symmetric_tri(s: str) -> np.ndarray:
    vals = [float(x) for x in s.replace(";", " ").split()]
    if len(vals) != 6:
        raise ValueError("Voigt input needs 6 numbers: exx eyy ezz gyz gxz gxy")
    exx, eyy, ezz, gyz, gxz, gxy = vals
    E = np.array([[exx,   gxy/2, gxz/2],
                  [gxy/2, eyy,   gyz/2],
                  [gxz/2, gyz/2, ezz]], float)
    return E



def _rounded_list2d(M: np.ndarray, digits: int) -> list[list[float]]:
    if digits is None:
        return M.tolist()
    return [[float(np.round(x, digits)) for x in row] for row in M]

def _rounded(M: np.ndarray, digits: int) -> np.ndarray:
    return np.round(M.astype(float), digits)

def _format_matrix_3x3(M: np.ndarray, digits: int) -> str:
    M = _rounded(M, digits)
    rows = [" ".join(f"{x:.{digits}g}" for x in row) for row in M]
    return "; ".join(rows)

# ---------- main ----------
def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    # Build E (3×3)
    try:
        if args.E_str is not None:
            E = _parse_3x3(args.E_str)
        else:
            E = _parse_voigt6_to_symmetric_tri(args.E_voigt_str)
    except Exception as exc:
        parser.error(str(exc))
        return 2  # not reached because parser.error exits

    # Compute F
    try:
        F = defgrad_from_strain(E, measure=args.measure)
    except Exception as exc:
        parser.error(f"Failed to compute F: {exc}")
        return 2

    # Write JSON
    if args.output is not None:
        payload = _rounded_list2d(F, args.precision)
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {args.output}")

    # Optional stdout print in 'a b c; d e f; g h i'
    if args.print_3x3:
        print(_format_matrix_3x3(F, args.precision))

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

