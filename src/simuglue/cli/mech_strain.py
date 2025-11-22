from __future__ import annotations
from pathlib import Path
from typing import Literal
import argparse
import sys
import numpy as np

from simuglue.mechanics.kinematics import strain_from_F
from simuglue.io.matrix_3x3 import parse_3x3, format_3x3, ensure_symmetric


def _format_strain_voigt(E: np.ndarray, precision: int) -> str:
    """
    Format symmetric E as Voigt:
      'exx eyy ezz gyz gxz gxy'
    using:
      exx = E[0,0], eyy = E[1,1], ezz = E[2,2],
      gyz = 2 E[1,2], gxz = 2 E[0,2], gxy = 2 E[0,1]
    """
    E = np.array(E, float)
    E = 0.5 * (E + E.T)
    E = np.round(E, precision)

    exx = E[0, 0]
    eyy = E[1, 1]
    ezz = E[2, 2]
    gyz = 2.0 * E[1, 2]
    gxz = 2.0 * E[0, 2]
    gxy = 2.0 * E[0, 1]

    vals = [exx, eyy, ezz, gyz, gxz, gxy]
    return " ".join(f"{x:.{precision}g}" for x in vals)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl mech strain",
        description=(
            "Convert deformation gradient F to strain tensor E.\n\n"
            "Input sources:\n"
            "  • --F \"...\"       direct tensor specification\n"
            "  • --input <file>   path or '-' for stdin\n\n"
            "  accepted formats:"
            "  'F11 F12 F13; F21 F22 F23; F31 F32 F33'\n"
            "  or 9 space-separated numbers.\n\n"
            "Measures:\n"
            "  engineering     : eps = 0.5 * ((F - I) + (F - I)^T)\n"
            "  green-lagrange  : E   = 0.5 * (F^T F - I)\n"
            "  hencky          : E   = log U, U = sqrtm(F^T F)\n\n"
            "Output:\n"
            "  --out-kind full : 'exx exy exz; eyx eyy eyz; ezx ezy ezz'\n"
            "  --out-kind voigt: 'exx eyy ezz gyz gxz gxy'\n"
        ),
    )

    p.add_argument(
        "-i",
        "--input",
        default="-",
        help="Input deformation gradient: path or '-' for stdin (default: '-').",
    )

    p.add_argument(
        "--F",
        help=(
            "Direct deformation gradient tensor input: 'F11 F12 F13; F21 F22 F23; F31 F32 F33'"
            "Overrides --input."
        ),
    )

    p.add_argument(
        "--measure",
        choices=["engineering", "green-lagrange", "hencky"],
        default="engineering",
        help="Strain measure used to compute E from F.",
    )

    p.add_argument(
        "--out-kind",
        choices=["full", "voigt"],
        default="full",
        help="Output format for E (default: full 3x3).",
    )

    p.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output target: path or '-' for stdout (default: '-').",
    )

    p.add_argument(
        "--precision",
        type=int,
        default=12,
        help="Decimal digits in the output.",
    )

    return p


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    """
    Read F, compute strain E via chosen measure, and write E.

    - input_source: path or '-' for stdin
    - measure: 'engineering' | 'green-lagrange' | 'hencky'
    - out_kind: 'full' -> 3x3 text, 'voigt' -> Voigt 6
    - precision: decimals
    - output_target: path or '-' for stdout
    """
    # read
    if args.F is not None:
        text = args.F
    else:
        if args.input == "-":
            text = sys.stdin.read()
        else:
            text = Path(args.input).read_text(encoding="utf-8")

    # parse F
    F = parse_3x3(text)

    # compute strain
    E = strain_from_F(F, measure=args.measure)

    # symmetry check
    ensure_symmetric(E)

    # format
    if args.out_kind == "full":
        out = format_3x3(E, precision=args.precision)
    else:
        out = _format_strain_voigt(E, args.precision)

    if not out.endswith("\n"):
        out += "\n"

    # write
    if args.output == "-":
        sys.stdout.write(out)
    else:
        Path(args.output).write_text(out, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

