from __future__ import annotations
import argparse

from simuglue.mechanics.kinematics import defgrad_from_strain
from simuglue.io.matrix_3x3 import parse_3x3, ensure_symmetric, format_3x3
from simuglue.mechanics.voigt import parse_voigt6

from pathlib import Path
from typing import Literal
import sys
import numpy as np

StrainKind = Literal["auto", "full", "voigt"]

def parse_strain(text: str, kind: StrainKind = "auto") -> np.ndarray:
    """
    Parse a 3x3 strain tensor E from plain text.

    kind:
      - 'full'  : 3x3 (9 numbers, ';' optional between rows), must be symmetric
      - 'voigt' : exx eyy ezz gyz gxz gxy
      - 'auto'  : 9 -> full, 6 -> voigt, else error
    """
    s = text.strip()
    if not s:
        raise ValueError("Empty strain input.")

    if kind == "full":
        strain = parse_3x3(s)
        ensure_symmetric(strain)
        return strain

    if kind == "voigt":
        return parse_voigt6(s)

    # auto-detect
    vals = s.replace(";", " ").split()
    if len(vals) == 9:
        strain = parse_3x3(s)
        ensure_symmetric(strain)
        return strain
    if len(vals) == 6:
        return parse_voigt6(s)

    raise ValueError(
        "Could not infer strain format in auto mode: expected 9 (3x3) or 6 (Voigt) numbers."
    )


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl mech defgrad",
        description=(
            "Convert a symmetric strain tensor E to a deformation gradient F.\n\n"
            "Input sources:\n"
            "  • --E \"...\"       direct tensor specification\n"
            "  • --input <file>   path or '-' for stdin\n\n"
            "Accepted formats for --E / --input:\n"
            "  --kind full  : 3x3 matrix, e.g. 'exx exy exz; eyx eyy eyz; ...'\n"
            "  --kind voigt : exx eyy ezz gyz gxz gxy (gij = 2eij)\n"
            "  --kind auto  : detect 9 → full, 6 → voigt\n\n"
            "Measures:\n"
            "  engineering     : F = I + E (small-strain, rotation-free approximation)\n"
            "  green-lagrange  : E = 0.5(F^T F - I), inverted as pure stretch (R = I)\n"
            "  hencky          : E = log U, inverted as F = U (pure stretch)\n\n"
            "Note: This helper assumes no rotation. If you need full F with rotation,\n"
            "      specify F directly in your deformation tool."
        ),
    )

    p.add_argument(
        "-i",
        "--input",
        default="-",
        help="Input strain source: path or '-' for stdin (ignored if --E is given).",
    )

    p.add_argument(
        "--E",
        help=(
            "Direct strain tensor input: either 9 numbers (full symmetric 3x3) "
            "or 6 Voigt components. Overrides --input."
        ),
    )

    p.add_argument(
        "--kind",
        choices=["auto", "full", "voigt"],
        default="auto",
        help="Interpretation of input strain.",
    )

    p.add_argument(
        "--measure",
        choices=["engineering", "green-lagrange", "hencky"],
        default="engineering",
        help="Strain measure used to compute F.",
    )

    p.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output target for F: path or '-' for stdout.",
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

    # read
    if args.E is not None:
        text = args.E
    else:
        if args.input == "-":
            text = sys.stdin.read()
        else:
            text = Path(args.input).read_text(encoding="utf-8")

    # parse strain
    E = parse_strain(text, kind=args.kind)

    # compute F using the kinematics core (which also symmetrizes defensively)
    F = defgrad_from_strain(E, measure=args.measure)

    # format
    out = format_3x3(F, precision=args.precision)
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

