from __future__ import annotations
import argparse

from simuglue.io._defgradconv import convert_strain_to_F_stream


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl mech defgrad",
        description=(
            "Convert a symmetric strain tensor E to a deformation gradient F.\n\n"
            "Input (from -i/--input or stdin) is plain text:\n"
            "  --kind full : 3x3 as 'exx exy exz; eyx eyy eyz; ezx ezy ezz'\n"
            "  --kind voigt: 'exx eyy ezz gyz gxz gxy' (no 2x on shear)\n"
            "  --kind auto : guess 9 -> full, 6 -> voigt\n\n"
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
        help="Input strain source: path or '-' for stdin (default: '-').",
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
        help="Output target for F: path or '-' for stdout (default: '-').",
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

    try:
        convert_strain_to_F_stream(
            input_source=args.input,
            input_kind=args.kind,
            measure=args.measure,
            precision=args.precision,
            output_target=args.output,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

