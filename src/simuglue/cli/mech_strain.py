from __future__ import annotations
import argparse

from simuglue.io._strainconv import convert_F_to_strain_stream


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl mech strain",
        description=(
            "Convert deformation gradient F to strain tensor E.\n\n"
            "Input (from -i/--input or stdin) must be F as:\n"
            "  'F11 F12 F13; F21 F22 F23; F31 F32 F33'\n"
            "or 9 space-separated numbers.\n\n"
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

    try:
        convert_F_to_strain_stream(
            input_source=args.input,
            measure=args.measure,
            out_kind=args.out_kind,
            precision=args.precision,
            output_target=args.output,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

