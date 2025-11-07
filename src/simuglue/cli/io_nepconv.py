from __future__ import annotations
import argparse

from simuglue.io.nepconv import convert_stream


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl io nepconv",
        description=(
            "Convert between NEP-style and ASE/extxyz-style XYZ headers.\n\n"
            "Examples:\n"
            "  sgl io nepconv in.xyz --to ase -o out.xyz\n"
            "  sgl io nepconv in.xyz --to nep -o -\n"
            "  sgl io nepconv - --to ase -o -  < in.xyz > out.xyz\n"
        ),
    )
    p.add_argument(
        "input",
        help="Input XYZ file, or '-' for stdin.",
    )
    p.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file path, or '-' for stdout (default: '-').",
    )
    p.add_argument(
        "--to",
        choices=["ase", "nep"],
        required=True,
        help="Target style: 'ase' or 'nep'.",
    )
    return p


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    try:
        convert_stream(
            input_path=args.input,
            output_path=args.output,
            to=args.to,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

