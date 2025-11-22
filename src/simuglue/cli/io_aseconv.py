# simuglue/cli/io_aseconv.py
from __future__ import annotations

import argparse
from typing import Dict, Any

from simuglue.io.aseconv import convert


def _parse_fmt_opts(arg: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse an options string of the form:

        "fmt.key=val,otherfmt.otherkey=val2,..."

    into:

        {
            "fmt": {"key": val},
            "otherfmt": {"otherkey": val2},
        }

    Values are parsed as:
        - 'true'/'false' (case-insensitive) -> bool
        - int if possible
        - float if possible
        - otherwise left as string.
    """
    opts: Dict[str, Dict[str, Any]] = {}
    if not arg:
        return opts

    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue

        if "=" not in item:
            raise ValueError(f"Invalid opts entry (missing '='): {item!r}")

        lhs, val_str = item.split("=", 1)
        lhs = lhs.strip()
        val_str = val_str.strip()

        if "." not in lhs:
            raise ValueError(
                f"Invalid opts entry (expected 'fmt.key=val'): {item!r}"
            )

        fmt, key = lhs.split(".", 1)
        fmt = fmt.strip()
        key = key.strip()
        if not fmt or not key:
            raise ValueError(
                f"Invalid opts entry (empty fmt/key): {item!r}"
            )

        # Parse value
        v: Any
        low = val_str.lower()
        if low in {"true", "false"}:
            v = (low == "true")
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
        description="Convert between ASE-supported structure formats.",
    )
    p.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input file or '-' for stdin.",
    )
    p.add_argument(
        "-o",
        "--output",
        default="-",
        help="Output file or '-' for stdout.",
    )
    p.add_argument(
        "--iformat",
        default="auto",
        choices=[
            "auto",
            "extxyz",
            "traj",
            "espresso-in",
            "espresso-out",
            "lammps-data",
            "lammps-dump-text",
        ],
        help="Input format (default: infer from extension when not stdin).",
    )
    p.add_argument(
        "--oformat",
        choices=[
            "extxyz",
            "traj",
            "lammps-data",
        ],
        help="Output format (default: infer from extension when not stdout).",
    )
    p.add_argument(
        "--frames",
        default=None,
        help="Frame selection (ASE index syntax, e.g. ':', '0', '0:10:2').",
    )

    # Generic per-format options for input (read)
    p.add_argument(
        "--iopts",
        metavar="OPTS",
        help=(
            "Format-specific READ options in 'fmt.key=val,...' form. "
            "Example: 'lammps-data.style=full,lammps-data.units=metal'."
        ),
    )

    # Generic per-format options for output (write)
    p.add_argument(
        "--oopts",
        metavar="OPTS",
        help=(
            "Format-specific WRITE options in 'fmt.key=val,...' form. "
            "Example: 'lammps-data.force_skew=true,lammps-data.style=full'."
        ),
    )

    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    return p


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    read_opts: dict[str, dict[str, object]] = {}
    write_opts: dict[str, dict[str, object]] = {}

    if args.iopts:
        try:
            read_opts = _parse_fmt_opts(args.iopts)
        except ValueError as e:
            parser.error(str(e))

    if args.oopts:
        try:
            write_opts = _parse_fmt_opts(args.oopts)
        except ValueError as e:
            parser.error(str(e))

    convert(
        input_path=args.input,
        output_path=args.output,
        iformat=args.iformat,
        oformat=args.oformat,
        frames=args.frames,
        read_opts=read_opts,
        write_opts=write_opts,
        overwrite=args.overwrite,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

