# simuglue/cli/io_aseconv.py

from __future__ import annotations
import argparse
from typing import Dict, Any
from simuglue.io.aseconv import convert


def _parse_kv_opts(arg: str) -> Dict[str, Any]:
    """
    Parse: "key=val,key2=val2,..." -> {"key": val, "key2": val2}

    Value parsing:
      - true/false -> bool
      - int if possible
      - float if possible
      - else string
    """
    opts: Dict[str, Any] = {}
    if not arg:
        return opts

    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid opts entry (missing '='): {item!r}")

        key, val_str = item.split("=", 1)
        key = key.strip()
        val_str = val_str.strip()
        if not key:
            raise ValueError(f"Invalid opts entry (empty key): {item!r}")

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
                    v = val_str

        opts[key] = v

    return opts


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Convert between ASE-supported structure formats.",
    )
    p.add_argument("-i", "--input", required=True, help="Input file or '-' for stdin.")
    p.add_argument("-o", "--output", default="-", help="Output file or '-' for stdout.")
    p.add_argument(
        "--iformat",
        default="auto",
        choices=["auto", "xyz", "extxyz", "traj", "espresso-in", "espresso-out", "lammps-data", "lammps-dump-text"],
        help="Input format (default: infer from extension when not stdin).",
    )
    p.add_argument(
        "--oformat",
        choices=["xyz", "extxyz", "traj", "lammps-data"],
        help="Output format (default: infer from extension when not stdout).",
    )
    p.add_argument("--frames", default=None, help="Frame selection (ASE index syntax, e.g. ':', '0', '0:10:2').")

    p.add_argument(
        "--iopts",
        metavar="OPTS",
        help="READ options as 'key=val,...'. Example: 'style=full,units=metal'.",
    )
    p.add_argument(
        "--oopts",
        metavar="OPTS",
        help="WRITE options as 'key=val,...'. Example: 'force_skew=true,style=full'.",
    )

    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output file.")
    return p


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    read_opts: Dict[str, Any] = {}
    write_opts: Dict[str, Any] = {}

    if args.iopts:
        try:
            read_opts = _parse_kv_opts(args.iopts)
        except ValueError as e:
            parser.error(str(e))

    if args.oopts:
        try:
            write_opts = _parse_kv_opts(args.oopts)
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

