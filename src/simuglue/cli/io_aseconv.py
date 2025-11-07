# simuglue/cli/io_aseconv.py
from __future__ import annotations
import argparse
from simuglue.io.aseconv import convert

def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Convert between ASE-supported structure formats."
    )
    p.add_argument("-i", "--input", required=True,
                   help="Input file or '-' for stdin.")
    p.add_argument("-o", "--output", required=True,
                   help="Output file or '-' for stdout.")
    p.add_argument("--iformat", default="auto",
                   choices=[
                       "auto",
                       "extxyz",
                       "traj",
                       "espresso-in",
                       "espresso-out",
                       "lammps-data",
                       "lammps-dump-text",
                   ])
    p.add_argument("--oformat",
                   choices=[
                       "extxyz",
                       "traj",
                       "espresso-in",
                       "espresso-out",
                       "lammps-data",
                       "lammps-dump-text",
                   ])
    p.add_argument("--frames", default=None,
                   help="Frame selection (ASE index syntax, e.g. ':', '0', '0:10:2').")

    # LAMMPS-specific knobs (extensible later)
    p.add_argument("--lammps-style", default=None,
                   choices=["full", "atomic", "charge"],
                   help="LAMMPS atom_style for lammps-data read/write.")
    # placeholder for future:
    # p.add_argument("--lammps-units", default=None, ...)

    p.add_argument("--overwrite", action="store_true",
                   help="Allow overwriting existing output file.")
    return p

def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    read_opts: dict[str, dict[str, object]] = {}
    write_opts: dict[str, dict[str, object]] = {}

    if args.lammps_style:
        for key in ("lammps-data",):
            read_opts.setdefault(key, {})["style"] = args.lammps_style
            write_opts.setdefault(key, {})["style"] = args.lammps_style

    # if you add --lammps-units later:
    # if args.lammps_units:
    #     for key in ("lammps-data", "lammps-dump-text"):
    #         read_opts.setdefault(key, {})["units"] = args.lammps_units
    #         write_opts.setdefault(key, {})["units"] = args.lammps_units

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
