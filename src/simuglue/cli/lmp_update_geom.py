from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ase.io import read

from simuglue.io.lammps_topology import write_lammps_data_with_topology


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Update a LAMMPS data file with geometry (cell and coordinates) "
            "taken from an ASE-readable structure file, while preserving "
            "the original topology (bonds, angles, dihedrals, ...)."
        ),
    )
    p.add_argument(
        "--xyz",
        required=True,
        metavar="STRUCTURE",
        help="Input structure file (XYZ/extxyz, CIF, etc.) readable by ASE.",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to read from STRUCTURE (default: 0).",
    )
    p.add_argument(
        "--data",
        required=True,
        metavar="TOPOLOGY",
        help="Existing LAMMPS data file providing topology (bonds/angles/...).",
    )
    p.add_argument(
        "-o",
        "--output",
        metavar="OUT",
        help=(
            "Output LAMMPS data file. If omitted, the input TOPOLOGY file "
            "is overwritten in-place (requires --overwrite)."
        ),
    )
    p.add_argument(
        "--atom-style",
        dest="atom_style",
        help=(
            "LAMMPS atom_style (e.g. 'atomic', 'full', 'molecular'). "
            "If omitted, SimuGlue tries to infer it from the 'Atoms' header."
        ),
    )
    p.add_argument(
        "--units",
        default="metal",
        help="LAMMPS units keyword used when writing the geometry (default: metal).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    return p


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    atoms = read(args.xyz, index=args.frame)

    topo_path = Path(args.data)
    if args.output:
        out_path = Path(args.output)
    else:
        # Overwrite topology file in-place; require explicit --overwrite
        out_path = topo_path
        if not args.overwrite:
            parser.error(
                "Refusing to overwrite topology file without --overwrite; "
                "either pass -o/--output or use --overwrite."
            )

    write_lammps_data_with_topology(
        atoms=atoms,
        topology_data=topo_path,
        output_data=out_path,
        units=args.units,
        atom_style=args.atom_style,
        overwrite=args.overwrite,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

