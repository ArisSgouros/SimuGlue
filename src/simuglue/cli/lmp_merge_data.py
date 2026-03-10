from __future__ import annotations

import argparse

from simuglue.lmp.mergedata import merge_datafiles


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl lammps mergedata",
        description=(
            "Merge two LAMMPS data files.\n\n"
            "Assumptions:\n"
            "  - orthogonal boxes\n"
            "  - Atoms section in atom_style full format:\n"
            "    id mol type q x y z [ix iy iz]\n\n"
            "Examples:\n"
            "  sgl lammps mergedata a.data b.data merged.data\n"
            "  sgl lammps mergedata a.data b.data merged.data --side=+z\n"
            "  sgl lammps mergedata slab.data mol.data merged.data "
            "--side=0 --pos-offset 0.0 0.0 15.0 --atom-type-offset 4\n"
            "  sgl lammps mergedata a.data b.data merged.data --side=-x\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    p.add_argument("data_file_a", help="First LAMMPS data file.")
    p.add_argument("data_file_b", help="Second LAMMPS data file.")
    p.add_argument("output", help="Merged output data file.")

    p.add_argument(
        "--side",
        default="0",
        choices=["0", "-x", "+x", "-y", "+y", "-z", "+z"],
        help=(
            "Attach file B relative to file A along a box face. "
            "For negative values, use the equals form, e.g. --side=-x"
        ),
    )

    p.add_argument(
        "--mol-offset",
        type=int,
        default=0,
        help=(
            "Additional molecule-id offset applied to file B. "
            "The effective offset is max(--mol-offset, max mol-id in file A)."
        ),
    )
    p.add_argument(
        "--atom-type-offset", "--poffset",
        dest="atom_type_offset",
        type=int,
        default=0,
        help="Atom type offset applied to file B.",
    )
    p.add_argument(
        "--bond-type-offset", "--boffset",
        dest="bond_type_offset",
        type=int,
        default=0,
        help="Bond type offset applied to file B.",
    )
    p.add_argument(
        "--angle-type-offset", "--aoffset",
        dest="angle_type_offset",
        type=int,
        default=0,
        help="Angle type offset applied to file B.",
    )
    p.add_argument(
        "--dihedral-type-offset", "--doffset",
        dest="dihedral_type_offset",
        type=int,
        default=0,
        help="Dihedral type offset applied to file B.",
    )
    p.add_argument(
        "--improper-type-offset", "--ioffset",
        dest="improper_type_offset",
        type=int,
        default=0,
        help="Improper type offset applied to file B.",
    )

    p.add_argument(
        "--pos-offset", "--posoffset",
        dest="pos_offset",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="Extra Cartesian shift applied to file B after the side-based shift.",
    )

    p.add_argument(
        "--keep-box-origin",
        action="store_true",
        help="Do not shift the merged box so that lo = 0 0 0.",
    )
    p.add_argument(
        "--no-dump",
        action="store_true",
        help="Do not also write output.lammpstrj.",
    )

    return p


def main(argv: list[str] | None = None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    try:
        merge_datafiles(
            data_file_a=args.data_file_a,
            data_file_b=args.data_file_b,
            output_path=args.output,
            side=args.side,
            mol_offset=args.mol_offset,
            atom_type_offset=args.atom_type_offset,
            bond_type_offset=args.bond_type_offset,
            angle_type_offset=args.angle_type_offset,
            dihedral_type_offset=args.dihedral_type_offset,
            improper_type_offset=args.improper_type_offset,
            pos_offset=tuple(args.pos_offset),
            set_lo_at_zero=not args.keep_box_origin,
            write_dump=not args.no_dump,
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
