#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from simuglue.ase_patches.lammpsdata import (
    read_lammps_data,
    write_lammps_data,
    get_lmp_type_table,
)

from simuglue.topology import (
    build_topology_from_atoms,
    ensure_atom_tags_from_lmp_type_table,
    type_topology_inplace,
    attach_topology_arrays_to_atoms,
)
from simuglue.topology.typing import TypingOptions
from simuglue.topology.export import ExportTypesTopo

from simuglue.cli._parser import _parse_triplet_bools, _parse_float_list



def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl build lmp-topo",
        description=(
            "Infer bonds/angles/dihedrals from an input structure (LAMMPS data file) "
            "and write a new LAMMPS data file that includes the inferred topology. "
            "Bond inference uses distance windows centered at --rc with half-width --drc."
        ),
    )

    p.add_argument("input", help="Input LAMMPS data file.")

    p.add_argument("-atom_style", "--atom_style", default="full", help="Lammps atom style.")
    p.add_argument("-units", "--units", default="metal", help="Lammps units.")

    p.add_argument(
        "-periodic",
        "--periodic",
        default=None,
        help="Comma-separated periodicity directions (e.g. '1,1,1'). Default: keep ASE atoms.pbc.",
    )

    p.add_argument(
        "-rc",
        "--rc",
        default="",
        help="Comma-separated list of reference bond lengths (Angstrom). Required if --bond=1.",
    )
    p.add_argument(
        "-drc",
        "--drc",
        type=float,
        default=0.01,
        help="Tolerance around each rc: bond if r in (rc-drc, rc+drc).",
    )

    p.add_argument("-bond", "--bond", type=int, default=0, help="Calculate bonds (0/1).")
    p.add_argument("-angle", "--angle", type=int, default=0, help="Calculate angles (0/1).")
    p.add_argument("-dihed", "--dihed", type=int, default=0, help="Calculate dihedrals (0/1).")

    p.add_argument("-diff_bond_len", "--diff_bond_len", type=int, default=0, help="Differentiate bond types based on length (0/1).")
    p.add_argument("-diff_bond_fmt", "--diff_bond_fmt", default="%.2f", help="Format string for bond lengths (printf-style).")

    p.add_argument("-angle_symmetry", "--angle_symmetry", type=int, default=0, help="Differentiate angle symmetry (0/1).")
    p.add_argument("-diff_angle_theta", "--diff_angle_theta", type=int, default=0, help="Differentiate angle types based on theta (0/1).")
    p.add_argument("-diff_angle_theta_fmt", "--diff_angle_theta_fmt", default="%.2f", help="Format string for angle (deg).")

    p.add_argument("-cis_trans", "--cis_trans", type=int, default=0, help="Differentiate cis/trans dihedrals (0/1).")
    p.add_argument("-diff_dihed_theta", "--diff_dihed_theta", type=int, default=0, help="Differentiate dihedral types based on phi (0/1).")
    p.add_argument("-diff_dihed_theta_abs", "--diff_dihed_theta_abs", type=int, default=1, help="Use |phi| for dihedral typing (0/1).")
    p.add_argument("-diff_dihed_theta_fmt", "--diff_dihed_theta_fmt", default="%.2f", help="Format string for dihedral (deg).")

    p.add_argument(
        "-type_delimeter",
        "--type_delimeter",
        default=" ",
        help="String used to join the type tag parts.",
    )

    p.add_argument("-file_pos", "--file_pos", default="out.data", help="Output LAMMPS data file name.")
    p.add_argument("-file_types", "--file_types", default="", help="Optional: write type tables to this file.")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files.")

    return p


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        parser.exit(1, f"Input file not found: {in_path}\n")

    if args.drc < 0:
        parser.exit(1, "Error: --drc must be >= 0\n")

    calc_bonds = bool(args.bond)
    calc_angles = bool(args.angle)
    calc_diheds = bool(args.dihed)

    atoms = read_lammps_data(str(in_path), atom_style=args.atom_style, units=args.units)

    if args.periodic is not None:
        try:
            atoms.pbc = _parse_triplet_bools(args.periodic)
        except ValueError as exc:
            parser.exit(1, f"Invalid --periodic: {exc}\n")

    rc_list = _parse_float_list(args.rc)

    # Topology inference
    if calc_bonds:
        if not rc_list:
            parser.exit(1, "Error: --rc is required when --bond=1\n")

        topo, neighbors = build_topology_from_atoms(
            atoms,
            calc_bonds=True,
            calc_angles=calc_angles,
            calc_diheds=calc_diheds,
            rc_list=rc_list,
            drc=float(args.drc),
            deduplicate=True,
        )

        if calc_angles and not topo.bonds:
            parser.exit(1, "Error: no bonds inferred; cannot build angles (set --bond=1 and valid --rc)\n")
        if calc_diheds and not topo.bonds:
            parser.exit(1, "Error: no bonds inferred; cannot build dihedrals (set --bond=1 and valid --rc)\n")

        # Typing
        opts = TypingOptions(
            diff_bond_len=bool(args.diff_bond_len),
            diff_bond_fmt=str(args.diff_bond_fmt),
            angle_symmetry=bool(args.angle_symmetry),
            diff_angle_theta=bool(args.diff_angle_theta),
            diff_angle_theta_fmt=str(args.diff_angle_theta_fmt),
            cis_trans=bool(args.cis_trans),
            diff_dihed_theta=bool(args.diff_dihed_theta),
            diff_dihed_theta_abs=bool(args.diff_dihed_theta_abs),
            diff_dihed_theta_fmt=str(args.diff_dihed_theta_fmt),
            type_delimeter=str(args.type_delimeter),
        )

        lmp_type_table = get_lmp_type_table(atoms)
        if lmp_type_table is None:
            parser.exit(1, "Error: could not parse lmp_type_table\n")

        try:
            atom_tag = ensure_atom_tags_from_lmp_type_table(lmp_type_table)
        except Exception as exc:
            parser.exit(1, f"Error: {exc}\n")

        type_topology_inplace(
            atoms,
            topo,
            atom_tag=atom_tag,
            opts=opts,
            calc_bonds=True,
            calc_angles=calc_angles,
            calc_diheds=calc_diheds,
        )

        attach_topology_arrays_to_atoms(atoms, topo)

        # Optional: export type tables
        if args.file_types:
            types_path = Path(args.file_types)
            if types_path.exists() and not args.overwrite:
                parser.exit(1, f"Refusing to overwrite existing file: {types_path}\n")
            ExportTypesTopo(str(types_path), topo, lmp_type_table)

    else:
        # Preserve old behavior: if not building bonds, we also don't attach topology arrays.
        # We still write the output file below.
        if (calc_angles or calc_diheds):
            parser.exit(1, "Error: no bonds inferred; cannot build angles/dihedrals (set --bond=1 and valid --rc)\n")

    # Outputs
    out_path = Path(args.file_pos)
    if out_path.exists() and not args.overwrite:
        parser.exit(1, f"Refusing to overwrite existing file: {out_path}\n")

    write_lammps_data(
        str(out_path),
        atoms,
        atom_style="full",          # keep existing behavior (hard-coded)
        masses=True,
        force_skew=True,
        preserve_atom_types=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

