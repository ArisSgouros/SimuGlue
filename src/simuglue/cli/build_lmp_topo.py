#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

from ase.io.lammpsdata import read_lammps_data

from simuglue.topology.infer import (
    infer_bonds_by_distance,
    infer_angles_from_adjacency,
    infer_dihedrals_from_bonds,
)
from simuglue.topology.topo import Topo
from simuglue.topology.typing import TypingOptions, type_angles, type_bonds, type_dihedrals
from simuglue.topology.export import ExportLammpsDataFileTopo, ExportTypesTopo


def _parse_triplet_bools(s: str) -> Tuple[bool, bool, bool]:
    parts = [p.strip() for p in s.split(',') if p.strip() != '']
    if len(parts) != 3:
        raise ValueError("Expected 3 comma-separated values like '1,1,1'")
    out = []
    for p in parts:
        if p.lower() in ("t", "true", "1", "y", "yes"):
            out.append(True)
        elif p.lower() in ("f", "false", "0", "n", "no"):
            out.append(False)
        else:
            raise ValueError(f"Invalid boolean: {p!r}")
    return (out[0], out[1], out[2])


def _parse_float_list(s: str) -> list[float]:
    if s is None:
        return []
    s = s.strip()
    if not s:
        return []
    return [float(x) for x in s.split(',') if x.strip()]


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl build lmp-topo",
        description=(
            "Infer bonds/angles/dihedrals from an input structure (LAMMPS data file) "
            "and write a new LAMMPS data file that includes the inferred topology. "
            "Bond inference uses distance windows centered at --rc with half-width --drc."
        ),
    )

    # Keep a positional input to keep your existing dev/hbn/auto.sh usable.
    p.add_argument(
        "input",
        help="Input structure file (LAMMPS data file or extxyz).",
    )

    p.add_argument(
        "-atom_style",
        "--atom_style",
        default='full',
        help="Lammps atom style.",
    )

    p.add_argument(
        "-units",
        "--units",
        default='metal',
        help="Lammps units.",
    )

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

    # Keep legacy-style int toggles for compatibility with your current scripts.
    p.add_argument("-bond", "--bond", type=int, default=0, help="Calculate bonds (0/1).")
    p.add_argument("-angle", "--angle", type=int, default=0, help="Calculate angles (0/1).")
    p.add_argument("-dihed", "--dihed", type=int, default=0, help="Calculate dihedrals (0/1).")

    # Typing rules
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

    # Outputs (kept names)
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

    atoms = read_lammps_data(str(in_path), atom_style=args.atom_style, units=args.units)

    if args.periodic is not None:
        try:
            periodicity = _parse_triplet_bools(args.periodic)
        except ValueError as exc:
            parser.exit(1, f"Invalid --periodic: {exc}\n")
        atoms.pbc = periodicity

    # Inputs for bond inference
    rc_list = _parse_float_list(args.rc)
    if args.drc < 0:
        parser.exit(1, "Error: --drc must be >= 0\n")

    calc_bonds = bool(args.bond)
    calc_angles = bool(args.angle)
    calc_diheds = bool(args.dihed)

    topo = Topo()
    neighbors = [[] for _ in range(len(atoms))]

    if calc_bonds:
        if not rc_list:
            parser.exit(1, "Error: --rc is required when --bond=1\n")
        topo_b, neighbors, _lens = infer_bonds_by_distance(
            atoms,
            rc_list=rc_list,
            drc=float(args.drc),
            deduplicate=True,
            return_lengths=False,
        )
        topo = topo_b
    else:
        topo = Topo(bonds=[])

    if calc_angles:
        if not topo.bonds:
            parser.exit(1, "Error: no bonds inferred; cannot build angles (set --bond=1 and valid --rc)\n")
        topo.angles = infer_angles_from_adjacency(neighbors, sort=True)

    if calc_diheds:
        if not topo.bonds:
            parser.exit(1, "Error: no bonds inferred; cannot build dihedrals (set --bond=1 and valid --rc)\n")
        topo.dihedrals = infer_dihedrals_from_bonds(topo.bonds, neighbors, sort=True)

    # Type assignment (tags -> 1-based type ids)
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

    if calc_bonds:
        type_bonds(atoms, topo, opts=opts)
    if calc_angles:
        type_angles(atoms, topo, opts=opts)
    if calc_diheds:
        type_dihedrals(atoms, topo, opts=opts)

    topo.validate(len(atoms), strict_types=False)

    # Outputs
    out_path = Path(args.file_pos)
    if out_path.exists() and not args.overwrite:
        parser.exit(1, f"Refusing to overwrite existing file: {out_path}\n")
    ExportLammpsDataFileTopo(str(out_path), atoms, topo)

    if args.file_types:
        types_path = Path(args.file_types)
        if types_path.exists() and not args.overwrite:
            parser.exit(1, f"Refusing to overwrite existing file: {types_path}\n")
        ExportTypesTopo(str(types_path), topo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
