#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from simuglue.ase_patches.lammpsdata import (
    read_lammps_data,
    write_lammps_data,
    get_lmp_type_table,
)

from simuglue.topology.core import (
    build_topology_from_atoms,
    ensure_atom_tags_from_lmp_type_table,
    type_topology_inplace,
    attach_topology_arrays_to_atoms,
)
from simuglue.topology.typing import TypingOptions
from simuglue.topology.export import ExportTypesTopo
from simuglue.cli._parser import _parse_triplet_bools, _parse_float_list


def _refuse_overwrite(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {path} (use --overwrite)")


def _make_typing_options(args: argparse.Namespace) -> TypingOptions:
    return TypingOptions(
        diff_bond_len=bool(args.diff_bond_len),
        diff_bond_fmt=str(args.bond_len_fmt),
        angle_symmetry=bool(args.angle_symmetry),
        diff_angle_theta=bool(args.diff_angle_theta),
        diff_angle_theta_fmt=str(args.angle_theta_fmt),
        cis_trans=bool(args.cis_trans),
        diff_dihed_theta=bool(args.diff_dihed_theta),
        diff_dihed_theta_abs=bool(args.dihed_abs),
        diff_dihed_theta_fmt=str(args.dihed_theta_fmt),
        type_delimiter=str(args.type_delimiter),
    )


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog or "sgl build lmp-topo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Infer topology (bonds/angles/dihedrals) for a LAMMPS data file and write a new data file.",
    )

    # I/O
    p.add_argument("-i", "--input", required=True, help="Input LAMMPS data file.")
    p.add_argument("-o", "--output", required=True, help="Output LAMMPS data file.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    p.add_argument("--types-out", default="", help="Optional: export type tables to this file (o.types style).")
    p.add_argument("--dry-run", action="store_true", help="Infer + type, print counts, do not write output files.")

    # LAMMPS I/O options
    g_io = p.add_argument_group("LAMMPS I/O")
    g_io.add_argument("--units", default="metal", help="LAMMPS units for reading.")
    g_io.add_argument("--in-atom-style", default="full", help="LAMMPS atom_style for reading.")
    g_io.add_argument("--out-atom-style", default=None, help="LAMMPS atom_style for writing (default: same as input).")
    g_io.add_argument("--pbc", type=_parse_triplet_bools, default=None, help="Override periodicity, e.g. 1,1,1.")

    # Topology inference
    g_top = p.add_argument_group("Topology inference")
    g_top.add_argument("--bonds", action="store_true", help="Infer bonds.")
    g_top.add_argument("--angles", action="store_true", help="Infer angles (implies bonds).")
    g_top.add_argument("--dihedrals", action="store_true", help="Infer dihedrals (implies bonds).")

    g_top.add_argument("--rc", type=_parse_float_list, default=None, help="Bond reference lengths, e.g. 1.44,1.53.")
    g_top.add_argument("--drc", type=float, default=0.01, help="Bond tolerance half-width.")

    # Typing
    g_typ = p.add_argument_group("Typing")
    g_typ.add_argument("--no-typing", action="store_true", help="Skip type assignment (not recommended).")
    g_typ.add_argument("--type-delimiter", default=" ", help="Delimiter used to join type tag parts.")

    g_typ.add_argument("--diff-bond-len", action="store_true", help="Differentiate bond types by length.")
    g_typ.add_argument("--bond-len-fmt", default="%.2f", help="Printf-style format for bond lengths.")

    g_typ.add_argument("--angle-symmetry", action="store_true", help="Add angle symmetry label to type tags.")
    g_typ.add_argument("--diff-angle-theta", action="store_true", help="Differentiate angle types by theta.")
    g_typ.add_argument("--angle-theta-fmt", default="%.2f", help="Printf-style format for angle theta (deg).")

    g_typ.add_argument("--cis-trans", action="store_true", help="Add cis/trans to dihedral type tags.")
    g_typ.add_argument("--diff-dihed-theta", action="store_true", help="Differentiate dihedral types by phi.")
    g_typ.add_argument("--dihed-abs", dest="dihed_abs", action="store_true", default=True, help="Use |phi| for typing.")
    g_typ.add_argument("--no-dihed-abs", dest="dihed_abs", action="store_false", help="Use signed phi for typing.")
    g_typ.add_argument("--dihed-theta-fmt", default="%.2f", help="Printf-style format for dihedral phi (deg).")

    return p


def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")

    out_path = Path(args.output)
    if not args.dry_run:
        _refuse_overwrite(out_path, args.overwrite)
        if args.types_out:
            _refuse_overwrite(Path(args.types_out), args.overwrite)

    if args.drc < 0:
        raise SystemExit("--drc must be >= 0")

    # Normalize requested operations
    calc_bonds = bool(args.bonds)
    calc_angles = bool(args.angles)
    calc_diheds = bool(args.dihedrals)

    # Friendly behavior: angles/dihedrals imply bonds
    if (calc_angles or calc_diheds) and not calc_bonds:
        calc_bonds = True

    if not (calc_bonds or calc_angles or calc_diheds):
        raise SystemExit("Select at least one: --bonds, --angles or --dihedrals")

    if calc_bonds and args.rc is None:
        raise SystemExit("--rc is required when inferring bonds (directly or via --angles/--dihedrals)")

    out_atom_style = args.out_atom_style or args.in_atom_style

    atoms = read_lammps_data(str(in_path), atom_style=args.in_atom_style, units=args.units)
    if args.pbc is not None:
        atoms.pbc = args.pbc

    topo, _neighbors = build_topology_from_atoms(
        atoms,
        calc_bonds=calc_bonds,
        calc_angles=calc_angles,
        calc_diheds=calc_diheds,
        rc_list=args.rc or [],
        drc=float(args.drc),
        deduplicate=True,
    )

    # LAMMPS type table (needed for atom tags and optional types-out)
    lmp_type_table = get_lmp_type_table(atoms)
    if lmp_type_table is None:
        raise SystemExit("Could not parse lmp_type_table from input (Masses/type tags missing?)")

    if args.no_typing:
        raise SystemExit("--no-typing is currently unsupported for LAMMPS topology output (types are required).")

    atom_tag = ensure_atom_tags_from_lmp_type_table(lmp_type_table)
    typing_opts = _make_typing_options(args)

    type_topology_inplace(
        atoms,
        topo,
        atom_tag=atom_tag,
        opts=typing_opts,
        calc_bonds=calc_bonds,
        calc_angles=calc_angles,
        calc_diheds=calc_diheds,
    )

    attach_topology_arrays_to_atoms(atoms, topo)

    # Dry run output
    if args.dry_run:
        nb = len(topo.bonds or [])
        na = len(topo.angles or [])
        nd = len(topo.dihedrals or [])
        print(f"atoms: {len(atoms)}")
        print(f"bonds: {nb} (types: {len(set(topo.bond_types or [])) if topo.bond_types else 0})")
        print(f"angles: {na} (types: {len(set(topo.angle_types or [])) if topo.angle_types else 0})")
        print(f"dihedrals: {nd} (types: {len(set(topo.dihedral_types or [])) if topo.dihedral_types else 0})")
        if args.types_out:
            print("note: --types-out ignored in --dry-run")
        return 0

    # Write outputs
    write_lammps_data(
        str(out_path),
        atoms,
        atom_style=out_atom_style,
        masses=True,
        force_skew=True,
        preserve_atom_types=True,
    )

    if args.types_out:
        ExportTypesTopo(str(Path(args.types_out)), topo, lmp_type_table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

