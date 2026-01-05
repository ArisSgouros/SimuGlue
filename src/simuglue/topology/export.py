import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from simuglue.io.lammps_cell import ase_cell_to_lammps_triclinic
from simuglue.topology.topo import Topo


def ExportLammpsDataFileTopo(
    filename: str,
    atoms,
    topo: Optional["Topo"],
    *,
    atom_types: Optional[Dict[int, Dict[str, object]]] = None,
    style: str = "full",
) -> None:
    """Write a LAMMPS data file from an ASE Atoms object + optional Topo.

    If topo is None, bond/angle/dihedral/etc sections and counts are omitted.
    """
    lx, ly, lz, xy, xz, yz = ase_cell_to_lammps_triclinic(atoms.cell)

    xlo = ylo = zlo = 0.0
    xhi, yhi, zhi = float(lx), float(ly), float(lz)
    is_triclinic = (abs(xy) > 0.0 or abs(xz) > 0.0 or abs(yz) > 0.0)

    natoms = len(atoms)

    has_topo = topo is not None
    nb = len(topo.bonds) if has_topo else 0
    na = len(topo.angles) if has_topo else 0
    nd = len(topo.dihedrals) if has_topo else 0
    # If you later add impropers, etc., follow the same pattern.

    # Atom types
    if "type" in atoms.arrays:
        atypes = [int(x) for x in atoms.arrays["type"]]
    else:
        # symbol -> 1..K mapping
        syms = atoms.get_chemical_symbols()
        mapping: Dict[str, int] = {}
        atypes = []
        for s in syms:
            if s not in mapping:
                mapping[s] = len(mapping) + 1
            atypes.append(mapping[s])

    n_atom_types = int(max(atypes)) if atypes else 0

    # Masses per type (prefer actual masses)
    masses = atoms.get_masses()
    mass_by_type: Dict[int, float] = {}
    for i, t in enumerate(atypes):
        if t not in mass_by_type:
            mass_by_type[t] = float(masses[i])

    # Optional type names
    name_by_type: Dict[int, str] = {}
    if atom_types is not None:
        for t, d in atom_types.items():
            name_by_type[int(t)] = str(d.get("name", t))
    else:
        # fallback: derive from symbols (first occurrence)
        syms = atoms.get_chemical_symbols()
        for i, t in enumerate(atypes):
            if t not in name_by_type:
                name_by_type[t] = syms[i]

    # Molecule ids
    mol = None
    for key in ("mol-id", "molid", "mol", "molecule"):
        if key in atoms.arrays:
            mol = atoms.arrays[key]
            break
    molids = [1] * natoms if mol is None else [int(x) for x in mol]

    # Charges
    qq = None
    for key in ("mmcharges", "initial_charges", "charges", "charge"):
        if key in atoms.arrays:
            qq = atoms.arrays[key]
            break
    charges = [0.0] * natoms if qq is None else [float(x) for x in qq]

    # Types counts (only if topo exists)
    n_bond_types = int(max(topo.bond_types)) if (has_topo and topo.bond_types) else 0
    n_angle_types = int(max(topo.angle_types)) if (has_topo and topo.angle_types) else 0
    n_dihed_types = int(max(topo.dihedral_types)) if (has_topo and topo.dihedral_types) else 0

    with open(filename, "w") as f:
        f.write("# LAMMPS data file (SimuGlue lmp-topo)\n\n")
        f.write(f"{natoms} atoms\n")
        if has_topo:
            f.write(f"{nb} bonds\n")
            if na:
                f.write(f"{na} angles\n")
            if nd:
                f.write(f"{nd} dihedrals\n")
        f.write("\n")

        f.write(f"{n_atom_types} atom types\n")
        if has_topo:
            if nb:
                f.write(f"{n_bond_types} bond types\n")
            if na:
                f.write(f"{n_angle_types} angle types\n")
            if nd:
                f.write(f"{n_dihed_types} dihedral types\n")
        f.write("\n")

        f.write(f"{xlo:<16.9f} {xhi:<16.9f} xlo xhi\n")
        f.write(f"{ylo:<16.9f} {yhi:<16.9f} ylo yhi\n")
        f.write(f"{zlo:<16.9f} {zhi:<16.9f} zlo zhi\n")
        if is_triclinic:
            f.write(f"{xy:<16.9f} {xz:<16.9f} {yz:<16.9f} xy xz yz\n")

        f.write("\nMasses\n\n")
        for t in range(1, n_atom_types + 1):
            f.write(
                f"{t:6d} {mass_by_type.get(t, 0.0):16.9f} # {name_by_type.get(t, t)}\n"
            )

        f.write("\nAtoms # full\n\n")
        pos = atoms.positions
        for i in range(natoms):
            aid = i + 1
            f.write(
                f"{aid:6d} {molids[i]:4d} {atypes[i]:4d} {charges[i]:16.9f} "
                f"{pos[i,0]:16.9f} {pos[i,1]:16.9f} {pos[i,2]:16.9f} # {name_by_type.get(atypes[i], '')}\n"
            )

        if not has_topo:
            return

        # Topology sections
        if nb:
            f.write("\nBonds\n\n")
            btypes = topo.bond_types or [1] * nb
            btags = topo.bond_tags
            for idx, ((i, j), bt) in enumerate(zip(topo.bonds, btypes)):
                comment = f" # {btags[idx]}" if btags is not None else ""
                f.write(f"{idx+1:6d} {bt:4d} {i+1:6d} {j+1:6d}{comment}\n")

        if na:
            f.write("\nAngles\n\n")
            atypes2 = topo.angle_types or [1] * na
            atags = topo.angle_tags
            for idx, ((i, j, k), at) in enumerate(zip(topo.angles, atypes2)):
                comment = f" # {atags[idx]}" if atags is not None else ""
                f.write(f"{idx+1:6d} {at:4d} {i+1:6d} {j+1:6d} {k+1:6d}{comment}\n")

        if nd:
            f.write("\nDihedrals\n\n")
            dtypes = topo.dihedral_types or [1] * nd
            dtags = topo.dihedral_tags
            for idx, ((i, j, k, l), dt) in enumerate(zip(topo.dihedrals, dtypes)):
                comment = f" # {dtags[idx]}" if dtags is not None else ""
                f.write(
                    f"{idx+1:6d} {dt:4d} {i+1:6d} {j+1:6d} {k+1:6d} {l+1:6d}{comment}\n"
                )


def ExportTypesTopo(filename: str, topo: Topo) -> None:
   """Export type tables from `topo.meta` (if available)."""
   with open(filename, 'w') as fout:
      fout.write('\n')
      bt = topo.meta.get('bond_type_table', {})
      at = topo.meta.get('angle_type_table', {})
      dt = topo.meta.get('dihedral_type_table', {})
      if bt:
         fout.write(f"{len(bt)} bond types\n")
      if at:
         fout.write(f"{len(at)} angle types\n")
      if dt:
         fout.write(f"{len(dt)} dihedral types\n")
      fout.write('\n')
      if bt:
         fout.write('Bond Types\n\n')
         for tid in sorted(bt.keys()):
            fout.write(f"{tid} {bt[tid].get('tag','')}\n")
      if at:
         fout.write('\nAngle Types\n\n')
         for tid in sorted(at.keys()):
            fout.write(f"{tid} {at[tid].get('tag','')}\n")
      if dt:
         fout.write('\nDihedral Types\n\n')
         for tid in sorted(dt.keys()):
            fout.write(f"{tid} {dt[tid].get('tag','')}\n")
