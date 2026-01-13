from simuglue.topology.topo import Topo
from typing import Optional, Dict

def ExportTypesTopo(
    filename: str,
    topo: Topo,
    masses_table: Optional[Dict] = None,
) -> None:
    """Export type tables from `topo.meta` and optionally a LAMMPS-style Masses section.

    Parameters
    ----------
    filename : str
        Output file.
    topo : Topo
        Topology object with `meta` tables.
    masses_table : dict, optional
        LAMMPS type table as returned by `_get_lmp_type_table(atoms)`.
        Expected keys: 'n_types', 'mass', 'tag' (all optional but recommended).
    """
    with open(filename, 'w') as fout:
        fout.write('\n')

        # --- Topology type tables ---
        bt = topo.meta.get('bond_type_table', {})
        at = topo.meta.get('angle_type_table', {})
        dt = topo.meta.get('dihedral_type_table', {})

        if bt:
            fout.write(f"{len(bt)} bond types\n")
        if at:
            fout.write(f"{len(at)} angle types\n")
        if dt:
            fout.write(f"{len(dt)} dihedral types\n")

        # --- Masses header (if provided) ---
        if masses_table:
            n_types = int(masses_table.get("n_types", 0))
            if n_types <= 0:
                # infer from keys if not explicitly provided
                mass_map = masses_table.get("mass", {})
                if mass_map:
                    n_types = max(mass_map.keys())
            if n_types > 0:
                fout.write(f"{n_types} atom types\n")

        fout.write('\n')

        # --- Masses section ---
        if masses_table:
            fout.write("Masses\n\n")
            mass_map = masses_table.get("mass", {})
            tag_map = masses_table.get("tag", {})
            n_types = int(masses_table.get("n_types", 0))
            if n_types <= 0 and mass_map:
                n_types = max(mass_map.keys())

            for tid in range(1, n_types + 1):
                mass = mass_map.get(tid, 0.0)
                tag = tag_map.get(tid, "")
                if tag:
                    fout.write(f"{tid} {mass} # {tag}\n")
                else:
                    fout.write(f"{tid} {mass}\n")
            fout.write('\n')

        # --- Bond types ---
        if bt:
            fout.write('Bond Types\n\n')
            for tid in sorted(bt.keys()):
                fout.write(f"{tid} {bt[tid].get('tag','')}\n")

        # --- Angle types ---
        if at:
            fout.write('\nAngle Types\n\n')
            for tid in sorted(at.keys()):
                fout.write(f"{tid} {at[tid].get('tag','')}\n")

        # --- Dihedral types ---
        if dt:
            fout.write('\nDihedral Types\n\n')
            for tid in sorted(dt.keys()):
                fout.write(f"{tid} {dt[tid].get('tag','')}\n")

