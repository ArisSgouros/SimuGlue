from __future__ import annotations

from pathlib import Path
from typing import Optional

from ase.atoms import Atoms
from ase.io.lammpsdata import write_lammps_data


def _find_box_lines(lines):
    """Return mapping name->line for box lines in a LAMMPS data file."""
    box = {}
    for line in lines:
        s = line.strip()
        if not s:
            continue
        lower = s.lower()
        if "xlo" in lower and "xhi" in lower:
            box["x"] = line
        elif "ylo" in lower and "yhi" in lower:
            box["y"] = line
        elif "zlo" in lower and "zhi" in lower:
            box["z"] = line
        elif "xy" in lower and "xz" in lower and "yz" in lower:
            box["tilt"] = line
    return box


def _find_atoms_block(lines):
    """
    Find [start, end) indices for the Atoms block, including header line.

    Returns (start, end) where:
      - lines[start] is the "Atoms" header line
      - lines[start+1:end] are the atom records
    """
    start = None
    end = None
    in_atoms = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Atoms"):
            start = i
            in_atoms = True
            continue
        if in_atoms:
            # End conditions: blank line or another section header
            if not stripped:
                end = i
                break
            if any(stripped.startswith(sec) for sec in (
                "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
                "Masses", "Pair Coeffs", "Bond Coeffs", "Angle Coeffs",
                "Dihedral Coeffs", "Improper Coeffs",
            )):
                end = i
                break
    if in_atoms and end is None:
        end = len(lines)
    if start is None:
        raise ValueError("Could not find 'Atoms' section in topology data file")
    return start, end


def _detect_atom_style_from_atoms_header(line: str) -> Optional[str]:
    """
    Parse an 'Atoms' header line like 'Atoms # full' or 'Atoms # atomic'
    and return the style, or None if not present.
    """
    stripped = line.strip()
    if "#" in stripped:
        after = stripped.split("#", 1)[1].strip()
        return after or None
    return None


def write_lammps_data_with_topology(
    atoms: Atoms,
    topology_data: Path | str,
    output_data: Path | str,
    *,
    units: str = "metal",
    atom_style: Optional[str] = None,
    overwrite: bool = False,
) -> None:
    """
    Write a LAMMPS data file that preserves topology (bonds/angles/etc.)
    from an existing data file, but updates box bounds and atomic positions
    from a provided ASE Atoms object.

    Parameters
    ----------
    atoms
        ASE Atoms object with the desired cell and atomic positions.
    topology_data
        Path to an existing LAMMPS data file containing the desired topology.
    output_data
        Path to the output data file. May be the same as ``topology_data``
        if overwrite=True.
    units
        LAMMPS units flag to pass to ASE's write_lammps_data (default 'metal').
    atom_style
        LAMMPS atom_style to pass to ASE. If None, will attempt to infer from
        the 'Atoms' header of ``topology_data`` (e.g. 'Atoms # full').
    overwrite
        If False (default) and output_data already exists, a FileExistsError
        is raised.
    """
    topo_path = Path(topology_data)
    out_path = Path(output_data)

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    topo_lines = topo_path.read_text(encoding="utf-8").splitlines(keepends=True)

    # Detect atom_style if not provided
    if atom_style is None:
        start, _ = _find_atoms_block(topo_lines)
        atom_style = _detect_atom_style_from_atoms_header(topo_lines[start])
        if atom_style is None:
            raise ValueError(
                "Could not infer atom_style from 'Atoms' header; "
                "please pass atom_style explicitly."
            )

    # Write a temporary data file using ASE for the new geometry
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp_geom")
    write_lammps_data(
        tmp_path,
        atoms,
        atom_style=atom_style,
        units=units,
        force_skew=True,
    )

    tmp_lines = tmp_path.read_text(encoding="utf-8").splitlines(keepends=True)
    tmp_path.unlink(missing_ok=True)

    # Extract box bounds and Atoms block from temp file
    tmp_box = _find_box_lines(tmp_lines)
    atoms_start, atoms_end = _find_atoms_block(tmp_lines)
    atoms_block = tmp_lines[atoms_start:atoms_end]

    # Patch topology header with new box bounds
    new_lines = topo_lines.copy()
    for i, line in enumerate(new_lines):
        lower = line.strip().lower()
        if "xlo" in lower and "xhi" in lower and "x" in tmp_box:
            new_lines[i] = tmp_box["x"]
        elif "ylo" in lower and "yhi" in lower and "y" in tmp_box:
            new_lines[i] = tmp_box["y"]
        elif "zlo" in lower and "zhi" in lower and "z" in tmp_box:
            new_lines[i] = tmp_box["z"]
        elif "xy" in lower and "xz" in lower and "yz" in lower and "tilt" in tmp_box:
            new_lines[i] = tmp_box["tilt"]

    # Replace Atoms block
    topo_atoms_start, topo_atoms_end = _find_atoms_block(new_lines)
    new_lines[topo_atoms_start:topo_atoms_end] = atoms_block

    out_path.write_text("".join(new_lines), encoding="utf-8")

