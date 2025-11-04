#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from ase.io import read, write
import numpy as np

# ------------------------- helpers -------------------------

def _parse_type_map(s: str) -> Dict[int, str]:
    """Parse '1=Si,2=O' -> {1:'Si', 2:'O'}; whitespace tolerated."""
    mapping: Dict[int, str] = {}
    if not s:
        return mapping
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split("=")
        mapping[int(k.strip())] = v.strip()
    return mapping


def read_lammps(input_file: str, style: str = "atomic", types: str = "") -> Atoms:
    """
    Read a LAMMPS data file.

    Parameters
    ----------
    input_file : str
        Path to LAMMPS data file.
    style : str
        LAMMPS atom style used in the file ('atomic', 'charge', 'molecular', 'full').
    types : str
        Optional mapping like "1=Si,2=O" to map integer types -> element symbols.

    Returns
    -------
    Atoms
    """
    in_path = Path(input_file)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    symbol_of_type = _parse_type_map(types)
    Z_of_type = {itype: atomic_numbers[sym] for itype, sym in symbol_of_type.items()} if symbol_of_type else None

    atoms = read(
        str(in_path),
        format="lammps-data",
        style=style,
        Z_of_type=Z_of_type
    )
    return atoms

def write_lammps(
    atoms: Atoms,
    output_file: str,
    style: str = "atomic",
    units: str = "metal",
    force_skew: bool = False,
    write_velocities: bool = True,
) -> Path:
    """
    Write a LAMMPS data file.

    Parameters
    ----------
    atoms : Atoms
        Structure to write.
    output_file : str
        Path to output .data
    style : str
        LAMMPS atom style ('atomic', 'charge', 'molecular', 'full').
    units : str
        LAMMPS units ('metal', 'real', 'si', 'lj', 'cgs', 'electron').
    force_skew : bool
        Allow triclinic (skewed) cell output if present.
    write_velocities : bool
        Include velocities if present.

    Returns
    -------
    Path
    """
    out_path = Path(output_file)
    has_vel = atoms.get_velocities() is not None

    kw = {
        "format": "lammps-data",
        "atom_style": style,
        "units": units,
        "force_skew": force_skew,
    }

    if has_vel and write_velocities:
        try:
            write(str(out_path), atoms, **kw, velocities=True)
        except TypeError:
            # Older ASE versions may not accept 'velocities' kwarg
            write(str(out_path), atoms, **kw)
    else:
        write(str(out_path), atoms, **kw)

    return out_path

