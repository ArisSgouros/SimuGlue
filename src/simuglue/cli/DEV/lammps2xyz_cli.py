#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Union
import sys

import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.io.lammpsdata import read_lammps_data

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


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Apply a linear deformation gradient to a lammps data file.")
    p.add_argument("data_in", help="Input lammps data file.")
    return p

def main(argv=None, prog: str | None = None) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    data_in_path = Path(args.data_in)
    if not data_in_path.exists():
        raise FileNotFoundError(f"XYZ not found: {data_in_path}")

    #Eatoms = read_lammps(data_in_path, style="atomic", types="1=Si,2=O,3=H")
    type_map = {1: "Bi", 2: "Se", 3: "H"}
    style = "atomic"
    atoms = read_lammps_data(str(data_in_path), style=style)
    symbols = [type_map[t] for t in atoms.get_array('type')]
    atoms.set_chemical_symbols(symbols)
    #atoms = read(
    #    str(data_in_path),
    #    format="lammps-data",
    #    style=style,
    #    type_map=type_map
    #    #Z_of_type=Z_of_type
    #)

    write(sys.stdout, atoms, format="extxyz")
    #write_lammps(atoms_def, f"{args.output}", style="atomic", units="metal", force_skew=True)

if __name__ == "__main__":
    main()
