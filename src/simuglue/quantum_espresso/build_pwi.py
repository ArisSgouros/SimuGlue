#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from ase import Atoms


def _format_cell_parameters(atoms: Atoms) -> str:
    cell = atoms.get_cell().array  # (3,3)
    lines = ["CELL_PARAMETERS {angstrom}"]
    for vec in cell:
        lines.append(f"{vec[0]:.16f} {vec[1]:.16f} {vec[2]:.16f}")
    return "\n".join(lines)


def _format_atomic_positions(atoms: Atoms) -> str:
    pos = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    if len(pos) != len(syms):
        raise ValueError("Mismatch between positions and symbols.")
    lines = ["ATOMIC_POSITIONS {angstrom}"]
    for s, (x, y, z) in zip(syms, pos):
        lines.append(
            f"{s} {x:.16f} {y:.16f} {z:.16f} "
        )
    return "\n".join(lines)


def build_pwi_from_header(
    header_text: str,
    atoms: Atoms,
) -> str:
    """
    Assemble QE input by concatenating a provided header (without
    CELL_PARAMETERS / ATOMIC_POSITIONS) with those sections from atoms.
    """
    header_text = header_text.rstrip()
    parts = [
        header_text,
        _format_cell_parameters(atoms),
        _format_atomic_positions(atoms),
    ]
    return "\n".join(parts)

