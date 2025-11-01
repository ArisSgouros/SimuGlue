#!/usr/bin/env python3
from __future__ import annotations
import sys, re, json, argparse
from pathlib import Path
from glob import glob
from typing import Iterator
from simuglue import C

# -------------------- utils --------------------
def _strip_comment(line: str) -> str:
    # QE treats '!' as comment; keep content before the first unquoted '!'
    # (inputs are simple—ignore quoted edge cases for brevity)
    idx = line.find("!")
    return line[:idx] if idx >= 0 else line

def _clean_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = _strip_comment(raw).rstrip()
        if line:
            out.append(line)
        else:
            out.append("")  # keep block separators
    return out

def _find_block_start(lines: list[str], key: str) -> int | None:
    for i, ln in enumerate(lines):
        if ln.upper().startswith(key):
            return i
    return None

# --- card detection ---
_QE_CARD_NAMES = {
    # sections / cards commonly appearing in pw.x inputs
    "K_POINTS", "CELL_PARAMETERS", "ATOMIC_POSITIONS", "ATOMIC_SPECIES",
    "OCCUPATIONS", "CONSTRAINTS", "HUBBARD", "ATOMIC_FORCES",
    "STARTING_NS_EIGENVECTORS", "ATOMIC_VELOCITIES",
    # also common cards in tools/workflows
    "END", "EOF"
}

def _is_card_header(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # Namelists like &control, &system, &electrons, &ions, &cell, &phonon
    if s.startswith("&"):
        return True
    # Known cards, possibly with unit suffixes (e.g., "ATOMIC_POSITIONS angstrom")
    first = s.split()[0]
    return first.upper() in _QE_CARD_NAMES

def _block_iter(lines: list[str], start_idx: int):
    """Yield lines after 'start_idx' until a blank line or the next QE card header."""
    i = start_idx + 1
    while i < len(lines):
        raw = lines[i]
        if not raw.strip():
            break
        if _is_card_header(raw):
            break
        yield raw
        i += 1

import re

_number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"

def _var_pattern(var: str) -> re.Pattern:
    name = re.escape(var)
    # Token boundaries: no [A-Za-z0-9_] just before or after the var name.
    # Works for names like "A", "ecutwfc", and "celldm(1)".
    return re.compile(
        rf"(?<![A-Za-z0-9_]){name}(?![A-Za-z0-9_])\s*=\s*({_number})"
    )

def parse_qe_numeric(lines, var: str):
    """
    Parse first numeric assigned to `var`. Returns int if integer-like, else float; None if not found.
    Handles Fortran 'D' exponents.
    """
    pat = _var_pattern(var)
    for ln in lines:
        m = pat.search(ln)
        if m:
            s = m.group(1).replace("D", "E").replace("d", "e")
            return int(s) if re.fullmatch(r"[+-]?\d+", s) else float(s)
    return None


def _parse_units(header_line: str) -> str:
    if 'alat' in header_line:
        unit = 'alat'
    elif 'ang' in header_line:
        unit = 'angstrom'
    elif ('bohr' in header_line or 'a.u.' in header_line):
        unit = 'bohr'
    else: # default
        unit = 'alat'
    return unit

# ---------------- lattice helpers ----------------
def _parse_cell(lines: list[str]) -> tuple[list[float], str] | None:
    """
    Return ([ax,ay,az, bx,by,bz, cx,cy,cz] in Å, unit_tag_used)
    Supported input units: angstrom, bohr, alat
    Supported output unit: angstrom
    """
    unit_out = "angstrom"
    i = _find_block_start(lines, "CELL_PARAMETERS")
    if i is None:
        return None
    unit = _parse_units(lines[i])  # QE defaults CELL_PARAMETERS to 'alat'
    rows = []
    for j, ln in enumerate(_block_iter(lines, i)):
        parts = ln.split()
        if len(parts) < 3:
            break
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) == 3:
            break
    if len(rows) != 3:
        raise ValueError("CELL_PARAMETERS: need 3 rows with 3 numbers each")

    # Convert to Angstrom
    if unit == "angstrom" or unit == "ang":
        pass  # already Angstrom
    elif unit == "bohr" or unit == "a.u.":
        rows = [[v * C.BOHR_TO_ANGSTROM for v in r] for r in rows]
    elif unit == "alat":
        alat_bohr = parse_qe_numeric(lines, "celldm(1)")
        alat_angstrom = parse_qe_numeric(lines, "A")
        if (alat_bohr is not None) and (alat_angstrom is not None):
            raise ValueError(
                f"CELL_PARAMETERS (alat): ambiguous scale — both celldm(1)={alat_bohr} (bohr) "
                f"and A={alat_angstrom} (angstrom) are set. Specify only one."
            )
        elif alat_bohr is not None:
            scale = alat_bohr * C.BOHR_TO_ANGSTROM
        elif alat_angstrom is not None:
            scale = alat_angstrom
        else:
            raise ValueError("CELL_PARAMETERS (alat): missing celldm(1) or A to resolve alat")
        rows = [[v * scale for v in r] for r in rows]
    else:
        raise ValueError(f"CELL_PARAMETERS: unsupported unit '{unit}'")

    flat = [rows[0][0], rows[0][1], rows[0][2],
            rows[1][0], rows[1][1], rows[1][2],
            rows[2][0], rows[2][1], rows[2][2]]
    return flat, unit_out

def _matmul3(M: list[list[float]], v: list[float]) -> list[float]:
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]

# ---------------- positions ----------------
def _parse_positions(lines: list[str], lattice_A: list[float] | None) -> tuple[list[dict], str] | None:
    """
    Return (atoms_data, unit_tag_used). Positions converted to Å.
    Supports input units: angstrom, bohr, alat, crystal
    Supported output unit: angstrom
    """
    unit_out = "angstrom"
    i = _find_block_start(lines, "ATOMIC_POSITIONS")
    if i is None:
        return None
    unit = _parse_units(lines[i])  # QE default is alat
    # Prepare lattice matrix if needed
    L = None
    if lattice_A is not None:
        L = [
            [lattice_A[0], lattice_A[1], lattice_A[2]],
            [lattice_A[3], lattice_A[4], lattice_A[5]],
            [lattice_A[6], lattice_A[7], lattice_A[8]],
        ]

    alat_bohr = parse_qe_numeric(lines, "celldm(1)")
    alat_angstrom = parse_qe_numeric(lines, "A")
    if (alat_bohr is not None) and (alat_angstrom is not None):
        raise ValueError(
            f"CELL_PARAMETERS (alat): ambiguous scale — both celldm(1)={alat_bohr} (bohr) "
            f"and A={alat_angstrom} (angstrom) are set. Specify only one."
        )
    elif alat_bohr is not None:
        alat_scale_A = alat_bohr * C.BOHR_TO_ANGSTROM
    elif alat_angstrom is not None:
        alat_scale_A = alat_angstrom

    atoms: list[dict] = []
    for ln in _block_iter(lines, i):
        parts = ln.split()
        if len(parts) < 4:
            break
        sym = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            break  # stop at malformed line

        # Ignore any trailing if_pos flags or constraints
        posA: list[float]
        if unit in ("angstrom", "ang"):
            posA = [x, y, z]
        elif unit in ("bohr", "a.u."):
            posA = [x * C.BOHR_TO_ANGSTROM, y * C.BOHR_TO_ANGSTROM, z * C.BOHR_TO_ANGSTROM]
        elif unit == "alat":
            if alat_scale_A is None:
                raise ValueError("ATOMIC_POSITIONS (alat): missing celldm(1) or A to resolve alat")
            posA = [x * alat_scale_A, y * alat_scale_A, z * alat_scale_A]
        elif unit == "crystal":
            if L is None:
                raise ValueError("ATOMIC_POSITIONS (crystal) requires CELL_PARAMETERS")
            posA = _matmul3(L, [x, y, z])
        else:
            raise ValueError(f"ATOMIC_POSITIONS: unsupported unit '{unit}'")

        atoms.append({"symbols": sym, "positions": posA})

    if not atoms:
        return None
    return atoms, unit_out

# ---------------- top-level parse ----------------
def pwi2json(file_path: str | Path) -> dict:
    """
    Parse QE pw.x input and return json-compatible dict matching the output-parser shape.
    """
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    lines = _clean_lines(text)

    # CELL_PARAMETERS (→ Å)
    cell_tuple = _parse_cell(lines)
    if cell_tuple is None:
        raise ValueError("Could not find CELL_PARAMETERS block")
    cell_A, cell_unit = cell_tuple

    # ATOMIC_POSITIONS (→ Å)
    pos_tuple = _parse_positions(lines, lattice_A=cell_A)
    if pos_tuple is None:
        raise ValueError("Could not find ATOMIC_POSITIONS block")
    atoms, pos_unit = pos_tuple

    # Shape compatible with your out-parser
    data: dict = {
        "num_atoms": len(atoms),
        "cell_data": cell_A,  # Å
        "info": {
            "source": "qe_input",
            "cell_unit": cell_unit,
            "positions_unit": pos_unit,
        },
        "atoms_data": atoms,
    }

    return data
