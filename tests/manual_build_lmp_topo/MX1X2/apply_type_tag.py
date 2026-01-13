#!/usr/bin/env python3
"""
Tag a LAMMPS data file's Bonds/Angles entries using an o.types mapping file.

Usage:
  python tag_lammps_topo.py in.data o.types out.data

What it does:
- Reads Masses section of the LAMMPS data file to map: atom_type_id -> element/tag (e.g. 1 -> "H")
- Reads Bond Types / Angle Types sections of o.types:
    Bond Types:  <bond_type_id> <a> <b>
    Angle Types: <angle_type_id> <a> <b> <c> <extra_tag>
  where a,b,c are atom type ids (as integers) and extra_tag is a string (e.g. "T","A","N")
- Appends comments to each "Bonds" and "Angles" line in the data file:
    Bonds:  # <elem(a)> <elem(b)>
    Angles: # <elem(a)> <elem(b)> <elem(c)> <extra_tag>
- Leaves everything else unchanged.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


_SECTION_HEADERS = {
    "Masses": "Masses",
    "Atoms": "Atoms",
    "Bonds": "Bonds",
    "Angles": "Angles",
}


def _is_section_header(line: str) -> Optional[str]:
    s = line.strip()
    # Accept e.g. "Atoms # full"
    for key in _SECTION_HEADERS:
        if s == key or s.startswith(key + " "):
            return key
    return None


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip("\n")


def _split_fields_no_comment(line: str) -> List[str]:
    core = _strip_comment(line).strip()
    return core.split() if core else []


@dataclass
class TypesDB:
    atom_type_to_tag: Dict[int, str]
    bond_type_to_pair: Dict[int, Tuple[int, int]]
    angle_type_to_triplet_tag: Dict[int, Tuple[int, int, int, str]]


def parse_o_types(path: Path) -> Tuple[Dict[int, Tuple[int, int]], Dict[int, Tuple[int, int, int, str]]]:
    """
    Parse o.types file.

    Expected format:

    Bond Types

    1 1 2
    2 1 3

    Angle Types

    1 1 2 1 T
    2 1 3 1 T
    ...
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    bond_map: Dict[int, Tuple[int, int]] = {}
    angle_map: Dict[int, Tuple[int, int, int, str]] = {}

    section: Optional[str] = None
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line == "Bond Types":
            section = "Bond Types"
            continue
        if line == "Angle Types":
            section = "Angle Types"
            continue
        if line in ("Masses", "2 bond types", "6 angle types", "3 atom types"):
            # ignore header-like lines
            continue

        if section == "Bond Types":
            f = _split_fields_no_comment(raw)
            if len(f) >= 3:
                bt = int(f[0]); a = int(f[1]); b = int(f[2])
                bond_map[bt] = (a, b)
            continue

        if section == "Angle Types":
            f = _split_fields_no_comment(raw)
            # angle_id a b c tag
            if len(f) >= 5:
                at = int(f[0]); a = int(f[1]); b = int(f[2]); c = int(f[3]); t = str(f[4])
                angle_map[at] = (a, b, c, t)
            continue

    return bond_map, angle_map


def parse_data_masses_atomtags(path: Path) -> Dict[int, str]:
    """
    Parse the 'Masses' section of a LAMMPS data file to build: atom_type_id -> tag.
    Uses the trailing '# TAG' if present; otherwise uses the integer type id as string.
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    section: Optional[str] = None
    atom_type_to_tag: Dict[int, str] = {}

    i = 0
    while i < len(lines):
        raw = lines[i]
        hdr = _is_section_header(raw)
        if hdr is not None:
            section = hdr
            i += 1
            continue

        if section == "Masses":
            # Mass line examples:
            # "     1      1.9999 # H"
            # "2 6.0 # Li"
            f = _split_fields_no_comment(raw)
            if len(f) >= 2:
                try:
                    t = int(f[0])
                except ValueError:
                    i += 1
                    continue
                tag = None
                if "#" in raw:
                    tag = raw.split("#", 1)[1].strip()
                if not tag:
                    tag = str(t)
                atom_type_to_tag[t] = tag
        i += 1

    return atom_type_to_tag


def tag_topology(
    data_in: Path,
    otypes_in: Path,
    data_out: Path,
) -> None:
    bond_map, angle_map = parse_o_types(otypes_in)
    atom_tag = parse_data_masses_atomtags(data_in)

    lines = data_in.read_text(encoding="utf-8", errors="replace").splitlines(True)

    out_lines: List[str] = []
    section: Optional[str] = None

    for raw in lines:
        hdr = _is_section_header(raw)
        if hdr is not None:
            section = hdr
            out_lines.append(raw)
            continue

        if section == "Bonds":
            # Bond line: id type i j
            f = _split_fields_no_comment(raw)
            if len(f) >= 4:
                try:
                    btype = int(f[1])
                except ValueError:
                    out_lines.append(raw)
                    continue

                pair = bond_map.get(btype, None)
                if pair is not None:
                    a_t, b_t = pair
                    a_tag = atom_tag.get(a_t, str(a_t))
                    b_tag = atom_tag.get(b_t, str(b_t))
                    base = _strip_comment(raw).rstrip()
                    out_lines.append(f"{base}  # {a_tag} {b_tag}\n")
                    continue

            out_lines.append(raw)
            continue

        if section == "Angles":
            # Angle line: id type i j k
            f = _split_fields_no_comment(raw)
            if len(f) >= 5:
                try:
                    atype = int(f[1])
                except ValueError:
                    out_lines.append(raw)
                    continue

                spec = angle_map.get(atype, None)
                if spec is not None:
                    a_t, b_t, c_t, extra = spec
                    a_tag = atom_tag.get(a_t, str(a_t))
                    b_tag = atom_tag.get(b_t, str(b_t))
                    c_tag = atom_tag.get(c_t, str(c_t))
                    base = _strip_comment(raw).rstrip()
                    out_lines.append(f"{base}  # {a_tag} {b_tag} {c_tag} {extra}\n")
                    continue

            out_lines.append(raw)
            continue

        # other sections: unchanged
        out_lines.append(raw)

    data_out.write_text("".join(out_lines), encoding="utf-8")


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print("Usage: python tag_lammps_topo.py in.data o.types out.data", file=sys.stderr)
        return 2

    data_in = Path(argv[1])
    otypes_in = Path(argv[2])
    data_out = Path(argv[3])

    if not data_in.exists():
        print(f"ERROR: input data file not found: {data_in}", file=sys.stderr)
        return 2
    if not otypes_in.exists():
        print(f"ERROR: o.types file not found: {otypes_in}", file=sys.stderr)
        return 2

    tag_topology(data_in, otypes_in, data_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

