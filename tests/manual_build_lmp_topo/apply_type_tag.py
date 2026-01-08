#!/usr/bin/env python3
"""
Append the *type-description string* from o.types to each Bonds/Angles/Dihedrals entry
in a LAMMPS data file.

New o.types specification (examples):

Bond Types
  1 B N

Angle Types
  1 B N B
  2 N B N

Dihedral Types
  1 B N B N cis
  2 B N B N trans

Rule:
- For each topology line in the data file, append:
    # <everything after the type id in o.types for that type>
  i.e. for a bond line with bond_type=1, append "# B N"
       for an angle line with angle_type=2, append "# N B N"
       for a dihedral line with dihedral_type=2, append "# B N B N trans"

Usage:
  python tag_lammps_topo.py in.data o.types out.data
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _is_section_header(line: str) -> Optional[str]:
    s = line.strip()
    for key in ("Masses", "Atoms", "Bonds", "Angles", "Dihedrals"):
        if s == key or s.startswith(key + " "):
            return key
    return None


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip("\n")


def _split_fields_no_comment(line: str) -> List[str]:
    core = _strip_comment(line).strip()
    return core.split() if core else []


def parse_o_types_strings(path: Path) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Parse o.types, returning:
      bond_type_str[type_id] = "B N"
      angle_type_str[type_id] = "B N B"
      dihedral_type_str[type_id] = "B N B N cis"
    """
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)

    bond_map: Dict[int, str] = {}
    angle_map: Dict[int, str] = {}
    dihedral_map: Dict[int, str] = {}

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
        if line == "Dihedral Types":
            section = "Dihedral Types"
            continue

        # Ignore headers/counters and other sections
        if line in ("Masses", "Atoms", "Bonds", "Angles", "Dihedrals"):
            section = None
            continue
        if line.endswith(" atom types") or line.endswith(" bond types") or line.endswith(" angle types") or line.endswith(" dihedral types"):
            continue

        if section in ("Bond Types", "Angle Types", "Dihedral Types"):
            f = _split_fields_no_comment(raw)
            if len(f) >= 2:
                try:
                    tid = int(f[0])
                except ValueError:
                    continue
                desc = " ".join(f[1:]).strip()
                if section == "Bond Types":
                    bond_map[tid] = desc
                elif section == "Angle Types":
                    angle_map[tid] = desc
                else:
                    dihedral_map[tid] = desc

    return bond_map, angle_map, dihedral_map


def tag_datafile(data_in: Path, otypes_in: Path, data_out: Path) -> None:
    bond_map, angle_map, dihedral_map = parse_o_types_strings(otypes_in)

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
            # <id> <type> <i> <j>
            f = _split_fields_no_comment(raw)
            if len(f) >= 4:
                try:
                    btype = int(f[1])
                except ValueError:
                    out_lines.append(raw)
                    continue
                desc = bond_map.get(btype)
                if desc:
                    base = _strip_comment(raw).rstrip()
                    out_lines.append(f"{base}  # {desc}\n")
                    continue
            out_lines.append(raw)
            continue

        if section == "Angles":
            # <id> <type> <i> <j> <k>
            f = _split_fields_no_comment(raw)
            if len(f) >= 5:
                try:
                    atype = int(f[1])
                except ValueError:
                    out_lines.append(raw)
                    continue
                desc = angle_map.get(atype)
                if desc:
                    base = _strip_comment(raw).rstrip()
                    out_lines.append(f"{base}  # {desc}\n")
                    continue
            out_lines.append(raw)
            continue

        if section == "Dihedrals":
            # <id> <type> <i> <j> <k> <l>
            f = _split_fields_no_comment(raw)
            if len(f) >= 6:
                try:
                    dtype = int(f[1])
                except ValueError:
                    out_lines.append(raw)
                    continue
                desc = dihedral_map.get(dtype)
                if desc:
                    base = _strip_comment(raw).rstrip()
                    out_lines.append(f"{base}  # {desc}\n")
                    continue
            out_lines.append(raw)
            continue

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

    tag_datafile(data_in, otypes_in, data_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

