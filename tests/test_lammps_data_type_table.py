# tests/test_lammpsdata_type_table.py
import base64
import io
import json
import re
from typing import Dict, Tuple, List

import numpy as np
import pytest

from simuglue.ase_patches.lammpsdata import (
    read_lammps_data,
    write_lammps_data,
    get_lmp_type_table,  # internal helper in your patch
)


def _lammps_data_with_tags_and_unused_type() -> str:
    # Header declares 4 types, but Atoms only use types {1,2,4} (type 3 unused).
    # Atoms lines are deliberately out-of-order by id.
    return """# test file
5 atoms
4 atom types

0       2.504   xlo xhi
0       2.168527611     ylo yhi
0       6.6610  zlo zhi
1.252   0       0       xy xz yz

Masses

1      10.811 # BZX
2      140.0067 # ASDF
3      14.0067 # N3
4      14.0067 # NOTYPE

Atoms # full

3       1       4       0.0       0       0       3.3305
1       1       1       0.0       0       0       0
2       1       2       0.0       1.252   0.722842537     0
4       1       1       0.0       1.252   0.722842537     3.3305
5       1       1       0.0       1.252   0.722842537     0
"""


def _lammps_data_without_tags() -> str:
    # No '#' comments in Masses lines -> tags should become '_' in stored table.
    return """# test file (no tags)
3 atoms
2 atom types

0  2.0  xlo xhi
0  2.0  ylo yhi
0  2.0  zlo zhi

Masses

1  10.811
2  14.0067

Atoms # full

2  1  2  0.0  0.0  0.0  0.0
1  1  1  0.0  1.0  0.0  0.0
3  1  2  0.0  0.0  1.0  0.0
"""


def _header_atom_types(text: str) -> int:
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)\s+atom\s+types\s*$", line)
        if m:
            return int(m.group(1))
    raise AssertionError("Did not find 'atom types' header line")


def _has_section(text: str, name: str) -> bool:
    pat = re.compile(rf"^\s*{re.escape(name)}\s*$", re.M)
    return pat.search(text) is not None


def _parse_masses_section(text: str) -> Dict[int, Tuple[float, str]]:
    """Return {type: (mass, tag)} from a LAMMPS data file string."""
    lines = text.splitlines()
    out: Dict[int, Tuple[float, str]] = {}

    # Find section
    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.strip() == "Masses")
    except StopIteration:
        return out

    # Skip blank lines after "Masses"
    i = i0 + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    # Read until next blank line or next section header
    while i < len(lines):
        ln = lines[i].strip()
        if ln == "":
            break
        if re.match(r"^[A-Za-z]", ln):  # next section header
            break
        # Example: "1  10.811 # BZX"
        if "#" in ln:
            left, tag = ln.split("#", 1)
            tag = tag.strip()
        else:
            left, tag = ln, ""
        fields = left.split()
        t = int(fields[0])
        mass = float(fields[1])
        out[t] = (mass, tag)
        i += 1

    return out


def _parse_atoms_types_full(text: str) -> List[int]:
    """Return list of type ints from 'Atoms # full' section, in file order."""
    lines = text.splitlines()

    # Find Atoms section header line (Atoms, Atoms # full, etc.)
    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.strip().startswith("Atoms"))
    except StopIteration:
        raise AssertionError("Did not find 'Atoms' section")

    i = i0 + 1
    while i < len(lines) and lines[i].strip() == "":
        i += 1

    types: List[int] = []
    while i < len(lines):
        ln = lines[i].strip()
        if ln == "":
            break
        if re.match(r"^[A-Za-z]", ln):  # next section header
            break
        fields = ln.split()
        # full: id mol-id type q x y z ...
        types.append(int(fields[2]))
        i += 1
    return types


def test_read_stores_type_table_in_info_and_preserves_types_sorted_by_id():
    atoms = read_lammps_data(io.StringIO(_lammps_data_with_tags_and_unused_type()))

    # default sort_by_id=True => ids should become 1..N in order
    assert np.array_equal(atoms.arrays["id"], np.arange(1, 6))

    # types in id order: id1->1, id2->2, id3->4, id4->1, id5->1
    assert np.array_equal(atoms.arrays["type"], np.array([1, 2, 4, 1, 1]))

    table = get_lmp_type_table(atoms)
    assert table is not None
    assert table["n_types"] == 4

    # tags preserved for all declared types (including unused type 3)
    assert table["tag"][1] == "BZX"
    assert table["tag"][2] == "ASDF"
    assert table["tag"][3] == "N3"
    assert table["tag"][4] == "NOTYPE"

    # masses present for all declared types (type 3 and 4 share mass on purpose)
    assert pytest.approx(table["mass"][1], rel=1e-12, abs=0.0) == table["mass"][1]
    assert table["mass"][2] > 0.0
    assert table["mass"][3] > 0.0
    assert table["mass"][4] > 0.0


def test_read_sort_by_id_false_preserves_file_atom_order():
    atoms = read_lammps_data(
        io.StringIO(_lammps_data_with_tags_and_unused_type()),
        sort_by_id=False,
    )
    # In-file order is ids: 3,1,2,4,5
    assert np.array_equal(atoms.arrays["id"], np.array([3, 1, 2, 4, 5]))
    # Corresponding types: 4,1,2,1,1
    assert np.array_equal(atoms.arrays["type"], np.array([4, 1, 2, 1, 1]))


def test_write_preserve_atom_types_true_writes_full_type_table_and_keeps_types():
    atoms = read_lammps_data(io.StringIO(_lammps_data_with_tags_and_unused_type()))

    buf = io.StringIO()
    write_lammps_data(
        buf,
        atoms,
        atom_style="full",
        preserve_atom_types=True,
        masses=False,  # should auto-write Masses from info table
        units="metal",
    )
    out = buf.getvalue()

    assert _header_atom_types(out) == 4
    assert _has_section(out, "Masses")

    masses = _parse_masses_section(out)
    assert set(masses.keys()) == {1, 2, 3, 4}
    assert masses[1][1] == "BZX"
    assert masses[2][1] == "ASDF"
    assert masses[3][1] == "N3"
    assert masses[4][1] == "NOTYPE"
    # type 3 must not be 0.0 in this case (it exists in the stored table)
    assert masses[3][0] > 0.0

    # Types in Atoms section should follow atoms.arrays['type'] order
    types_out = _parse_atoms_types_full(out)
    assert types_out == [1, 2, 4, 1, 1]


def test_write_preserve_atom_types_false_does_not_use_type_table_by_default():
    atoms = read_lammps_data(io.StringIO(_lammps_data_with_tags_and_unused_type()))

    buf = io.StringIO()
    write_lammps_data(
        buf,
        atoms,
        atom_style="full",
        preserve_atom_types=False,
        masses=False,  # should NOT auto-write Masses
        units="metal",
    )
    out = buf.getvalue()

    # Default ASE behavior: type count is number of unique symbols in the Atoms object.
    # With masses ~10.8 (B), ~14 (N), ~140 (likely Ce), that should be 3.
    assert _header_atom_types(out) == 3

    # Masses section should not appear unless masses=True
    assert not _has_section(out, "Masses")


def test_read_without_mass_tags_stores_underscore_tags_and_roundtrips():
    atoms = read_lammps_data(io.StringIO(_lammps_data_without_tags()))

    table = get_lmp_type_table(atoms)
    assert table is not None
    assert table["n_types"] == 2
    assert table["tag"][1] == "_"
    assert table["tag"][2] == "_"

    buf = io.StringIO()
    write_lammps_data(
        buf,
        atoms,
        atom_style="full",
        preserve_atom_types=True,
        masses=False,  # auto-write from table
        units="metal",
    )
    out = buf.getvalue()
    masses = _parse_masses_section(out)
    assert masses[1][1] == "_"
    assert masses[2][1] == "_"


def test_preserve_true_but_missing_info_table_falls_back_safely(tmp_path):
    atoms = read_lammps_data(io.StringIO(_lammps_data_with_tags_and_unused_type()))
    # Simulate metadata loss across a format that doesn't preserve atoms.info
    atoms.info.pop("lmp_type_table", None)

    buf = io.StringIO()
    write_lammps_data(
        buf,
        atoms,
        atom_style="full",
        preserve_atom_types=True,
        masses=False,  # no table => no Masses auto-write
        units="metal",
    )
    out = buf.getvalue()

    # Still preserves types in Atoms section
    types_out = _parse_atoms_types_full(out)
    assert types_out == [1, 2, 4, 1, 1]

    # Header n_atom_types falls back to max(type)
    assert _header_atom_types(out) == 4

    # And since masses=False and no table, no Masses section should be written
    assert not _has_section(out, "Masses")


def test_extxyz_roundtrip_preserves_type_table_and_types(tmp_path):
    # Requires ASE extxyz I/O
    from ase.io import read, write

    atoms = read_lammps_data(io.StringIO(_lammps_data_with_tags_and_unused_type()))

    p = tmp_path / "rt.xyz"
    write(p, atoms, format="extxyz")
    atoms2 = read(p, format="extxyz")

    assert "type" in atoms2.arrays
    assert np.array_equal(atoms2.arrays["type"], atoms.arrays["type"])

    table2 = get_lmp_type_table(atoms2)
    assert table2 is not None
    assert table2["n_types"] == 4
    assert table2["tag"][1] == "BZX"

    buf = io.StringIO()
    write_lammps_data(
        buf,
        atoms2,
        atom_style="full",
        preserve_atom_types=True,
        masses=False,
        units="metal",
    )
    out = buf.getvalue()
    assert _header_atom_types(out) == 4
    assert _has_section(out, "Masses")

