# tests/test_lammps_topo_roundtrip.py
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


LAMMPS_DATA_WITH_TOPO = """\
(written by ASE)

10 atoms
2 atom types

5 bonds
2 bond types

5 angles
2 angle types

5 dihedrals
3 dihedral types

0.0                  10.016  xlo xhi
0.0      8.6741104440000001  ylo yhi
0.0      13.321999999999999  zlo zhi
                  5.008                       0                       0  xy xz yz

Masses

     1      10.810999997218932 # B
     2      14.006699996396856 # N

Atoms # full

     1   1   1   1.0                       0                       0                       0
     2   1   2   1.0                   1.252     0.72284253999999992                       0
     3   1   2   1.0                       0                       0      3.3304999999999998
     4   1   1   1.0                   1.252     0.72284253999999992      3.3304999999999998
     5   1   1   1.0                   2.504                       0                       0
     6   1   2   1.0      3.7559999999999998     0.72284253999999992                       0
     7   1   2   1.0                   2.504                       0      3.3304999999999998
     8   1   1   1.0      3.7559999999999998     0.72284253999999992      3.3304999999999998
     9   1   1   1.0                   5.008                       0                       0
    10   1   2   1.0      6.2599999999999998     0.72284253999999992                       0

Bonds

  1   1   1   2
  2   2   2   5
  3   1   3   4
  4   1   4   7
  5   1   5   6

Angles

  1   2   1   2   5
  2   1   3   4   7
  3   1   2   5   6
  4   1   5   6   9
  5   1   4   7   8

Dihedrals

  1   3   1   2   5   6
  2   2   2   5   6   9
  3   1   3   4   7   8
  4   1   4   7   8   7
  5   1   4   7   8  10
"""


def _run_sgl(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """
    Prefer the installed `sgl` entrypoint; otherwise try running as a module.
    If neither works, fail with a helpful message.
    """
    sgl = shutil.which("sgl")
    if sgl:
        cmd = [sgl, *args]
    else:
        # Fallback: try `python -m simuglue` (adjust if your module entrypoint differs)
        cmd = [sys.executable, "-m", "simuglue", *args]

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    return proc


_SECTION_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 #/]*)\s*$")


def _parse_lammps_sections(text: str) -> dict[str, list[str]]:
    """
    Minimal section splitter: returns {section_name: [raw_lines_in_section]} for
    sections like Masses, Atoms, Bonds, Angles, Dihedrals.
    """
    sections: dict[str, list[str]] = {}
    cur = None

    for line in text.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            name = m.group(1).strip()
            # treat "Atoms # full" as "Atoms"
            if name.startswith("Atoms"):
                name = "Atoms"
            if name in {"Masses", "Atoms", "Bonds", "Angles", "Dihedrals"}:
                cur = name
                sections.setdefault(cur, [])
                continue

        if cur is not None:
            # stop at next blank line? keep blanks but filter later
            sections[cur].append(line.rstrip("\n"))

    return sections


def _extract_topo(sections: dict[str, list[str]]) -> dict[str, list[tuple[int, ...]]]:
    """
    Extract topology as tuples of ints (excluding the leading index).
    Bonds:     (type, i, j)
    Angles:    (type, i, j, k)
    Dihedrals: (type, i, j, k, l)
    """
    out: dict[str, list[tuple[int, ...]]] = {}

    def parse_lines(lines: list[str], expected_cols: int) -> list[tuple[int, ...]]:
        rows: list[tuple[int, ...]] = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            # ignore comments (none here) and headers
            parts = ln.split()
            if len(parts) < expected_cols:
                continue
            ints = tuple(int(x) for x in parts[:expected_cols])
            # drop leading id column
            rows.append(ints[1:])
        return rows

    if "Bonds" in sections:
        out["Bonds"] = parse_lines(sections["Bonds"], expected_cols=4)
    if "Angles" in sections:
        out["Angles"] = parse_lines(sections["Angles"], expected_cols=5)
    if "Dihedrals" in sections:
        out["Dihedrals"] = parse_lines(sections["Dihedrals"], expected_cols=6)

    return out


def _extract_counts_header(text: str) -> dict[str, int]:
    """
    Extract counts from the header (e.g. '5 bonds', '5 angles', '5 dihedrals').
    """
    counts = {}
    for key in ("atoms", "bonds", "angles", "dihedrals", "atom types", "bond types", "angle types", "dihedral types"):
        m = re.search(rf"^\s*(\d+)\s+{re.escape(key)}\s*$", text, flags=re.MULTILINE)
        if m:
            counts[key] = int(m.group(1))
    return counts


@pytest.mark.parametrize("iformat,oformat", [("lammps-data", "lammps-data")])
def test_aseconv_roundtrip_preserves_topology(tmp_path: Path, iformat: str, oformat: str) -> None:
    """
    Round-trip:
        i.pos.dat (lammps-data with topology)
            -> o.pos.xyz
            -> o.pos.cycle.dat (lammps-data with topology)
    and check that Bonds/Angles/Dihedrals are preserved (as sets of tuples).
    """
    in_dat = tmp_path / "i.pos.dat"
    out_xyz = tmp_path / "o.pos.xyz"
    out_dat = tmp_path / "o.pos.cycle.dat"

    in_dat.write_text(LAMMPS_DATA_WITH_TOPO, encoding="utf-8")

    # 1) dat -> xyz
    p1 = _run_sgl(
        ["io", "aseconv", "-i", str(in_dat), "--iformat", iformat, "-o", str(out_xyz), "--overwrite"],
        cwd=tmp_path,
    )
    assert p1.returncode == 0, f"aseconv dat->xyz failed\nSTDOUT:\n{p1.stdout}\nSTDERR:\n{p1.stderr}"
    assert out_xyz.exists() and out_xyz.stat().st_size > 0

    # 2) xyz -> dat
    p2 = _run_sgl(
        ["io", "aseconv", "-i", str(out_xyz), "--oformat", oformat, "-o", str(out_dat), "--overwrite"],
        cwd=tmp_path,
    )
    assert p2.returncode == 0, f"aseconv xyz->dat failed\nSTDOUT:\n{p2.stdout}\nSTDERR:\n{p2.stderr}"
    assert out_dat.exists() and out_dat.stat().st_size > 0

    # Parse original and cycled
    orig_txt = in_dat.read_text(encoding="utf-8")
    cyc_txt = out_dat.read_text(encoding="utf-8")

    # Header counts should match
    orig_counts = _extract_counts_header(orig_txt)
    cyc_counts = _extract_counts_header(cyc_txt)

    for k in ("atoms", "bonds", "angles", "dihedrals", "atom types", "bond types", "angle types", "dihedral types"):
        assert orig_counts.get(k) == cyc_counts.get(k), f"Header count mismatch for '{k}': {orig_counts.get(k)} vs {cyc_counts.get(k)}"

    # Topology content should match (order-independent)
    orig_secs = _parse_lammps_sections(orig_txt)
    cyc_secs = _parse_lammps_sections(cyc_txt)

    orig_topo = _extract_topo(orig_secs)
    cyc_topo = _extract_topo(cyc_secs)

    for sec in ("Bonds", "Angles", "Dihedrals"):
        assert sec in orig_topo, f"Missing {sec} in original parse"
        assert sec in cyc_topo, f"Missing {sec} in cycled parse"
        assert sorted(orig_topo[sec]) == sorted(cyc_topo[sec]), (
            f"{sec} mismatch.\n"
            f"Only in original: {sorted(set(orig_topo[sec]) - set(cyc_topo[sec]))[:10]}\n"
            f"Only in cycled:   {sorted(set(cyc_topo[sec]) - set(orig_topo[sec]))[:10]}"
        )

