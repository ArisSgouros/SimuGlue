#!/usr/bin/env python3
"""
compare_lammps_data.py (enhanced)

Compares two LAMMPS data files for equivalence, with support for comparing
bond/angle/dihedral "types" via inline comment tags after '#'.

Typical use for your case (type-ID order differs but tags match):
  python compare_lammps_data.py old.data new.data --compare-tags dihedrals --verbose

Other modes:
  python compare_lammps_data.py old.data new.data --ignore-term-types
  python compare_lammps_data.py old.data new.data --compare-tags all
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from collections import Counter, defaultdict, OrderedDict
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np


SECTION_NAMES = {
    "Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals", "Impropers",
    "Bond Coeffs", "Angle Coeffs", "Dihedral Coeffs", "Improper Coeffs",
    "Pair Coeffs",
}

CompareTagsMode = Literal["none", "dihedrals", "angles", "bonds", "all"]


def _is_section_header(line: str) -> Optional[str]:
    s = line.strip()
    if not s:
        return None
    head = s.split()[0]  # supports "Atoms # full"
    return head if head in SECTION_NAMES else None


def _split_comment(line: str) -> Tuple[str, Optional[str]]:
    """
    Split 'data # comment' into ('data', 'comment').
    Returns comment without leading '#'.
    """
    if "#" not in line:
        return line.strip(), None
    left, right = line.split("#", 1)
    left = left.strip()
    right = right.strip()
    return left, (right if right else None)


@dataclass
class AtomRec:
    aid: int
    atype: int
    x: float
    y: float
    z: float
    mol: int = 0
    q: float = 0.0


@dataclass
class LmpData:
    path: str

    # box info (stored but not strictly used for matching)
    xlo: float = 0.0
    xhi: float = 0.0
    ylo: float = 0.0
    yhi: float = 0.0
    zlo: float = 0.0
    zhi: float = 0.0
    xy: float = 0.0
    xz: float = 0.0
    yz: float = 0.0

    masses: Dict[int, float] = None
    type_labels: Dict[int, str] = None  # from Masses comments, e.g. 1->"B"
    atoms: Dict[int, AtomRec] = None

    # store tags per term entry as parsed from inline comments
    bonds: List[Tuple[int, int, int, Optional[str]]] = None        # (type, a, b, tag)
    angles: List[Tuple[int, int, int, int, Optional[str]]] = None  # (type, i, j, k, tag)
    dihedrals: List[Tuple[int, int, int, int, int, Optional[str]]] = None  # (type, i, j, k, l, tag)

    def __post_init__(self):
        self.masses = {} if self.masses is None else self.masses
        self.type_labels = {} if self.type_labels is None else self.type_labels
        self.atoms = {} if self.atoms is None else self.atoms
        self.bonds = [] if self.bonds is None else self.bonds
        self.angles = [] if self.angles is None else self.angles
        self.dihedrals = [] if self.dihedrals is None else self.dihedrals


def read_lammps_data(path: str) -> LmpData:
    data = LmpData(path=path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # parse header box bounds + tilt (best-effort)
    for ln in lines[:200]:
        s = ln.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) >= 4 and parts[-2:] == ["xlo", "xhi"]:
            data.xlo, data.xhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["ylo", "yhi"]:
            data.ylo, data.yhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 4 and parts[-2:] == ["zlo", "zhi"]:
            data.zlo, data.zhi = float(parts[0]), float(parts[1])
        elif len(parts) >= 6 and parts[-3:] == ["xy", "xz", "yz"]:
            data.xy, data.xz, data.yz = float(parts[0]), float(parts[1]), float(parts[2])

    # parse sections
    i = 0
    n = len(lines)
    while i < n:
        name = _is_section_header(lines[i])
        if name is None:
            i += 1
            continue

        i += 1
        while i < n and not lines[i].strip():
            i += 1

        block = []
        while i < n:
            hdr = _is_section_header(lines[i])
            if hdr is not None:
                break
            s = lines[i].strip()
            if s and not s.startswith("#"):
                block.append(s)
            i += 1

        if name == "Masses":
            for row in block:
                left, comment = _split_comment(row)
                parts = left.split()
                if len(parts) >= 2:
                    t = int(parts[0])
                    data.masses[t] = float(parts[1])
                    if comment:
                        # take first token as label, e.g. "# B"
                        data.type_labels[t] = comment.split()[0]

        elif name == "Atoms":
            for row in block:
                left, _comment = _split_comment(row)
                parts = left.split()
                if len(parts) < 5:
                    continue
                if len(parts) >= 7:
                    # full: id mol type q x y z
                    aid = int(parts[0]); mol = int(parts[1]); atype = int(parts[2])
                    q = float(parts[3]); x = float(parts[4]); y = float(parts[5]); z = float(parts[6])
                    data.atoms[aid] = AtomRec(aid=aid, mol=mol, atype=atype, q=q, x=x, y=y, z=z)
                else:
                    # atomic: id type x y z
                    aid = int(parts[0]); atype = int(parts[1])
                    x = float(parts[2]); y = float(parts[3]); z = float(parts[4])
                    data.atoms[aid] = AtomRec(aid=aid, mol=0, atype=atype, q=0.0, x=x, y=y, z=z)

        elif name == "Bonds":
            for row in block:
                left, comment = _split_comment(row)
                parts = left.split()
                if len(parts) >= 4:
                    btype = int(parts[1]); a = int(parts[2]); b = int(parts[3])
                    data.bonds.append((btype, a, b, comment))

        elif name == "Angles":
            for row in block:
                left, comment = _split_comment(row)
                parts = left.split()
                if len(parts) >= 5:
                    atype = int(parts[1]); a = int(parts[2]); b = int(parts[3]); c = int(parts[4])
                    data.angles.append((atype, a, b, c, comment))

        elif name == "Dihedrals":
            for row in block:
                left, comment = _split_comment(row)
                parts = left.split()
                if len(parts) >= 6:
                    dtype = int(parts[1]); a = int(parts[2]); b = int(parts[3]); c = int(parts[4]); d = int(parts[5])
                    data.dihedrals.append((dtype, a, b, c, d, comment))

    return data


def _try_import_ckdtree():
    try:
        from scipy.spatial import cKDTree  # type: ignore
        return cKDTree
    except Exception:
        return None


def build_atom_mapping(a: LmpData, b: LmpData, pos_tol: float) -> Dict[int, int]:
    """
    Map atom IDs from file a -> file b using grouped nearest-neighbor matching on (atype,mol).
    Requires positions to match within pos_tol inside each group.
    """
    ga = defaultdict(list)
    gb = defaultdict(list)
    for aid, ar in a.atoms.items():
        ga[(ar.atype, ar.mol)].append(aid)
    for bid, br in b.atoms.items():
        gb[(br.atype, br.mol)].append(bid)

    mapping: Dict[int, int] = {}
    cKDTree = _try_import_ckdtree()

    for key, a_ids in ga.items():
        if key not in gb:
            raise ValueError(f"Group {key} exists in A but not in B")
        b_ids = gb[key]
        if len(a_ids) != len(b_ids):
            raise ValueError(f"Group {key} size mismatch: A={len(a_ids)} B={len(b_ids)}")

        Apos = np.array([[a.atoms[i].x, a.atoms[i].y, a.atoms[i].z] for i in a_ids], dtype=float)
        Bpos = np.array([[b.atoms[i].x, b.atoms[i].y, b.atoms[i].z] for i in b_ids], dtype=float)

        used = np.zeros(len(b_ids), dtype=bool)

        if cKDTree is not None:
            tree = cKDTree(Bpos)
            dists, idxs = tree.query(Apos, k=1)
            order = np.argsort(dists)
            for oi in order:
                if dists[oi] > pos_tol:
                    raise ValueError(f"Atom position mismatch in group {key}, min dist={dists[oi]:.3e} > tol")
                j = int(idxs[oi])
                if used[j]:
                    neigh = tree.query_ball_point(Apos[oi], r=pos_tol)
                    chosen = None
                    for cand in neigh:
                        if not used[cand]:
                            chosen = cand
                            break
                    if chosen is None:
                        raise ValueError(f"Could not find unique match in group {key} within tol")
                    j = chosen
                used[j] = True
                mapping[a_ids[oi]] = b_ids[j]
        else:
            for ii, aid in enumerate(a_ids):
                best = None
                best_d = float("inf")
                for jj, bid in enumerate(b_ids):
                    if used[jj]:
                        continue
                    dx = Apos[ii, 0] - Bpos[jj, 0]
                    dy = Apos[ii, 1] - Bpos[jj, 1]
                    dz = Apos[ii, 2] - Bpos[jj, 2]
                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if d < best_d:
                        best_d = d
                        best = jj
                if best is None or best_d > pos_tol:
                    raise ValueError(f"Atom position mismatch in group {key}, min dist={best_d:.3e} > tol")
                used[best] = True
                mapping[aid] = b_ids[best]

    if len(mapping) != len(a.atoms):
        raise ValueError("Atom mapping incomplete")
    return mapping


def _canon_bond(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)

def _canon_angle(i: int, j: int, k: int) -> Tuple[int, int, int]:
    return (i, j, k) if i < k else (k, j, i)

def _canon_dihedral(i: int, j: int, k: int, l: int) -> Tuple[int, int, int, int]:
    fwd = (i, j, k, l)
    rev = (l, k, j, i)
    return fwd if fwd < rev else rev


def _normalize_tag(comment: Optional[str], type_labels: Dict[int, str], arity: int) -> Optional[str]:
    """
    Normalize inline tag strings to a canonical representation based on element labels.

    Examples:
      "1 2 1 2 trans" -> "B N B N trans"  (given Masses: 1->B, 2->N)
      "B N B N cis"   -> "B N B N cis"

    Returns None if no comment.
    """
    if not comment:
        return None

    tokens_raw = comment.strip().split()
    if not tokens_raw:
        return None

    # pull stereo token if present anywhere
    stereo = None
    keep = []
    for t in tokens_raw:
        tl = t.lower()
        if tl in ("cis", "trans"):
            stereo = tl
        else:
            keep.append(t)

    # support tokens like "B-N-B-N" or "1-2-1-2"
    exploded: List[str] = []
    for t in keep:
        if "-" in t and len(exploded) < arity:
            exploded.extend([x for x in t.split("-") if x])
        else:
            exploded.append(t)

    core = exploded[:arity]
    if len(core) < arity:
        # not enough info; still return something stable for comparison
        core = exploded

    # convert numeric atom-types -> labels using Masses comments
    norm = []
    for t in core:
        if t.isdigit():
            ti = int(t)
            norm.append(type_labels.get(ti, t))
        else:
            norm.append(t)

    out = " ".join(norm)
    if stereo:
        out = out + " " + stereo
    return out


def _compare_terms_with(
    A_items: List[Tuple],
    B_items: List[Tuple],
    verbose: bool,
    label: str,
) -> bool:
    ca, cb = Counter(A_items), Counter(B_items)
    if ca == cb:
        return True
    print(f"FAIL: {label} differ (A={sum(ca.values())} B={sum(cb.values())})")
    if verbose:
        only_a = list((ca - cb).items())[:10]
        only_b = list((cb - ca).items())[:10]
        if only_a:
            print("  Examples only in A:", only_a)
        if only_b:
            print("  Examples only in B:", only_b)
    return False


def compare(
    a: LmpData,
    b: LmpData,
    pos_tol: float,
    q_tol: float,
    ignore_term_types: bool,
    compare_tags: CompareTagsMode,
    verbose: bool,
) -> int:
    if len(a.atoms) != len(b.atoms):
        print(f"FAIL: natoms mismatch A={len(a.atoms)} B={len(b.atoms)}")
        return 2

    mapping = build_atom_mapping(a, b, pos_tol=pos_tol)

    # compare atom fields after mapping
    bad = 0
    for aid, ar in a.atoms.items():
        bid = mapping[aid]
        br = b.atoms[bid]
        if ar.atype != br.atype or ar.mol != br.mol:
            bad += 1
            if verbose and bad <= 10:
                print(f"Atom mismatch aid={aid} -> bid={bid}: (type,mol) A=({ar.atype},{ar.mol}) B=({br.atype},{br.mol})")
        if abs(ar.q - br.q) > q_tol:
            bad += 1
            if verbose and bad <= 10:
                print(f"Charge mismatch aid={aid} -> bid={bid}: A={ar.q} B={br.q}")
        dx = ar.x - br.x; dy = ar.y - br.y; dz = ar.z - br.z
        if math.sqrt(dx*dx + dy*dy + dz*dz) > pos_tol:
            bad += 1
            if verbose and bad <= 10:
                print(f"Pos mismatch aid={aid} -> bid={bid}: |dr|={math.sqrt(dx*dx+dy*dy+dz*dz):.3e}")

    if bad:
        print(f"FAIL: {bad} atom-field mismatches.")
        return 2

    use_tag_bonds = compare_tags in ("bonds", "all")
    use_tag_angles = compare_tags in ("angles", "all")
    use_tag_dihed = compare_tags in ("dihedrals", "all")

    # bonds
    Ab = []
    for (t, i, j, cmt) in a.bonds:
        ii = mapping[i]; jj = mapping[j]
        ii, jj = _canon_bond(ii, jj)
        if use_tag_bonds:
            tag = _normalize_tag(cmt, a.type_labels, arity=2)
            Ab.append((ii, jj, tag))
        elif ignore_term_types:
            Ab.append((ii, jj))
        else:
            Ab.append((t, ii, jj))

    Bb = []
    for (t, i, j, cmt) in b.bonds:
        ii, jj = _canon_bond(i, j)
        if use_tag_bonds:
            tag = _normalize_tag(cmt, b.type_labels, arity=2)
            Bb.append((ii, jj, tag))
        elif ignore_term_types:
            Bb.append((ii, jj))
        else:
            Bb.append((t, ii, jj))

    if not _compare_terms_with(Ab, Bb, verbose, "bonds"):
        return 2

    # angles
    Aa = []
    for (t, i, j, k, cmt) in a.angles:
        ii = mapping[i]; jj = mapping[j]; kk = mapping[k]
        ii, jj, kk = _canon_angle(ii, jj, kk)
        if use_tag_angles:
            tag = _normalize_tag(cmt, a.type_labels, arity=3)
            Aa.append((ii, jj, kk, tag))
        elif ignore_term_types:
            Aa.append((ii, jj, kk))
        else:
            Aa.append((t, ii, jj, kk))

    Ba = []
    for (t, i, j, k, cmt) in b.angles:
        ii, jj, kk = _canon_angle(i, j, k)
        if use_tag_angles:
            tag = _normalize_tag(cmt, b.type_labels, arity=3)
            Ba.append((ii, jj, kk, tag))
        elif ignore_term_types:
            Ba.append((ii, jj, kk))
        else:
            Ba.append((t, ii, jj, kk))

    if not _compare_terms_with(Aa, Ba, verbose, "angles"):
        return 2

    # dihedrals
    Ad = []
    for (t, i, j, k, l, cmt) in a.dihedrals:
        ii = mapping[i]; jj = mapping[j]; kk = mapping[k]; ll = mapping[l]
        ii, jj, kk, ll = _canon_dihedral(ii, jj, kk, ll)
        if use_tag_dihed:
            tag = _normalize_tag(cmt, a.type_labels, arity=4)
            Ad.append((ii, jj, kk, ll, tag))
        elif ignore_term_types:
            Ad.append((ii, jj, kk, ll))
        else:
            Ad.append((t, ii, jj, kk, ll))

    Bd = []
    for (t, i, j, k, l, cmt) in b.dihedrals:
        ii, jj, kk, ll = _canon_dihedral(i, j, k, l)
        if use_tag_dihed:
            tag = _normalize_tag(cmt, b.type_labels, arity=4)
            Bd.append((ii, jj, kk, ll, tag))
        elif ignore_term_types:
            Bd.append((ii, jj, kk, ll))
        else:
            Bd.append((t, ii, jj, kk, ll))

    if not _compare_terms_with(Ad, Bd, verbose, "dihedrals"):
        return 2

    print("OK: files are equivalent under the chosen comparison settings.")
    print(f"  atoms: {len(a.atoms)}")
    print(f"  bonds: {len(a.bonds)}")
    print(f"  angles: {len(a.angles)}")
    print(f"  dihedrals: {len(a.dihedrals)}")
    if compare_tags != "none":
        print(f"  note: compared {compare_tags} by normalized inline tags (using Masses labels when available).")
    elif ignore_term_types:
        print("  note: term type IDs were ignored (connectivity only).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a", help="LAMMPS data file A (e.g., CrystalBuilder output)")
    ap.add_argument("file_b", help="LAMMPS data file B (e.g., build lmp-topo output)")
    ap.add_argument("--pos-tol", type=float, default=1e-6, help="Position tolerance for atom matching and comparison.")
    ap.add_argument("--q-tol", type=float, default=1e-8, help="Charge tolerance.")
    ap.add_argument("--ignore-term-types", action="store_true",
                    help="Ignore bond/angle/dihedral type IDs; compare connectivity only.")
    ap.add_argument("--compare-tags", choices=("none", "bonds", "angles", "dihedrals", "all"),
                    default="none",
                    help="Compare selected terms using normalized inline tags after '#', instead of numeric type IDs.")
    ap.add_argument("--verbose", action="store_true", help="Print example differences on failure.")
    args = ap.parse_args()

    A = read_lammps_data(args.file_a)
    B = read_lammps_data(args.file_b)

    rc = compare(
        A, B,
        pos_tol=args.pos_tol,
        q_tol=args.q_tol,
        ignore_term_types=args.ignore_term_types,
        compare_tags=args.compare_tags,  # type: ignore[arg-type]
        verbose=args.verbose,
    )
    raise SystemExit(rc)


if __name__ == "__main__":
    main()

