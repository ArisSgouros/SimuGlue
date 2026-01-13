from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic


def bond_lengths(
    atoms: Atoms,
    bonds: Sequence[Tuple[int, int]],
) -> List[float]:
    """Compute minimum-image bond lengths aligned with `bonds`."""
    if not bonds:
        return []
    cell = atoms.cell
    pos = atoms.positions
    out: List[float] = []
    for i, j in bonds:
        rij = pos[j] - pos[i]
        rij_mic, _ = find_mic(rij, cell, pbc=atoms.pbc)
        out.append(float(np.linalg.norm(rij_mic)))
    return out


def angle_thetas_deg(
    atoms: Atoms,
    angles: Sequence[Tuple[int, int, int]],
) -> List[float]:
    """Compute angle values (degrees) aligned with `angles` (i, j, k), j is central."""
    if not angles:
        return []
    cell = atoms.cell
    pos = atoms.positions

    out: List[float] = []
    for i, j, k in angles:
        rji = pos[i] - pos[j]
        rjk = pos[k] - pos[j]
        rji, _ = find_mic(rji, cell, pbc=atoms.pbc)
        rjk, _ = find_mic(rjk, cell, pbc=atoms.pbc)
        # normalize
        n1 = np.linalg.norm(rji)
        n2 = np.linalg.norm(rjk)
        if n1 == 0.0 or n2 == 0.0:
            out.append(float("nan"))
            continue
        u1 = rji / n1
        u2 = rjk / n2
        costheta = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
        theta = math.degrees(math.acos(costheta))
        out.append(float(theta))
    return out


def angle_symmetry_labels(
    atoms: Atoms,
    angles: Sequence[Tuple[int, int, int]],
    tol: float = 1e-5,
) -> List[str]:
    """Classify angles using the i<->k relative vector.

    Labels (kept compatible with your current logic):
      - 'T': coplanar ends (delta_z ~ 0)
      - 'N': vertical stacking (delta_x ~ 0 and delta_y ~ 0)
      - 'A': other

    Uses minimum-image displacement for robustness under PBC.
    """
    if not angles:
        return []
    cell = atoms.cell
    pos = atoms.positions

    out: List[str] = []
    for i, _j, k in angles:
        dik = pos[k] - pos[i]
        dik, _ = find_mic(dik, cell, pbc=atoms.pbc)
        if abs(float(dik[2])) < tol:
            out.append("T")
        elif abs(float(dik[0])) < tol and abs(float(dik[1])) < tol:
            out.append("N")
        else:
            out.append("A")
    return out


def _phi_dihedral_praxeolitic(
    r0: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
    r3: np.ndarray,
    cell,
    pbc,
) -> float:
    """Praxeolitic formula for torsion angle (radians)."""
    b0 = -1.0 * (r1 - r0)
    b1 = r2 - r1
    b2 = r3 - r2

    b0, _ = find_mic(b0, cell, pbc=pbc)
    b1, _ = find_mic(b1, cell, pbc=pbc)
    b2, _ = find_mic(b2, cell, pbc=pbc)

    n1 = np.linalg.norm(b1)
    if n1 == 0.0:
        return float("nan")
    b1 = b1 / n1
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = float(np.dot(v, w))
    y = float(np.dot(np.cross(b1, v), w))
    return float(np.arctan2(y, x))


def dihedral_phis_rad(
    atoms: Atoms,
    dihedrals: Sequence[Tuple[int, int, int, int]],
) -> List[float]:
    """Compute dihedral torsion angles (radians) aligned with `dihedrals`."""
    if not dihedrals:
        return []
    cell = atoms.cell
    pos = atoms.positions

    out: List[float] = []
    for i, j, k, l in dihedrals:
        out.append(_phi_dihedral_praxeolitic(pos[i], pos[j], pos[k], pos[l], cell, atoms.pbc))
    return out


def dihedral_cis_trans(
    phi_rad: Sequence[float],
    *,
    threshold: float = 0.5 * math.pi,
) -> List[str]:
    """Classify dihedrals as 'cis'/'trans' using |phi| > threshold => trans."""
    out: List[str] = []
    for phi in phi_rad:
        if not math.isfinite(phi):
            out.append("unknown")
        elif abs(phi) > threshold:
            out.append("trans")
        else:
            out.append("cis")
    return out
