from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.neighborlist import neighbor_list

from simuglue.topology.topo import Topo


def infer_angles_from_adjacency(
    neighbors: List[List[int]],
    *,
    sort: bool = True,
) -> List[Tuple[int, int, int]]:
    """Infer angles (i, j, k) from an adjacency list built from bonds.

    Returns unique angles where j is the central atom and i < k.
    """
    angles: List[Tuple[int, int, int]] = []
    natoms = len(neighbors)
    for j in range(natoms):
        nbrs = neighbors[j]
        ln = len(nbrs)
        if ln < 2:
            continue
        # combinations(nbrs, 2) without importing itertools
        for a in range(ln - 1):
            i = nbrs[a]
            for b in range(a + 1, ln):
                k = nbrs[b]
                if i == k:
                    continue
                if i < k:
                    angles.append((i, j, k))
                else:
                    angles.append((k, j, i))
    if sort:
        angles.sort()
    return angles


def infer_dihedrals_from_bonds(
    bonds: List[Tuple[int, int]],
    neighbors: List[List[int]],
    *,
    sort: bool = True,
) -> List[Tuple[int, int, int, int]]:
    """Infer dihedrals (i, j, k, l) from bonds + adjacency.

    Strategy:
      - Treat each bond (j,k) as the central bond, using only j < k to avoid duplicates.
      - For each neighbor i of j (excluding k) and each neighbor l of k (excluding j),
        create dihedral (i, j, k, l) if all 4 indices are distinct.

    The returned ordering matches LAMMPS (i j k l).
    """
    diheds: List[Tuple[int, int, int, int]] = []
    for j, k in bonds:
        if j == k:
            continue
        if j > k:
            j, k = k, j
        nj = neighbors[j]
        nk = neighbors[k]
        for i in nj:
            if i == k:
                continue
            for l in nk:
                if l == j:
                    continue
                # require all distinct
                if len({i, j, k, l}) != 4:
                    continue
                diheds.append((i, j, k, l))

    if sort:
        diheds.sort()
    return diheds


def infer_bonds_by_distance(
    atoms: Atoms,
    rc_list: Sequence[float],
    drc: float = 0.0,
    *,
    deduplicate: bool = True,
    return_lengths: bool = False,
):
    n = len(atoms)
    topo = Topo()
    neighbors = [[] for _ in range(n)]

    if n == 0 or not rc_list:
        return topo, neighbors, [] if return_lengths else None
    if drc < 0:
        raise ValueError(f"drc must be non-negative, got {drc}")

    pos = atoms.positions
    cell = np.asarray(atoms.cell)
    pbc = atoms.pbc

    EPS = 1e-12
    cutoff = float(max(rc_list) + drc + EPS)
    if cutoff <= 0.0:
        return topo, neighbors, [] if return_lengths else None

    # windows (clamp lower bound to 0)
    rmin2 = np.array([(max(0.0, rc - drc)) ** 2 for rc in rc_list], dtype=float)
    rmax2 = np.array([(rc + drc) ** 2 for rc in rc_list], dtype=float)

    i, j, S = neighbor_list("ijS", atoms, cutoff)  # candidate pairs + cell shifts

    # Vector from i to the image of j defined by shift S
    vec = pos[j] + (S @ cell) - pos[i]
    d2 = np.einsum("ij,ij->i", vec, vec)

    # keep candidates within ANY window
    mask = ((d2[:, None] >= (rmin2[None, :] - EPS)) & (d2[:, None] <= (rmax2[None, :] + EPS))).any(axis=1)

    ii, jj = i[mask], j[mask]
    a = np.minimum(ii, jj)
    b = np.maximum(ii, jj)
    pairs = np.stack([a, b], axis=1)

    if deduplicate and len(pairs):
        pairs = np.unique(pairs, axis=0)

    topo = Topo(bonds=[tuple(p) for p in pairs.tolist()])
    topo.canonicalize_bonds(deduplicate=deduplicate, sort=True)
    neighbors = topo.build_adjacency(n, sort_neighbors=True, unique_neighbors=True)

    lengths = None
    if return_lengths:
        lengths = []
        for i_, j_ in topo.bonds:
            rij, _ = find_mic(pos[j_] - pos[i_], cell, pbc=pbc)
            lengths.append(float(np.linalg.norm(rij)))

    return topo, neighbors, lengths

