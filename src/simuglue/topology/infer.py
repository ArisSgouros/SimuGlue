from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

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

    pos, cell, pbc = atoms.positions, atoms.cell, atoms.pbc
    ranges2 = [((rc - drc) ** 2, (rc + drc) ** 2) for rc in rc_list]

    bonds, seen = [], set()
    for i in range(n - 1):
        ri = pos[i]
        for j in range(i + 1, n):
            rij, _ = find_mic(pos[j] - ri, cell, pbc=pbc)
            d2 = float(np.dot(rij, rij))
            if not any(rmin2 < d2 < rmax2 for rmin2, rmax2 in ranges2):
                continue
            key = (i, j)  # canonical
            if deduplicate and key in seen:
                continue
            seen.add(key)
            bonds.append(key)

    topo = Topo(bonds=bonds)
    topo.canonicalize_bonds(deduplicate=deduplicate, sort=True)
    neighbors = topo.build_adjacency(n, sort_neighbors=True, unique_neighbors=True)

    lengths = None
    if return_lengths:
        lengths = []
        for i, j in topo.bonds:
            rij, _ = find_mic(pos[j] - pos[i], cell, pbc=pbc)
            lengths.append(float(np.linalg.norm(rij)))

    return topo, neighbors, lengths

