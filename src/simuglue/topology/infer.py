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
) -> tuple[Topo, List[List[int]], Optional[List[float]]]:
    """
    Infer bonds using a distance window criterion: r in (rc-drc, rc+drc) for any rc in rc_list.

    Parameters
    ----------
    atoms
        ASE Atoms with positions and cell.
    rc_list
        List of reference distances (cutoff centers). A pair (i,j) is bonded if its
        minimum-image distance falls within any of these windows.
    drc
        Half-width of the distance window around each rc.
    deduplicate
        If True, ensure each pair (i,j) appears at most once even if multiple rc windows match.
    return_lengths
        If True, also return a list of bond lengths aligned with topo.bonds.

    Returns
    -------
    topo
        Topo instance with topo.bonds populated (0-based ASE indices).
    neighbors
        Adjacency list: neighbors[i] contains all bonded neighbors of atom i.
    lengths
        List of bond lengths (same order as topo.bonds) if return_lengths=True, else None.

    Notes
    -----
    - This is an O(N^2) implementation (like your current CrystalBuilder version).
    - Later, you can replace the pair loop with an ASE NeighborList-based implementation
      without changing the returned data structures (Topo + neighbors).
    """
    natoms = len(atoms)
    if natoms == 0:
        topo = Topo()
        return topo, [], [] if return_lengths else None

    if not rc_list:
        topo = Topo()
        neighbors = [[] for _ in range(natoms)]
        return topo, neighbors, [] if return_lengths else None

    if drc < 0:
        raise ValueError(f"drc must be non-negative, got {drc}")

    cell = atoms.cell

    # If any periodic dimension is True, ensure a valid cell exists
    if any(atoms.pbc) and cell is None:
        raise ValueError("Periodic bonding requested but atoms.cell is None")

    pos = atoms.positions
    ranges2 = [((rc - drc) ** 2, (rc + drc) ** 2) for rc in rc_list]

    bonds: List[Tuple[int, int]] = []
    lengths: List[float] = []
    neighbors: List[List[int]] = [[] for _ in range(natoms)]
    seen: set[Tuple[int, int]] = set()

    for i in range(natoms - 1):
        ri = pos[i]
        for j in range(i + 1, natoms):
            rij = pos[j] - ri

            # Minimum-image displacement for triclinic/orthorhombic cells
            rij_mic, _ = find_mic(rij, cell, pbc=atoms.pbc)
            rij2 = float(np.dot(rij_mic, rij_mic))

            # Check against all windows; stop at first match
            matched = False
            for rmin2, rmax2 in ranges2:
                if rmin2 < rij2 < rmax2:
                    matched = True
                    break
            if not matched:
                continue

            key = (i, j)
            if deduplicate and key in seen:
                continue
            seen.add(key)

            bonds.append(key)
            neighbors[i].append(j)
            neighbors[j].append(i)
            if return_lengths:
                lengths.append(float(np.sqrt(rij2)))

    topo = Topo(bonds=bonds)
    # Optional: canonicalize and sort neighbors for determinism
    topo.canonicalize_bonds(deduplicate=deduplicate, sort=True)
    # Rebuild neighbors if you canonicalized/sorted bonds and want strict alignment:
    neighbors = topo.build_adjacency(natoms, sort_neighbors=True, unique_neighbors=True)

    return topo, neighbors, lengths if return_lengths else None

