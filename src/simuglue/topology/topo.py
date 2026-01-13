#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


Index = int


@dataclass(slots=True)
class Topo:
    """
    Topology container referencing ASE atom indices (0-based).

    Connectivity:
      - bonds:     (i, j)
      - angles:    (i, j, k)   where j is the central atom
      - dihedrals: (i, j, k, l) (LAMMPS-style ordering)

    Optional type IDs:
      - bond_types, angle_types, dihedral_types are LAMMPS-style positive integers (>=1)

    meta:
      - free-form metadata, typically used for type tables, e.g.
          meta["bond_type_table"][type_id] = {"types": (...), "label": "..."}
          meta["angle_type_table"][type_id] = {"types": (...), "sym": "...", "label": "..."}
          meta["dihedral_type_table"][type_id] = {"types": (...), "orient": "...", "label": "..."}
    """

    bonds: List[Tuple[Index, Index]] = field(default_factory=list)
    angles: List[Tuple[Index, Index, Index]] = field(default_factory=list)
    dihedrals: List[Tuple[Index, Index, Index, Index]] = field(default_factory=list)

    # Optional per-term type IDs (LAMMPS-style positive integers, 1-based)
    bond_types: Optional[List[int]] = None
    angle_types: Optional[List[int]] = None
    dihedral_types: Optional[List[int]] = None

    # Optional per-term tags (debug labels / rule keys). Same order/length as the corresponding list.
    bond_tags: Optional[List[str]] = None
    angle_tags: Optional[List[str]] = None
    dihedral_tags: Optional[List[str]] = None

    meta: Dict[str, object] = field(default_factory=dict)

    # ----------------------------
    # Validation
    # ----------------------------
    def validate(self, natoms: int, *, strict_types: bool = True) -> None:
        """
        Validate index ranges and type array lengths.
        Raises ValueError on any issue.
        """
        # Allow natoms == 0 only if the topology is empty.
        if natoms <= 0:
            if (not self.bonds) and (not self.angles) and (not self.dihedrals):
                return
            raise ValueError("natoms must be > 0 for non-empty topology")

        def _check_arity_and_range(items: Sequence[tuple], arity: int, name: str) -> None:
            for n, item in enumerate(items):
                if len(item) != arity:
                    raise ValueError(f"{name}[{n}] has arity {len(item)} (expected {arity}): {item}")
                for a in item:
                    if not isinstance(a, int):
                        raise ValueError(f"{name}[{n}] contains non-int index: {item}")
                    if not (0 <= a < natoms):
                        raise ValueError(f"{name}[{n}] index out of range [0,{natoms-1}]: {item}")

        _check_arity_and_range(self.bonds, 2, "bonds")
        _check_arity_and_range(self.angles, 3, "angles")
        _check_arity_and_range(self.dihedrals, 4, "dihedrals")

        def _check_types(types: Optional[List[int]], items: Sequence[tuple], name: str) -> None:
            if types is None:
                return
            if len(types) != len(items):
                raise ValueError(f"{name} length mismatch: {len(types)} types vs {len(items)} items")
            if strict_types:
                for t in types:
                    if not isinstance(t, int) or t < 1:
                        raise ValueError(f"{name} must contain positive ints (>=1), got {t}")

        _check_types(self.bond_types, self.bonds, "bond_types")
        _check_types(self.angle_types, self.angles, "angle_types")
        _check_types(self.dihedral_types, self.dihedrals, "dihedral_types")

        def _check_tags(tags: Optional[List[str]], items: Sequence[tuple], name: str) -> None:
            if tags is None:
                return
            if len(tags) != len(items):
                raise ValueError(f"{name} length mismatch: {len(tags)} tags vs {len(items)} items")

        _check_tags(self.bond_tags, self.bonds, "bond_tags")
        _check_tags(self.angle_tags, self.angles, "angle_tags")
        _check_tags(self.dihedral_tags, self.dihedrals, "dihedral_tags")


    # ----------------------------
    # Graph / neighbors
    # ----------------------------
    def build_adjacency(
        self,
        natoms: int,
        *,
        sort_neighbors: bool = True,
        unique_neighbors: bool = True,
    ) -> List[List[Index]]:
        """
        Build neighbor adjacency list from bonds:
          neighbors[i] = [j1, j2, ...]
        """
        neigh: List[List[Index]] = [[] for _ in range(natoms)]
        for i, j in self.bonds:
            neigh[i].append(j)
            neigh[j].append(i)

        if unique_neighbors:
            # preserve determinism
            neigh = [list(dict.fromkeys(lst)) for lst in neigh]  # stable unique
        if sort_neighbors:
            for lst in neigh:
                lst.sort()
        return neigh


    # ----------------------------
    # Canonicalization / deduplication
    # ----------------------------
    def canonicalize_bonds(self, *, deduplicate: bool = True, sort: bool = True) -> None:
        """
        Canonicalize bonds so each bond is (min(i,j), max(i,j)).
        Optionally deduplicate and sort. Operates in-place.

        If bond_types exist and duplicates are removed, the first occurrence wins.
        """
        bonds = [(min(i, j), max(i, j)) for (i, j) in self.bonds]

        if self.bond_types is None and self.bond_tags is None:
            if deduplicate:
                bonds = list(dict.fromkeys(bonds))  # stable unique
            if sort:
                bonds.sort()
            self.bonds = bonds
            return

        # With types and/or tags: keep first occurrence for each bond key
        bt = self.bond_types
        btag = self.bond_tags
        seen: Dict[Tuple[int, int], int] = {}
        new_bonds: List[Tuple[int, int]] = []
        new_bt: Optional[List[int]] = [] if bt is not None else None
        new_btag: Optional[List[str]] = [] if btag is not None else None
        for idx, b in enumerate(bonds):
            if deduplicate and b in seen:
                continue
            seen[b] = 1
            new_bonds.append(b)
            if new_bt is not None:
                new_bt.append(bt[idx])  # type: ignore[index]
            if new_btag is not None:
                new_btag.append(btag[idx])  # type: ignore[index]

        if sort:
            order = sorted(range(len(new_bonds)), key=lambda k: new_bonds[k])
            new_bonds = [new_bonds[i] for i in order]
            if new_bt is not None:
                new_bt = [new_bt[i] for i in order]
            if new_btag is not None:
                new_btag = [new_btag[i] for i in order]

        self.bonds = new_bonds
        self.bond_types = new_bt
        self.bond_tags = new_btag

