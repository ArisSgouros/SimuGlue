# simuglue/topology/core.py
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from ase import Atoms

from simuglue.topology.infer import (
    infer_bonds_by_distance,
    infer_angles_from_adjacency,
    infer_dihedrals_from_bonds,
)
from simuglue.topology.topo import Topo
from simuglue.topology.typing import TypingOptions, type_bonds, type_angles, type_dihedrals


def build_topology_from_atoms(
    atoms: Atoms,
    *,
    calc_bonds: bool,
    calc_angles: bool,
    calc_diheds: bool,
    rc_list: Sequence[float],
    drc: float,
    deduplicate: bool = True,
) -> tuple[Topo, List[List[int]]]:
    """
    Preserve current CLI behavior:
      - bonds inferred only if calc_bonds
      - angles inferred from adjacency only if calc_angles
      - dihedrals inferred from bonds+adjacency only if calc_diheds
      - returns (topo, neighbors)
    """
    n = len(atoms)
    neighbors: List[List[int]] = [[] for _ in range(n)]

    if not calc_bonds:
        topo = Topo(bonds=[])
        return topo, neighbors

    topo_b, neighbors, _lens = infer_bonds_by_distance(
        atoms,
        rc_list=rc_list,
        drc=float(drc),
        deduplicate=deduplicate,
        return_lengths=False,
    )
    topo = topo_b

    if calc_angles:
        topo.angles = infer_angles_from_adjacency(neighbors, sort=True)

    if calc_diheds:
        topo.dihedrals = infer_dihedrals_from_bonds(topo.bonds, neighbors, sort=True)

    return topo, neighbors


def ensure_atom_tags_from_lmp_type_table(lmp_type_table: dict) -> dict:
    atom_tag_in = lmp_type_table.get('tag', None)
    if atom_tag_in is None:
        raise ValueError("lmp_type_table has no 'tag' field")

    atom_tag = dict(atom_tag_in)
    for type_id, tag in atom_tag.items():
        if tag == '_':
            atom_tag[type_id] = str(type_id)
    return atom_tag


def type_topology_inplace(
    atoms: Atoms,
    topo: Topo,
    *,
    atom_tag: dict,
    opts: TypingOptions,
    calc_bonds: bool,
    calc_angles: bool,
    calc_diheds: bool,
) -> None:
    """
    Populate topo.{bond,angle,dihedral}_types and meta tables (in-place).
    """
    if calc_bonds:
        type_bonds(atoms, topo, atom_tag, opts=opts)
    if calc_angles:
        type_angles(atoms, topo, atom_tag, opts=opts)
    if calc_diheds:
        type_dihedrals(atoms, topo, atom_tag, opts=opts)

    topo.validate(len(atoms), strict_types=False)


def _require_types_for_terms(
    *,
    kind: str,
    terms: Sequence[Sequence[int]],
    types: Optional[Sequence[int]],
) -> List[int]:
    """
    Fail-fast validation for attach_topology_arrays_to_atoms():
    if terms exist, corresponding types must exist and match length.
    """
    if not terms:
        return [] if types is None else list(types)
    # Treat None or empty as "missing" when terms exist.
    if types is None or len(types) == 0:
        raise ValueError(f"missing {kind}_types")
    if len(types) != len(terms):
        raise ValueError(
            f"mismatched {kind}_types (got {len(types)} for {len(terms)} {kind}s)"
        )
    return list(types)


def attach_topology_arrays_to_atoms(atoms: Atoms, topo: Topo) -> None:
    """
    Preserve current on-Atoms encoding exactly:
      - atoms.arrays['bonds']     per-atom string: "j(type),k(type),..." stored on at1 only
      - atoms.arrays['angles']    stored on central atom at2: "at1-at3(type),..."
      - atoms.arrays['dihedrals'] stored on at1: "at2-at3-at4(type),..."
    """
    natoms = len(atoms)

    bonds_in = topo.bonds
    angles_in = topo.angles
    dihedrals_in = topo.dihedrals

    bond_types = topo.bond_types or []
    angle_types = topo.angle_types or []
    dihedral_types = topo.dihedral_types or []

    bonds = [""] * natoms if len(bonds_in) > 0 else None
    angles = [""] * natoms if len(angles_in) > 0 else None
    dihedrals = [""] * natoms if len(dihedrals_in) > 0 else None

    if bonds is not None:
        bond_types = _require_types_for_terms(kind="bond", terms=bonds_in, types=topo.bond_types)
        for type_, (at1, at2) in zip(bond_types, bonds_in):
            if bonds[at1]:
                bonds[at1] += ","
            bonds[at1] += f"{at2:d}({type_:d})"
        for i, s in enumerate(bonds):
            if not s:
                bonds[i] = "_"
        atoms.arrays["bonds"] = np.array(bonds)

    if angles is not None:
        angle_types = _require_types_for_terms(kind="angle", terms=angles_in, types=topo.angle_types)
        for type_, (at1, at2, at3) in zip(angle_types, angles_in):
            if angles[at2]:
                angles[at2] += ","
            angles[at2] += f"{at1:d}-{at3:d}({type_:d})"
        for i, s in enumerate(angles):
            if not s:
                angles[i] = "_"
        atoms.arrays["angles"] = np.array(angles)

    if dihedrals is not None:
        dihedral_types = _require_types_for_terms(kind="dihedral", terms=dihedrals_in, types=topo.dihedral_types)
        for type_, (at1, at2, at3, at4) in zip(dihedral_types, dihedrals_in):
            if dihedrals[at1]:
                dihedrals[at1] += ","
            dihedrals[at1] += f"{at2:d}-{at3:d}-{at4:d}({type_:d})"
        for i, s in enumerate(dihedrals):
            if not s:
                dihedrals[i] = "_"
        atoms.arrays["dihedrals"] = np.array(dihedrals)

