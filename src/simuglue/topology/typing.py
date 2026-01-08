from __future__ import annotations

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ase import Atoms

from simuglue.topology.topo import Topo
from simuglue.topology import features as _feat
from itertools import permutations

def canonical_by_perms(chem_list, perms):
    t = tuple(map(str, chem_list))
    return list(min(tuple(t[i] for i in p) for p in perms))

BOND_PERMS    = [(0,1), (1,0)]
ANGLE_PERMS   = [(0,1,2), (2,1,0)]
DIHED_PERMS   = [(0,1,2,3), (3,2,1,0)]

def sort_bond(chem_list):    return canonical_by_perms(chem_list, BOND_PERMS)
def sort_angle(chem_list):   return canonical_by_perms(chem_list, ANGLE_PERMS)
def sort_dihedral(chem_list):return canonical_by_perms(chem_list, DIHED_PERMS)

def sort_improper(chem_list, center=1):
    # center fixed; permute outers
    idx = [0,1,2,3]
    outers = [i for i in idx if i != center]
    perms = []
    for p in permutations(outers):
        seq = []
        it = iter(p)
        for i in idx:
            seq.append(center if i == center else next(it))
        perms.append(tuple(seq))
    return canonical_by_perms(chem_list, perms)

def _get_atom_types(atoms: Atoms) -> List[int]:
    """Return LAMMPS-style integer atom types for each atom.

    Preference order:
      1) atoms.arrays['type']
      2) atoms.arrays['types']
      3) atoms.get_tags() if non-zero
      4) map chemical symbols to 1..K
    """
    if 'type' in atoms.arrays:
        return [int(x) for x in atoms.arrays['type']]
    if 'types' in atoms.arrays:
        return [int(x) for x in atoms.arrays['types']]

    tags = getattr(atoms, 'get_tags', lambda: None)()
    if tags is not None:
        tags = list(tags)
        if any(int(t) != 0 for t in tags):
            return [int(t) for t in tags]

    # fallback: symbols -> 1..K
    syms = atoms.get_chemical_symbols()
    mapping: Dict[str, int] = {}
    out: List[int] = []
    for s in syms:
        if s not in mapping:
            mapping[s] = len(mapping) + 1
        out.append(mapping[s])
    return out

def _assign_types_from_tags(tags):
    uniq = sorted(set(tags))
    reg = {t: i + 1 for i, t in enumerate(uniq)}  # deterministic
    out = [reg[t] for t in tags]
    return out, reg

@dataclass
class TypingOptions:
    # bond
    diff_bond_len: bool = False
    diff_bond_fmt: str = "%.2f"

    # angle
    angle_symmetry: bool = False
    diff_angle_theta: bool = False
    diff_angle_theta_fmt: str = "%.2f"

    # dihedral
    cis_trans: bool = False
    diff_dihed_theta: bool = False
    diff_dihed_theta_abs: bool = True
    diff_dihed_theta_fmt: str = "%.2f"

    # joiner
    type_delimiter: str = " "


def type_bonds(
    atoms: Atoms,
    topo: Topo,
    atom_tags,
    opts: TypingOptions = TypingOptions(),
) -> None:
    """Populate topo.bond_types and topo.bond_tags (in-place)."""
    if not topo.bonds:
        return

    atypes = _get_atom_types(atoms)
    lens = _feat.bond_lengths(atoms, topo.bonds)
    topo.meta.setdefault('features', {})['bond_length'] = lens

    tags: List[str] = []
    for (i, j), rlen in zip(topo.bonds, lens):
        # canonical pair
        parts = [atom_tags[atypes[i]], atom_tags[atypes[j]]]
        parts = sort_bond(parts)
        if opts.diff_bond_len:
            parts.append(str(opts.diff_bond_fmt % rlen))
        tags.append(opts.type_delimiter.join(parts))

    types, reg = _assign_types_from_tags(tags)
    topo.bond_tags = tags
    topo.bond_types = types
    topo.meta['bond_type_registry'] = reg  # tag -> type_id
    topo.meta['bond_type_table'] = {v: {'tag': k} for k, v in reg.items()}  # 1-based


def type_angles(
    atoms: Atoms,
    topo: Topo,
    atom_tags,
    opts: TypingOptions = TypingOptions(),
) -> None:
    """Populate topo.angle_types and topo.angle_tags (in-place)."""
    if not topo.angles:
        return

    atypes = _get_atom_types(atoms)
    thetas = _feat.angle_thetas_deg(atoms, topo.angles) if opts.diff_angle_theta else []
    syms = _feat.angle_symmetry_labels(atoms, topo.angles) if opts.angle_symmetry else []
    if thetas:
        topo.meta.setdefault('features', {})['angle_theta_deg'] = thetas
    if syms:
        topo.meta.setdefault('features', {})['angle_symmetry'] = syms

    tags: List[str] = []
    for n, (i, j, k) in enumerate(topo.angles):
        parts = [atom_tags[atypes[i]], atom_tags[atypes[j]], atom_tags[atypes[k]]]
        parts = sort_angle(parts)
        if opts.angle_symmetry:
            parts.append(syms[n])
        if opts.diff_angle_theta:
            parts.append(str(opts.diff_angle_theta_fmt % thetas[n]))
        tags.append(opts.type_delimiter.join(parts))

    types, reg = _assign_types_from_tags(tags)
    topo.angle_tags = tags
    topo.angle_types = types
    topo.meta['angle_type_registry'] = reg
    topo.meta['angle_type_table'] = {v: {'tag': k} for k, v in reg.items()}


def type_dihedrals(
    atoms: Atoms,
    topo: Topo,
    atom_tags,
    opts: TypingOptions = TypingOptions(),
) -> None:
    """Populate topo.dihedral_types and topo.dihedral_tags (in-place)."""
    if not topo.dihedrals:
        return

    atypes = _get_atom_types(atoms)
    phis = _feat.dihedral_phis_rad(atoms, topo.dihedrals) if (opts.cis_trans or opts.diff_dihed_theta) else []
    orient = _feat.dihedral_cis_trans(phis) if opts.cis_trans else []
    if phis:
        topo.meta.setdefault('features', {})['dihedral_phi_rad'] = phis
    if orient:
        topo.meta.setdefault('features', {})['dihedral_orient'] = orient

    tags: List[str] = []
    for n, (i, j, k, l) in enumerate(topo.dihedrals):
        parts = [atom_tags[atypes[i]], atom_tags[atypes[j]], atom_tags[atypes[k]], atom_tags[atypes[l]]]
        parts = sort_dihedral(parts)
        if opts.cis_trans:
            parts.append(orient[n])
        if opts.diff_dihed_theta:
            phi_deg = math.degrees(phis[n])
            if opts.diff_dihed_theta_abs:
                phi_deg = abs(phi_deg)
            parts.append(str(opts.diff_dihed_theta_fmt % phi_deg))
        tags.append(opts.type_delimiter.join(parts))

    types, reg = _assign_types_from_tags(tags)
    topo.dihedral_tags = tags
    topo.dihedral_types = types
    topo.meta['dihedral_type_registry'] = reg
    topo.meta['dihedral_type_table'] = {v: {'tag': k} for k, v in reg.items()}
