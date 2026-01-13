import numpy as np
import pytest

from ase import Atoms

from simuglue.topology.core import ensure_atom_tags_from_lmp_type_table
from simuglue.topology.infer import infer_bonds_by_distance
from simuglue.topology.topo import Topo
from simuglue.topology.typing import TypingOptions, type_bonds, type_angles, type_dihedrals
from simuglue.topology import features as feat


def _atoms_water_like(*, cell=10.0, pbc=True) -> Atoms:
    atoms = Atoms(
        symbols=["O", "H", "H"],
        positions=[
            (0.0, 0.0, 0.0),
            (0.96, 0.0, 0.0),
            (0.0, 0.96, 0.0),
        ],
        cell=[cell, cell, cell],
        pbc=[pbc, pbc, pbc],
    )
    atoms.arrays["type"] = np.array([1, 2, 2], dtype=int)
    return atoms


def _atoms_chain_4(*, phi: str, cell=20.0, pbc=True) -> Atoms:
    """4-atom chain with a dihedral close to 0° (cis) or 180° (trans)."""
    s = 1.0 / np.sqrt(2.0)
    if phi == "trans":
        r3 = (1.0 + s, -s, 0.0)
    elif phi == "cis":
        r3 = (1.0 + s, +s, 0.0)
    else:
        raise ValueError("phi must be 'cis' or 'trans'")

    atoms = Atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            (-s, +s, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            r3,
        ],
        cell=[cell, cell, cell],
        pbc=[pbc, pbc, pbc],
    )
    atoms.arrays["type"] = np.array([1, 1, 1, 1], dtype=int)
    return atoms


def test_ensure_atom_tags_from_lmp_type_table_replaces_underscore():
    lmp_type_table = {"tag": {1: "_", 2: "H"}}
    atom_tag = ensure_atom_tags_from_lmp_type_table(lmp_type_table)
    assert atom_tag == {1: "1", 2: "H"}


def test_type_bonds_canonicalizes_pair_and_assigns_single_type():
    atoms = _atoms_water_like()
    topo, _ = infer_bonds_by_distance(atoms, rc_list=[0.96], drc=0.05)

    atom_tag = {1: "O", 2: "H"}
    type_bonds(atoms, topo, atom_tag, opts=TypingOptions(type_delimiter=" "))

    assert topo.bond_types == [1, 1]

    # Canonicalization uses lexicographic min over (A B) vs (B A)
    # so O-H becomes "H O".
    assert topo.bond_tags == ["H O", "H O"]

    # Type table should exist and be 1-based.
    bt = topo.meta.get("bond_type_table", {})
    assert bt[1]["tag"] == "H O"


def test_type_angles_with_symmetry_and_theta_appends_to_tags():
    atoms = _atoms_water_like()
    topo, neighbors = infer_bonds_by_distance(atoms, rc_list=[0.96], drc=0.05)
    topo.angles = [(1, 0, 2)]

    atom_tag = {1: "O", 2: "H"}
    opts = TypingOptions(angle_symmetry=True, diff_angle_theta=True, diff_angle_theta_fmt="%.1f")
    type_angles(atoms, topo, atom_tag, opts=opts)

    # For i-k in plane, symmetry label should be 'T' (delta_z ~ 0)
    # Angle is 90.0 degrees in this geometry.
    assert topo.angle_tags == ["H O H T 90.0"]
    assert topo.angle_types == [1]

    # Stored features are exposed via topo.meta
    feats = topo.meta.get("features", {})
    assert "angle_theta_deg" in feats
    assert abs(feats["angle_theta_deg"][0] - 90.0) < 1e-8
    assert feats["angle_symmetry"][0] == "T"


def test_type_dihedrals_cis_trans_classification():
    atom_tag = {1: "C"}

    atoms_t = _atoms_chain_4(phi="trans")
    topo_t, neighbors_t = infer_bonds_by_distance(atoms_t, rc_list=[1.0], drc=0.05)
    topo_t.dihedrals = [(0, 1, 2, 3)]

    opts = TypingOptions(cis_trans=True)
    type_dihedrals(atoms_t, topo_t, atom_tag, opts=opts)

    assert topo_t.dihedral_types == [1]
    assert topo_t.dihedral_tags == ["C C C C trans"]

    # Cross-check against the feature function directly.
    phis = feat.dihedral_phis_rad(atoms_t, topo_t.dihedrals)
    orient = feat.dihedral_cis_trans(phis)
    assert orient == ["trans"]

    atoms_c = _atoms_chain_4(phi="cis")
    topo_c, neighbors_c = infer_bonds_by_distance(atoms_c, rc_list=[1.0], drc=0.05)
    topo_c.dihedrals = [(0, 1, 2, 3)]

    type_dihedrals(atoms_c, topo_c, atom_tag, opts=opts)
    assert topo_c.dihedral_tags == ["C C C C cis"]


def test_type_dihedrals_theta_abs_vs_signed_changes_tag():
    # Build a signed -180° by flipping the final atom out of plane.
    s = 1.0 / np.sqrt(2.0)
    atoms = Atoms(
        symbols=["C", "C", "C", "C"],
        positions=[
            (-s, +s, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            # Small out-of-plane displacement to make the dihedral *signed* (negative).
            (1.0 + s, -s, -1e-3),
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=[True, True, True],
    )
    atoms.arrays["type"] = np.array([1, 1, 1, 1], dtype=int)

    topo, _ = infer_bonds_by_distance(atoms, rc_list=[1.0], drc=0.05)
    topo.dihedrals = [(0, 1, 2, 3)]

    atom_tag = {1: "C"}

    # Signed phi
    opts_signed = TypingOptions(diff_dihed_theta=True, diff_dihed_theta_abs=False, diff_dihed_theta_fmt="%.0f")
    type_dihedrals(atoms, topo, atom_tag, opts=opts_signed)
    tag_signed = topo.dihedral_tags[0]

    # Absolute phi
    topo2 = Topo(bonds=topo.bonds, dihedrals=topo.dihedrals)
    opts_abs = TypingOptions(diff_dihed_theta=True, diff_dihed_theta_abs=True, diff_dihed_theta_fmt="%.0f")
    type_dihedrals(atoms, topo2, atom_tag, opts=opts_abs)
    tag_abs = topo2.dihedral_tags[0]

    # They should differ when phi is negative and abs is off.
    assert tag_signed != tag_abs
