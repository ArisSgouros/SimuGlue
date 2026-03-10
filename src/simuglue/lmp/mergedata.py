from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
import warnings


@dataclass
class Atom:
    id: int
    mol_id: int
    type: int
    q: float
    x: float
    y: float
    z: float


@dataclass
class Bond:
    id: int
    type: int
    i: int
    j: int


@dataclass
class Angle:
    id: int
    type: int
    i: int
    j: int
    k: int


@dataclass
class Dihedral:
    id: int
    type: int
    i: int
    j: int
    k: int
    l: int


@dataclass
class Box:
    lo: list[float]
    hi: list[float]

    @property
    def lengths(self) -> list[float]:
        return [self.hi[d] - self.lo[d] for d in range(3)]


@dataclass
class LammpsData:
    path: Path | None = None
    box: Box = field(default_factory=lambda: Box([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]))

    atoms: Dict[int, Atom] = field(default_factory=dict)
    bonds: Dict[int, Bond] = field(default_factory=dict)
    angles: Dict[int, Angle] = field(default_factory=dict)
    dihedrals: Dict[int, Dihedral] = field(default_factory=dict)
    impropers: Dict[int, Dihedral] = field(default_factory=dict)

    masses: Dict[int, str] = field(default_factory=dict)
    pair_coeffs: Dict[int, str] = field(default_factory=dict)
    bond_coeffs: Dict[int, str] = field(default_factory=dict)
    angle_coeffs: Dict[int, str] = field(default_factory=dict)
    dihedral_coeffs: Dict[int, str] = field(default_factory=dict)
    improper_coeffs: Dict[int, str] = field(default_factory=dict)


def _next_nonempty(lines: list[str], i: int) -> int:
    j = i + 1
    while j < len(lines) and not lines[j].strip():
        j += 1
    return j


def _max_key(d: dict[int, object]) -> int:
    return max(d.keys(), default=0)


def _copy_atom(a: Atom) -> Atom:
    return Atom(a.id, a.mol_id, a.type, a.q, a.x, a.y, a.z)


def _copy_bond(b: Bond) -> Bond:
    return Bond(b.id, b.type, b.i, b.j)


def _copy_angle(a: Angle) -> Angle:
    return Angle(a.id, a.type, a.i, a.j, a.k)


def _copy_dihedral(d: Dihedral) -> Dihedral:
    return Dihedral(d.id, d.type, d.i, d.j, d.k, d.l)


def _scan_metadata(lines: list[str]) -> dict:
    counts = {
        "atoms": 0,
        "bonds": 0,
        "angles": 0,
        "dihedrals": 0,
        "impropers": 0,
        "atom types": 0,
        "bond types": 0,
        "angle types": 0,
        "dihedral types": 0,
        "improper types": 0,
    }
    starts: dict[str, int] = {}
    lo = [0.0, 0.0, 0.0]
    hi = [0.0, 0.0, 0.0]

    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s:
            continue

        toks = s.split()

        if len(toks) == 2 and toks[1] in {"atoms", "bonds", "angles", "dihedrals", "impropers"}:
            counts[toks[1]] = int(toks[0])
            continue

        if len(toks) == 3 and " ".join(toks[1:]) in {
            "atom types",
            "bond types",
            "angle types",
            "dihedral types",
            "improper types",
        }:
            counts[" ".join(toks[1:])] = int(toks[0])
            continue

        if len(toks) >= 4 and toks[-2:] == ["xlo", "xhi"]:
            lo[0], hi[0] = float(toks[0]), float(toks[1])
            continue
        if len(toks) >= 4 and toks[-2:] == ["ylo", "yhi"]:
            lo[1], hi[1] = float(toks[0]), float(toks[1])
            continue
        if len(toks) >= 4 and toks[-2:] == ["zlo", "zhi"]:
            lo[2], hi[2] = float(toks[0]), float(toks[1])
            continue

        if s.startswith("Masses"):
            starts["Masses"] = _next_nonempty(lines, i)
        elif s.startswith("Pair Coeffs"):
            starts["Pair Coeffs"] = _next_nonempty(lines, i)
        elif s.startswith("Bond Coeffs"):
            starts["Bond Coeffs"] = _next_nonempty(lines, i)
        elif s.startswith("Angle Coeffs"):
            starts["Angle Coeffs"] = _next_nonempty(lines, i)
        elif s.startswith("Dihedral Coeffs"):
            starts["Dihedral Coeffs"] = _next_nonempty(lines, i)
        elif s.startswith("Improper Coeffs"):
            starts["Improper Coeffs"] = _next_nonempty(lines, i)
        elif s.startswith("Atoms"):
            starts["Atoms"] = _next_nonempty(lines, i)
        elif s.startswith("Bonds"):
            starts["Bonds"] = _next_nonempty(lines, i)
        elif s.startswith("Angles"):
            starts["Angles"] = _next_nonempty(lines, i)
        elif s.startswith("Dihedrals"):
            starts["Dihedrals"] = _next_nonempty(lines, i)
        elif s.startswith("Impropers"):
            starts["Impropers"] = _next_nonempty(lines, i)

    return {"counts": counts, "starts": starts, "box": Box(lo, hi)}


def _read_id_string_section(
    lines: list[str],
    start: int | None,
    nrows: int,
) -> dict[int, str]:
    out: dict[int, str] = {}
    if start is None or nrows <= 0:
        return out

    for ii in range(nrows):
        toks = lines[start + ii].split()
        if not toks:
            continue
        key = int(toks[0])
        out[key] = " ".join(toks[1:])
    return out


def _read_masses(lines: list[str], start: int | None, nrows: int) -> dict[int, str]:
    out: dict[int, str] = {}
    if start is None or nrows <= 0:
        return out

    for ii in range(nrows):
        toks = lines[start + ii].split()
        if not toks:
            continue
        out[int(toks[0])] = toks[1]
    return out


def _read_atoms_full(
    lines: list[str],
    start: int | None,
    nrows: int,
    box: Box,
) -> dict[int, Atom]:
    out: dict[int, Atom] = {}
    if start is None or nrows <= 0:
        return out

    lx, ly, lz = box.lengths

    for ii in range(nrows):
        toks = lines[start + ii].split()
        if len(toks) < 7:
            raise ValueError(
                f"Atoms section appears not to be 'full' style at line {start + ii + 1}: "
                f"expected at least 7 columns, got {len(toks)}"
            )

        atom_id = int(toks[0])
        mol_id = int(toks[1])
        atom_type = int(toks[2])
        q = float(toks[3])
        x = float(toks[4])
        y = float(toks[5])
        z = float(toks[6])

        if len(toks) >= 10:
            ix = int(toks[7])
            iy = int(toks[8])
            iz = int(toks[9])
            x += ix * lx
            y += iy * ly
            z += iz * lz

        out[atom_id] = Atom(atom_id, mol_id, atom_type, q, x, y, z)

    return out


def _read_bonds(lines: list[str], start: int | None, nrows: int) -> dict[int, Bond]:
    out: dict[int, Bond] = {}
    if start is None or nrows <= 0:
        return out

    for ii in range(nrows):
        toks = lines[start + ii].split()
        if len(toks) < 4:
            raise ValueError(f"Malformed Bonds line at {start + ii + 1}")
        out[int(toks[0])] = Bond(int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3]))
    return out


def _read_angles(lines: list[str], start: int | None, nrows: int) -> dict[int, Angle]:
    out: dict[int, Angle] = {}
    if start is None or nrows <= 0:
        return out

    for ii in range(nrows):
        toks = lines[start + ii].split()
        if len(toks) < 5:
            raise ValueError(f"Malformed Angles line at {start + ii + 1}")
        out[int(toks[0])] = Angle(
            int(toks[0]), int(toks[1]), int(toks[2]), int(toks[3]), int(toks[4])
        )
    return out


def _read_dihedrals(
    lines: list[str],
    start: int | None,
    nrows: int,
) -> dict[int, Dihedral]:
    out: dict[int, Dihedral] = {}
    if start is None or nrows <= 0:
        return out

    for ii in range(nrows):
        toks = lines[start + ii].split()
        if len(toks) < 6:
            raise ValueError(f"Malformed 4-body line at {start + ii + 1}")
        out[int(toks[0])] = Dihedral(
            int(toks[0]),
            int(toks[1]),
            int(toks[2]),
            int(toks[3]),
            int(toks[4]),
            int(toks[5]),
        )
    return out


def read_lammps_data(path: str | Path) -> LammpsData:
    p = Path(path)
    lines = p.read_text().splitlines()
    meta = _scan_metadata(lines)
    counts = meta["counts"]
    starts = meta["starts"]
    box = meta["box"]

    data = LammpsData(path=p, box=box)
    data.bond_coeffs = _read_id_string_section(lines, starts.get("Bond Coeffs"), counts["bond types"])
    data.angle_coeffs = _read_id_string_section(lines, starts.get("Angle Coeffs"), counts["angle types"])
    data.dihedral_coeffs = _read_id_string_section(lines, starts.get("Dihedral Coeffs"), counts["dihedral types"])
    data.improper_coeffs = _read_id_string_section(lines, starts.get("Improper Coeffs"), counts["improper types"])
    data.pair_coeffs = _read_id_string_section(lines, starts.get("Pair Coeffs"), counts["atom types"])
    data.masses = _read_masses(lines, starts.get("Masses"), counts["atom types"])

    data.atoms = _read_atoms_full(lines, starts.get("Atoms"), counts["atoms"], box)
    data.bonds = _read_bonds(lines, starts.get("Bonds"), counts["bonds"])
    data.angles = _read_angles(lines, starts.get("Angles"), counts["angles"])
    data.dihedrals = _read_dihedrals(lines, starts.get("Dihedrals"), counts["dihedrals"])
    data.impropers = _read_dihedrals(lines, starts.get("Impropers"), counts["impropers"])

    return data


def _shift_box(box: Box, shift: list[float]) -> Box:
    return Box(
        [box.lo[d] + shift[d] for d in range(3)],
        [box.hi[d] + shift[d] for d in range(3)],
    )


def _union_box(a: Box, b: Box) -> Box:
    return Box(
        [min(a.lo[d], b.lo[d]) for d in range(3)],
        [max(a.hi[d], b.hi[d]) for d in range(3)],
    )


def _merge_coeff_dict(
    target: dict[int, str],
    incoming: dict[int, str],
    offset: int,
    label: str,
) -> None:
    for key, value in incoming.items():
        new_key = key + offset
        if new_key in target and target[new_key] != value:
            raise ValueError(
                f"{label} type collision for id {new_key}. "
                f"Use the corresponding type offset."
            )
        target[new_key] = value


def _warn_cross_section_mismatch(box_a: Box, box_b: Box, side: str, tol: float = 1e-8) -> None:
    LA = box_a.lengths
    LB = box_b.lengths

    mismatch = (
        (side in {"-x", "+x"} and (abs(LA[1] - LB[1]) > tol or abs(LA[2] - LB[2]) > tol))
        or (side in {"-y", "+y"} and (abs(LA[0] - LB[0]) > tol or abs(LA[2] - LB[2]) > tol))
        or (side in {"-z", "+z"} and (abs(LA[0] - LB[0]) > tol or abs(LA[1] - LB[1]) > tol))
    )
    if mismatch:
        warnings.warn(
            "The box cross-sections perpendicular to the merge direction are not equal.",
            stacklevel=2,
        )


def merge_systems(
    data_a: LammpsData,
    data_b: LammpsData,
    *,
    side: str = "0",
    mol_offset: int = 0,
    atom_type_offset: int = 0,
    bond_type_offset: int = 0,
    angle_type_offset: int = 0,
    dihedral_type_offset: int = 0,
    improper_type_offset: int = 0,
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    set_lo_at_zero: bool = True,
) -> LammpsData:
    if side not in {"0", "-x", "+x", "-y", "+y", "-z", "+z"}:
        raise ValueError(f"Unsupported side={side!r}")

    _warn_cross_section_mismatch(data_a.box, data_b.box, side)

    side_shift = [0.0, 0.0, 0.0]
    if side == "-x":
        side_shift[0] = data_a.box.lo[0] - data_b.box.hi[0]
    elif side == "+x":
        side_shift[0] = data_a.box.hi[0] - data_b.box.lo[0]
    elif side == "-y":
        side_shift[1] = data_a.box.lo[1] - data_b.box.hi[1]
    elif side == "+y":
        side_shift[1] = data_a.box.hi[1] - data_b.box.lo[1]
    elif side == "-z":
        side_shift[2] = data_a.box.lo[2] - data_b.box.hi[2]
    elif side == "+z":
        side_shift[2] = data_a.box.hi[2] - data_b.box.lo[2]

    total_shift = [
        side_shift[0] + pos_offset[0],
        side_shift[1] + pos_offset[1],
        side_shift[2] + pos_offset[2],
    ]

    b_box_shifted = _shift_box(data_b.box, total_shift)
    merged_box = _union_box(data_a.box, b_box_shifted)

    atom_id_offset = _max_key(data_a.atoms)
    bond_id_offset = _max_key(data_a.bonds)
    angle_id_offset = _max_key(data_a.angles)
    dihedral_id_offset = _max_key(data_a.dihedrals)
    improper_id_offset = _max_key(data_a.impropers)

    max_mol_id_a = max((a.mol_id for a in data_a.atoms.values()), default=0)
    effective_mol_offset = max(mol_offset, max_mol_id_a)

    merged = LammpsData(path=None, box=merged_box)

    merged.masses = dict(data_a.masses)
    merged.pair_coeffs = dict(data_a.pair_coeffs)
    merged.bond_coeffs = dict(data_a.bond_coeffs)
    merged.angle_coeffs = dict(data_a.angle_coeffs)
    merged.dihedral_coeffs = dict(data_a.dihedral_coeffs)
    merged.improper_coeffs = dict(data_a.improper_coeffs)

    _merge_coeff_dict(merged.masses, data_b.masses, atom_type_offset, "Mass")
    _merge_coeff_dict(merged.pair_coeffs, data_b.pair_coeffs, atom_type_offset, "Pair coeff")
    _merge_coeff_dict(merged.bond_coeffs, data_b.bond_coeffs, bond_type_offset, "Bond coeff")
    _merge_coeff_dict(merged.angle_coeffs, data_b.angle_coeffs, angle_type_offset, "Angle coeff")
    _merge_coeff_dict(merged.dihedral_coeffs, data_b.dihedral_coeffs, dihedral_type_offset, "Dihedral coeff")
    _merge_coeff_dict(merged.improper_coeffs, data_b.improper_coeffs, improper_type_offset, "Improper coeff")

    for atom in data_a.atoms.values():
        merged.atoms[atom.id] = _copy_atom(atom)

    for atom in data_b.atoms.values():
        new_atom = Atom(
            id=atom.id + atom_id_offset,
            mol_id=atom.mol_id + effective_mol_offset,
            type=atom.type + atom_type_offset,
            q=atom.q,
            x=atom.x + total_shift[0],
            y=atom.y + total_shift[1],
            z=atom.z + total_shift[2],
        )
        merged.atoms[new_atom.id] = new_atom

    for bond in data_a.bonds.values():
        merged.bonds[bond.id] = _copy_bond(bond)

    for bond in data_b.bonds.values():
        new_bond = Bond(
            id=bond.id + bond_id_offset,
            type=bond.type + bond_type_offset,
            i=bond.i + atom_id_offset,
            j=bond.j + atom_id_offset,
        )
        merged.bonds[new_bond.id] = new_bond

    for angle in data_a.angles.values():
        merged.angles[angle.id] = _copy_angle(angle)

    for angle in data_b.angles.values():
        new_angle = Angle(
            id=angle.id + angle_id_offset,
            type=angle.type + angle_type_offset,
            i=angle.i + atom_id_offset,
            j=angle.j + atom_id_offset,
            k=angle.k + atom_id_offset,
        )
        merged.angles[new_angle.id] = new_angle

    for dih in data_a.dihedrals.values():
        merged.dihedrals[dih.id] = _copy_dihedral(dih)

    for dih in data_b.dihedrals.values():
        new_dih = Dihedral(
            id=dih.id + dihedral_id_offset,
            type=dih.type + dihedral_type_offset,
            i=dih.i + atom_id_offset,
            j=dih.j + atom_id_offset,
            k=dih.k + atom_id_offset,
            l=dih.l + atom_id_offset,
        )
        merged.dihedrals[new_dih.id] = new_dih

    for imp in data_a.impropers.values():
        merged.impropers[imp.id] = _copy_dihedral(imp)

    for imp in data_b.impropers.values():
        new_imp = Dihedral(
            id=imp.id + improper_id_offset,
            type=imp.type + improper_type_offset,
            i=imp.i + atom_id_offset,
            j=imp.j + atom_id_offset,
            k=imp.k + atom_id_offset,
            l=imp.l + atom_id_offset,
        )
        merged.impropers[new_imp.id] = new_imp

    if set_lo_at_zero:
        shift = list(merged.box.lo)
        merged.box = Box(
            [merged.box.lo[d] - shift[d] for d in range(3)],
            [merged.box.hi[d] - shift[d] for d in range(3)],
        )
        for atom in merged.atoms.values():
            atom.x -= shift[0]
            atom.y -= shift[1]
            atom.z -= shift[2]

    return merged


def _type_counts(data: LammpsData) -> dict[str, int]:
    return {
        "atom": max(
            _max_key(data.masses),
            _max_key(data.pair_coeffs),
            max((a.type for a in data.atoms.values()), default=0),
        ),
        "bond": max(
            _max_key(data.bond_coeffs),
            max((b.type for b in data.bonds.values()), default=0),
        ),
        "angle": max(
            _max_key(data.angle_coeffs),
            max((a.type for a in data.angles.values()), default=0),
        ),
        "dihedral": max(
            _max_key(data.dihedral_coeffs),
            max((d.type for d in data.dihedrals.values()), default=0),
        ),
        "improper": max(
            _max_key(data.improper_coeffs),
            max((i.type for i in data.impropers.values()), default=0),
        ),
    }


def write_lammps_data(
    data: LammpsData,
    output_path: str | Path,
    *,
    source_a: str | Path | None = None,
    source_b: str | Path | None = None,
) -> None:
    counts = _type_counts(data)
    out = Path(output_path)

    with out.open("w") as f:
        if source_a is not None and source_b is not None:
            f.write(f"# A data file merged from: {source_a} and {source_b}\n")
        else:
            f.write("# Merged LAMMPS data file\n")
        f.write("#\n")
        f.write(f"{len(data.atoms)} atoms\n")
        f.write(f"{len(data.bonds)} bonds\n")
        f.write(f"{len(data.angles)} angles\n")
        f.write(f"{len(data.dihedrals)} dihedrals\n")
        f.write(f"{len(data.impropers)} impropers\n")
        f.write("\n")
        f.write(f"{counts['atom']} atom types\n")
        f.write(f"{counts['bond']} bond types\n")
        f.write(f"{counts['angle']} angle types\n")
        f.write(f"{counts['dihedral']} dihedral types\n")
        f.write(f"{counts['improper']} improper types\n")
        f.write("\n")
        f.write(f"{data.box.lo[0]:.16g} {data.box.hi[0]:.16g} xlo xhi\n")
        f.write(f"{data.box.lo[1]:.16g} {data.box.hi[1]:.16g} ylo yhi\n")
        f.write(f"{data.box.lo[2]:.16g} {data.box.hi[2]:.16g} zlo zhi\n")

        if data.masses:
            f.write("\nMasses\n\n")
            for key in sorted(data.masses):
                f.write(f"{key} {data.masses[key]}\n")

        if data.pair_coeffs:
            f.write("\nPair Coeffs\n\n")
            for key in sorted(data.pair_coeffs):
                f.write(f"{key} {data.pair_coeffs[key]}\n")

        if data.bond_coeffs:
            f.write("\nBond Coeffs\n\n")
            for key in sorted(data.bond_coeffs):
                f.write(f"{key} {data.bond_coeffs[key]}\n")

        if data.angle_coeffs:
            f.write("\nAngle Coeffs\n\n")
            for key in sorted(data.angle_coeffs):
                f.write(f"{key} {data.angle_coeffs[key]}\n")

        if data.dihedral_coeffs:
            f.write("\nDihedral Coeffs\n\n")
            for key in sorted(data.dihedral_coeffs):
                f.write(f"{key} {data.dihedral_coeffs[key]}\n")

        if data.improper_coeffs:
            f.write("\nImproper Coeffs\n\n")
            for key in sorted(data.improper_coeffs):
                f.write(f"{key} {data.improper_coeffs[key]}\n")

        f.write("\nAtoms\n\n")
        for key in sorted(data.atoms):
            a = data.atoms[key]
            f.write(
                f"{a.id} {a.mol_id} {a.type} "
                f"{a.q:.16g} {a.x:.16g} {a.y:.16g} {a.z:.16g}\n"
            )

        if data.bonds:
            f.write("\nBonds\n\n")
            for key in sorted(data.bonds):
                b = data.bonds[key]
                f.write(f"{b.id} {b.type} {b.i} {b.j}\n")

        if data.angles:
            f.write("\nAngles\n\n")
            for key in sorted(data.angles):
                a = data.angles[key]
                f.write(f"{a.id} {a.type} {a.i} {a.j} {a.k}\n")

        if data.dihedrals:
            f.write("\nDihedrals\n\n")
            for key in sorted(data.dihedrals):
                d = data.dihedrals[key]
                f.write(f"{d.id} {d.type} {d.i} {d.j} {d.k} {d.l}\n")

        if data.impropers:
            f.write("\nImpropers\n\n")
            for key in sorted(data.impropers):
                d = data.impropers[key]
                f.write(f"{d.id} {d.type} {d.i} {d.j} {d.k} {d.l}\n")


def write_lammpstrj(data: LammpsData, output_path: str | Path) -> None:
    out = Path(str(output_path) + ".lammpstrj")
    with out.open("w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write("0\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{len(data.atoms)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        f.write(f"{data.box.lo[0]:.16g} {data.box.hi[0]:.16g}\n")
        f.write(f"{data.box.lo[1]:.16g} {data.box.hi[1]:.16g}\n")
        f.write(f"{data.box.lo[2]:.16g} {data.box.hi[2]:.16g}\n")
        f.write("ITEM: ATOMS id mol type x y z\n")
        for key in sorted(data.atoms):
            a = data.atoms[key]
            f.write(f"{a.id} {a.mol_id} {a.type} {a.x:.16g} {a.y:.16g} {a.z:.16g}\n")


def merge_datafiles(
    data_file_a: str | Path,
    data_file_b: str | Path,
    output_path: str | Path,
    *,
    side: str = "0",
    mol_offset: int = 0,
    atom_type_offset: int = 0,
    bond_type_offset: int = 0,
    angle_type_offset: int = 0,
    dihedral_type_offset: int = 0,
    improper_type_offset: int = 0,
    pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    set_lo_at_zero: bool = True,
    write_dump: bool = True,
) -> LammpsData:
    data_a = read_lammps_data(data_file_a)
    data_b = read_lammps_data(data_file_b)

    merged = merge_systems(
        data_a,
        data_b,
        side=side,
        mol_offset=mol_offset,
        atom_type_offset=atom_type_offset,
        bond_type_offset=bond_type_offset,
        angle_type_offset=angle_type_offset,
        dihedral_type_offset=dihedral_type_offset,
        improper_type_offset=improper_type_offset,
        pos_offset=pos_offset,
        set_lo_at_zero=set_lo_at_zero,
    )

    write_lammps_data(
        merged,
        output_path,
        source_a=data_file_a,
        source_b=data_file_b,
    )
    if write_dump:
        write_lammpstrj(merged, output_path)

    return merged
