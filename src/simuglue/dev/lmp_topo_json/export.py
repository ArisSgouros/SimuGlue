from __future__ import annotations
from simuglue.topology.topo import Topo
from typing import Optional, Dict, Any, Mapping, Tuple
import copy
import json

def ExportTypesTopo(
    filename: str,
    topo: Topo,
    masses_table: Optional[Dict] = None,
) -> None:
    """Export type tables from `topo.meta` and optionally a LAMMPS-style Masses section.

    Parameters
    ----------
    filename : str
        Output file.
    topo : Topo
        Topology object with `meta` tables.
    masses_table : dict, optional
        LAMMPS type table as returned by `_get_lmp_type_table(atoms)`.
        Expected keys: 'n_types', 'mass', 'tag' (all optional but recommended).
    """
    with open(filename, 'w') as fout:
        fout.write('\n')

        # --- Topology type tables ---
        bt = topo.meta.get('bond_type_table', {})
        at = topo.meta.get('angle_type_table', {})
        dt = topo.meta.get('dihedral_type_table', {})

        if bt:
            fout.write(f"{len(bt)} bond types\n")
        if at:
            fout.write(f"{len(at)} angle types\n")
        if dt:
            fout.write(f"{len(dt)} dihedral types\n")

        # --- Masses header (if provided) ---
        if masses_table:
            n_types = int(masses_table.get("n_types", 0))
            if n_types <= 0:
                # infer from keys if not explicitly provided
                mass_map = masses_table.get("mass", {})
                if mass_map:
                    n_types = max(mass_map.keys())
            if n_types > 0:
                fout.write(f"{n_types} atom types\n")

        fout.write('\n')

        # --- Masses section ---
        if masses_table:
            fout.write("Masses\n\n")
            mass_map = masses_table.get("mass", {})
            tag_map = masses_table.get("tag", {})
            n_types = int(masses_table.get("n_types", 0))
            if n_types <= 0 and mass_map:
                n_types = max(mass_map.keys())

            for tid in range(1, n_types + 1):
                mass = mass_map.get(tid, 0.0)
                tag = tag_map.get(tid, "")
                if tag:
                    fout.write(f"{tid} {mass} # {tag}\n")
                else:
                    fout.write(f"{tid} {mass}\n")
            fout.write('\n')

        # --- Bond types ---
        if bt:
            fout.write('Bond Types\n\n')
            for tid in sorted(bt.keys()):
                fout.write(f"{tid} {bt[tid].get('tag','')}\n")

        # --- Angle types ---
        if at:
            fout.write('\nAngle Types\n\n')
            for tid in sorted(at.keys()):
                fout.write(f"{tid} {at[tid].get('tag','')}\n")

        # --- Dihedral types ---
        if dt:
            fout.write('\nDihedral Types\n\n')
            for tid in sorted(dt.keys()):
                fout.write(f"{tid} {dt[tid].get('tag','')}\n")


#
# json format
#

TypesTopoDict = Dict[str, Any]


def types_topo_to_dict(
    topo: Topo,
    masses_table: Optional[Mapping[str, Any]] = None,
    *,
    include_empty: bool = False,
    schema_version: int = 1,
) -> TypesTopoDict:
    """
    Export topology type tables (and optionally masses_table) to a pure-Python dict.

    The returned dict is convenient for debugging and can be JSON/YAML serialized.

    Notes on serialization:
      - JSON requires string keys. This function keeps type-ids as ints in Python.
      - Use `types_topo_to_json_safe()` before json.dump, and
        `types_topo_from_json_safe()` after json.load, to roundtrip int keys.
    """
    meta = getattr(topo, "meta", {}) or {}

    bt = meta.get("bond_type_table", {}) or {}
    at = meta.get("angle_type_table", {}) or {}
    dt = meta.get("dihedral_type_table", {}) or {}

    out: TypesTopoDict = {
        "format": "simuglue.types_topo",
        "schema_version": int(schema_version),
        "counts": {
            "bond_types": int(len(bt)),
            "angle_types": int(len(at)),
            "dihedral_types": int(len(dt)),
            "atom_types": 0,
        },
        "tables": {},
        "masses": None,
    }

    # Copy tables so the dict is independent of topo.meta references
    if bt or include_empty:
        out["tables"]["bond"] = {int(k): copy.deepcopy(v) for k, v in bt.items()}
    if at or include_empty:
        out["tables"]["angle"] = {int(k): copy.deepcopy(v) for k, v in at.items()}
    if dt or include_empty:
        out["tables"]["dihedral"] = {int(k): copy.deepcopy(v) for k, v in dt.items()}

    if masses_table:
        # Normalize and copy (keep ids as ints)
        n_types = int(masses_table.get("n_types", 0) or 0)
        mass_map = dict(masses_table.get("mass", {}) or {})
        tag_map = dict(masses_table.get("tag", {}) or {})

        # infer n_types if missing
        if n_types <= 0 and mass_map:
            try:
                n_types = int(max(int(k) for k in mass_map.keys()))
            except ValueError:
                n_types = 0

        # normalize keys -> int
        mass_i = {int(k): float(v) for k, v in mass_map.items()}
        tag_i = {int(k): str(v) for k, v in tag_map.items()}

        out["masses"] = {
            "n_types": int(n_types),
            "mass": mass_i,
            "tag": tag_i,
        }
        out["counts"]["atom_types"] = int(n_types)

    return out


def types_topo_to_json_safe(d: TypesTopoDict) -> TypesTopoDict:
    """
    Convert int keys to strings (recursively for known sections) so json.dump works.
    """
    out = copy.deepcopy(d)

    def _kstr(m: Optional[Dict[Any, Any]]) -> Optional[Dict[str, Any]]:
        if m is None:
            return None
        return {str(k): v for k, v in m.items()}

    tables = out.get("tables", {}) or {}
    for name in ("bond", "angle", "dihedral"):
        if name in tables and isinstance(tables[name], dict):
            tables[name] = _kstr(tables[name])

    masses = out.get("masses", None)
    if isinstance(masses, dict):
        if "mass" in masses and isinstance(masses["mass"], dict):
            masses["mass"] = _kstr(masses["mass"])
        if "tag" in masses and isinstance(masses["tag"], dict):
            masses["tag"] = _kstr(masses["tag"])

    return out


def types_topo_from_json_safe(d: TypesTopoDict) -> TypesTopoDict:
    """
    Convert string keys back to ints for known sections after json.load.
    """
    out = copy.deepcopy(d)

    def _kint(m: Optional[Dict[Any, Any]]) -> Optional[Dict[int, Any]]:
        if m is None:
            return None
        return {int(k): v for k, v in m.items()}

    tables = out.get("tables", {}) or {}
    for name in ("bond", "angle", "dihedral"):
        if name in tables and isinstance(tables[name], dict):
            tables[name] = _kint(tables[name])

    masses = out.get("masses", None)
    if isinstance(masses, dict):
        if "mass" in masses and isinstance(masses["mass"], dict):
            masses["mass"] = _kint(masses["mass"])
        if "tag" in masses and isinstance(masses["tag"], dict):
            masses["tag"] = _kint(masses["tag"])

    return out


def write_types_topo_json(filename: str, topo_dict: TypesTopoDict, *, indent: int = 2) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(types_topo_to_json_safe(topo_dict), f, indent=indent, ensure_ascii=False)
        f.write("\n")


def read_types_topo_json(filename: str) -> TypesTopoDict:
    with open(filename, "r", encoding="utf-8") as f:
        return types_topo_from_json_safe(json.load(f))


def apply_types_topo_dict_to_topo(
    topo: Topo, topo_dict: TypesTopoDict
) -> Optional[Dict[str, Any]]:
    """
    Populate topo.meta tables from a dict produced by types_topo_to_dict().
    Returns a masses_table dict (or None if absent).
    """
    tables = topo_dict.get("tables", {}) or {}

    if getattr(topo, "meta", None) is None:
        topo.meta = {}

    if "bond" in tables:
        topo.meta["bond_type_table"] = {int(k): dict(v) for k, v in tables["bond"].items()}
    if "angle" in tables:
        topo.meta["angle_type_table"] = {int(k): dict(v) for k, v in tables["angle"].items()}
    if "dihedral" in tables:
        topo.meta["dihedral_type_table"] = {int(k): dict(v) for k, v in tables["dihedral"].items()}

    masses = topo_dict.get("masses", None)
    if not isinstance(masses, dict):
        return None

    # ensure int keys
    mass_map = masses.get("mass", {}) or {}
    tag_map = masses.get("tag", {}) or {}
    return {
        "n_types": int(masses.get("n_types", 0) or 0),
        "mass": {int(k): float(v) for k, v in mass_map.items()},
        "tag": {int(k): str(v) for k, v in tag_map.items()},
    }

