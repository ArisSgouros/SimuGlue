# src/simuglue/workflows/cij_run.py
from __future__ import annotations
import sys
import json, shutil, subprocess, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Union, Dict
import numpy as np
import yaml
from ase.io import read, write
from simuglue.io.util_ase_lammps import read_lammps, write_lammps
from simuglue.transform.linear import apply_transform
from ase.calculators.lammps.unitconvert import convert
from ase import units

# Internal 1-based Voigt mapping:
# 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
_NAME_TO_VOIGT1 = {
    "xx": 1, "yy": 2, "zz": 3,
    "yz": 4, "xz": 5, "xy": 6,
}

def normalize_components_to_voigt1(components: Iterable[Union[int, str]]) -> list[int]:
    """
    Accepts user-supplied components as ints (1..6) or strings ('xx','yy','zz','xy','xz','yz')
    and returns a validated list of 1-based Voigt indices following:
      1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
    Preserves user order, removes accidental whitespace, and is case-insensitive for names.
    """
    out: list[int] = []
    for c in components:
        if isinstance(c, int):
            if 1 <= c <= 6:
                out.append(c)
            else:
                raise ValueError(f"Voigt index out of range (must be 1..6): {c}")
        else:
            s = str(c).strip().lower()
            if s.isdigit():
                # If user passed "1","2",... as strings, honor them (still 1-based)
                v = int(s)
                if 1 <= v <= 6:
                    out.append(v)
                else:
                    raise ValueError(f"Voigt index (string) out of range 1..6: {c}")
            else:
                if s not in _NAME_TO_VOIGT1:
                    raise ValueError(
                        f"Unknown component name '{c}'. "
                        f"Allowed: xx, yy, zz, xy, xz, yz (case-insensitive), or 1..6."
                    )
                out.append(_NAME_TO_VOIGT1[s])
    return out

def stress_tensor_to_voigt6(S: np.ndarray) -> np.ndarray:
    # symmetrize to be safe
    S = 0.5 * (S + S.T)
    return np.array([S[0,0], S[1,1], S[2,2], S[1,2], S[0,2], S[0,1]], float)

def deformation_gradient(dir_idx: int, s: float) -> np.ndarray:
    """
    Build F for Voigt dir: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy.
    s = ±up (engineering strain / shear).
    """
    F = np.eye(3)
    if   dir_idx == 1:  # xx
        F[0, 0] += s
    elif dir_idx == 2:  # yy
        F[1, 1] += s
    elif dir_idx == 3:  # zz
        F[2, 2] += s
    elif dir_idx == 4:  # yz: y' += s * z
        F[1, 2] += s
    elif dir_idx == 5:  # xz: x' += s * z
        F[0, 2] += s
    elif dir_idx == 6:  # xy: x' += s * y
        F[0, 1] += s
    else:
        raise ValueError("dir_idx must be 1..6")
    return F


# ---------- config ----------
@dataclass
class Config:
    backend: str
    workdir: Path
    data_file: Path
    file_type: str
    components: List[int]
    strains: List[float]
    relax: Dict
    qe: Dict
    lammps: Dict
    output: Dict

def _load_config(path: str) -> Config:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return Config(
        backend=cfg["backend"],
        workdir=Path(cfg["workdir"]),
        data_file=Path(cfg["data_file"]),
        file_type=cfg["file_type"],
        components=list(cfg["components"]),
        strains=list(cfg["strains"]),
        relax=cfg.get("relax", {}),
        qe=cfg.get("qe", {}),
        lammps=cfg.get("lammps", {}),
        output=cfg.get("output", {}),
    )

# ---------- backend registry ----------
class RelaxResult:
    def __init__(self, atoms, energy: float, stress_tensor_3x3: np.ndarray):
        self.atoms = atoms
        self.energy = energy
        self.stress = stress_tensor_3x3  # in GPa (your canonical choice)

class Backend:
    def prepare_case(self, case_dir: Path, atoms, cfg: Config): ...
    def run_case(self, case_dir: Path, atoms, cfg: Config): ...
    def parse_case(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult: ...

_BACKENDS = {}
def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls()
        return cls
    return deco

def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(_BACKENDS)}")
    return _BACKENDS[name]

def is_done(case_dir: Path) -> bool:
    return (case_dir / ".done").exists()

def mark_done(case_dir: Path):
    (case_dir / ".done").write_text("ok\n", encoding="utf-8")

# ---------- LAMMPS backend (skeleton) ----------
@register_backend("lammps")
class LAMMPSBackend(Backend):

    def prepare_case(self, case_dir: Path, atoms, cfg: Config):

        # If already done, do nothing.
        if is_done(case_dir):
            return

        print("Preparing case!")
        # Minimal: write a data file with ASE; let template refer to it.
        write_lammps(atoms, case_dir / "str.data", style="atomic", units="metal", force_skew="False")
 
        # Render the user template (keep it simple: {datafile} placeholder)
        tpl = Path(cfg.lammps["input_template"]).read_text()

        # set datafile
        tpl = tpl.replace("${datafile}", "str.data")

        # append json thermo to end of file
        print_line = "print '{\"pxx\":$(pxx), \"pyy\":$(pyy), \"pzz\":$(pzz), \"pyz\":$(pyz), \"pxz\":$(pxz), \"pxy\":$(pxy), \"pe\":$(pe)}' file thermo.json"
        if "thermo.json" not in tpl:               # only append once
            tpl = tpl.rstrip() + "\n" + print_line + "\n"

        # write file
        dst = case_dir / "in.min"
        if not is_done(case_dir) and not dst.exists():
            dst.write_text(tpl, encoding="utf-8")

        # Copy include files if listed
        for p in cfg.lammps.get("include_files", []):
            src = Path(p)
            dst = case_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy(src, dst)

    def run_case(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult:
        if is_done(case_dir):
            return
        exe = cfg.lammps.get("exe", "lmp")
        log = (case_dir / "log.lammps")
        subprocess.run([exe, "-in", "in.min"], cwd=case_dir, check=True, stdout=log.open("w"), stderr=subprocess.STDOUT)
        text = log.read_text(errors="ignore")
        status = (case_dir / "thermo.json").is_file()
        if not status:
            raise RuntimeError("LAMMPS finished but thermo.json missing; check template/thermo line.")
        mark_done(case_dir)
        return

    def parse_case(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult:

        if not is_done(case_dir):
            raise ValueError(f"Attempted to parse a running/not done case")

        data = json.loads((case_dir / "thermo.json").read_text())

        # convert lammps units to ase
        lammps_units = cfg.lammps.get("units", "metal")
        p_evA3 = convert(1.0, "pressure", lammps_units, "ASE")
        p_GPa  = p_evA3 / units.GPa
        e_eV = convert(1.0, "energy", lammps_units, "ASE")

        pe = data["pe"]*e_eV

        # LAMMPS pressure tensor is +p (compression positive) = -σ
        S = -np.array([
            [data["pxx"], data["pxy"], data["pxz"]],
            [data["pxy"], data["pyy"], data["pyz"]],
            [data["pxz"], data["pyz"], data["pzz"]],
        ], dtype=float) * p_GPa

        return RelaxResult(atoms, pe, S)

# ---------- main workflow ----------
def run_cij(config_path: str):
    print("Hello from cij")
    cfg = _load_config(config_path)

    components = normalize_components_to_voigt1(cfg.components)   # e.g. [1,2,6]
    strains = [float(eps) for eps in cfg.strains]

    cfg.workdir.mkdir(parents=True, exist_ok=True)
    if cfg.file_type == "lammps":
        atoms_ref = read_lammps(cfg.data_file)
    else: # "xyz"
        atoms_ref = read(cfg.data_file)

    backend = get_backend(cfg.backend)

    rows = []

    # Calculate reference case
    eid = "ref"
    case_dir = cfg.workdir / eid
    case_dir.mkdir(parents=True, exist_ok=True)

    # backend prep & run
    backend.prepare_case(case_dir, atoms_ref, cfg)
    backend.run_case(case_dir, atoms_ref, cfg)
    res = backend.parse_case(case_dir, atoms_ref, cfg)
    s6_ref = stress_tensor_to_voigt6(res.stress)

    # For each requested Voigt component, deform the system and estimate the strain energy
    for eps in strains:
        for i in components:
            print(i, eps)
            eid = f"c{i}_eps{eps:g}"
            case_dir = cfg.workdir / eid
            case_dir.mkdir(parents=True, exist_ok=True)

            # deform
            F = deformation_gradient(i, eps)
            atoms_def = apply_transform(atoms_ref, F)

            # export deformed sample (optional)
            if cfg.output.get("save_traj", True):
                write(case_dir / "deformed.xyz", atoms_def)

            ## backend prep & run
            backend.prepare_case(case_dir, atoms_def, cfg)
            backend.run_case(case_dir, atoms_def, cfg)
            res = backend.parse_case(case_dir, atoms_def, cfg)

            s6 = stress_tensor_to_voigt6(res.stress)
            rows.append((i, eps, s6, res.energy, str(case_dir)))

    CC_all = {}
    for i in components:
        for j in components:
            CC_all[i, j] = []

    def _vpos(j_1based: int) -> int:
        """Map Voigt 1..6 → 0..5 for s6 indexing."""
        return j_1based - 1

    for (i, eps, s6_def, energy, case_dir) in rows:
        if not np.isfinite(s6_def).all():
            print(f"[cij] Non-finite stress for i={i}, eps={eps} in {case_dir}")
        for j in components:
           jpos = _vpos(j)  # 0..5
           # C_{ij} = (S_j(def) - S_j(ref)) / ε_i, where i=direction, j=stress component
           if abs(eps) < 1e-12:
               raise ValueError(f"Zero or tiny strain for component {i}: eps={eps}")
           CC_eps = (s6_def[jpos] - s6_ref[jpos]) / eps
           CC_all[i, j].append(CC_eps)


    # Statistics: mean and SEM per component
    CC_mean = {}
    CC_sem = {}
    for i in components:
        for j in components:
            arr = np.array(CC_all[i, j])
            if arr.size == 0:
                print(f"Warning: no samples for component {i}{j}", file=sys.stderr)
                CC_mean[i, j] = float("nan")
                CC_sem[i, j] = float("nan")
                continue
            CC_mean[i, j] = float(np.mean(arr))
            CC_sem[i, j] = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0

    # Check whether the cij matrix is symmetric
    for i in components:
        for j in components:
            if (j, i) in CC_mean:
                diff = abs(CC_mean[(i,j)] - CC_mean[(j,i)])
                if diff > 1e-3:  # GPa tolerance, tweak as you like
                    print(f"[cij] Note: C[{i},{j}] != C[{j},{i}] (diff={diff:.3f} GPa)")

    for key in CC_all:
        print(key, CC_mean[key], CC_sem[key])

    out = {
        "components": cfg.components,     # still 1-based
        "strains": strains,
        "C_mean": {f"{i}-{j}": CC_mean[(i,j)] for i in components for j in components},
        "C_sem":  {f"{i}-{j}": CC_sem[(i,j)]  for i in components for j in components},
        "units": {"stress":"GPa","strain":"-"},
    }
    (cfg.workdir / "cij.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out
