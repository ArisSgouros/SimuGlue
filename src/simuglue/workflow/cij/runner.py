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
from simuglue.mechanics.voigt import normalize_components_to_voigt1, stress_tensor_to_voigt6
from simuglue.mechanics.kinematics import F_from_voigt_component as deformation_gradient
from ase.calculators.lammps.unitconvert import convert
from ase import units

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
