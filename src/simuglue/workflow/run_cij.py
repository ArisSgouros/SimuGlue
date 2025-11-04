# src/simuglue/workflows/cij_run.py
from __future__ import annotations
import json, shutil, subprocess, re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import yaml
from ase.io import read, write
from simuglue.io.util_ase_lammps import read_lammps, write_lammps
from simuglue.transform.linear import apply_transform, voigt_to_cart

def stress_tensor_to_voigt6(S: np.ndarray) -> np.ndarray:
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
    def relax_and_stress(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult: ...

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

# ---------- LAMMPS backend (skeleton) ----------
@register_backend("lammps")
class LAMMPSBackend(Backend):
    def prepare_case(self, case_dir: Path, atoms, cfg: Config):
        print("Preparing case!")
        # Minimal: write a data file with ASE; let template refer to it.
        write_lammps(atoms, case_dir / "str.data", style="atomic", units="metal", force_skew="False")
 
        # Render the user template (keep it simple: {datafile} placeholder)
        tpl = Path(cfg.lammps["input_template"]).read_text()
        tpl = tpl.replace("${datafile}", "str.data")
        (case_dir / "in.min").write_text(tpl, encoding="utf-8")
        #(case_dir / "in.min").write_text("ASDF", encoding="utf-8")

        # Copy include files if listed
        for p in cfg.lammps.get("include_files", []):
            src = Path(p)
            dst = case_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy(src, dst)

    def relax_and_stress(self, case_dir: Path, atoms, cfg: Config) -> RelaxResult:
        exe = cfg.lammps.get("exe", "lmp")
        log = (case_dir / "log.lammps")
        subprocess.run([exe, "-in", "in.min"], cwd=case_dir, check=True, stdout=log.open("w"), stderr=subprocess.STDOUT)
        text = log.read_text(errors="ignore")
        # Parse per-box stress (bar) triplet and shear components if printed.
        # Recommend user prints "thermo_style custom pxx pyy pzz pxy pxz pyz pe"
        # Example quick parser:
        p = re.findall(r"^\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$",
                       text, flags=re.M)
        if not p:
            raise RuntimeError("Could not parse pressures from LAMMPS log; ensure thermo outputs pxx pyy pzz pxy pxz pyz pe")
        pxx, pyy, pzz, pxy, pxz, pyz, pe = map(float, p[-1])
        print("pxx:", pxx)
        ## LAMMPS pressures are in bar by default → GPa
        fac = 1e-4  # 1 bar = 0.0001 GPa
        S = -np.array([[pxx, pxy, pxz],
                      [pxy, pyy, pyz],
                      [pxz, pyz, pzz]], float) * fac
        return RelaxResult(atoms, pe, S)

# ---------- main workflow ----------
def run_cij(config_path: str):
    print("Hello from cij")
    cfg = _load_config(config_path)

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
    res = backend.relax_and_stress(case_dir, atoms_ref, cfg)
    s6_ref = stress_tensor_to_voigt6(res.stress)

    # For each requested Voigt component, deform the system and estimate the strain energy
    for eps in cfg.strains:
        for i in cfg.components:
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
            res = backend.relax_and_stress(case_dir, atoms_def, cfg)

            s6 = stress_tensor_to_voigt6(res.stress)
            rows.append((i, eps, s6, res.energy, str(case_dir)))

    CC_all = {}
    for i in cfg.components:
        for j in cfg.components:
            CC_all[i, j] = []
    for row in rows:
        i, eps, s6_def = row[:3]
        for j in cfg.components:
           # C_{ij} = (S_j(def) - S_j(ref)) / ε_i, where i=direction, j=stress component
           CC_eps = (s6_def[j-1] - s6_ref[j-1]) / eps
           CC_all[i, j].append(CC_eps)


    # Statistics: mean and SEM per component
    CC_mean = {}
    CC_sem = {}
    for i in cfg.components:
        for j in cfg.components:
            arr = np.array(CC_all[i, j])
            if arr.size == 0:
                print(f"Warning: no samples for {comp}", file=sys.stderr)
                CC_mean[i, j] = float("nan")
                CC_sem[i, j] = float("nan")
                continue
            CC_mean[i, j] = float(np.mean(arr))
            CC_sem[i, j] = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0


    for key in CC_all:
        print(key, CC_mean[key], CC_sem[key])

            
#
    ## Persist outputs
    #out_csv = cfg.output.get("csv", "results.csv")
    #with open(cfg.workdir / out_csv, "w", encoding="utf-8") as f:
        #f.write("i_component,eps,case_dir," + ",".join([f"s{j}" for j in range(6)]) + ",energy\n")
        #for i, e, s6, E, d in rows:
            #f.write(f"{i},{e},{d}," + ",".join(f"{x:.8g}" for x in s6) + f",{E:.8g}\n")
#
    #out_json = cfg.output.get("cij_json", "C_partial.json")
    #payload = {
        #"components": cfg.components,
        #"C_shape": [6, len(cfg.components)],
        #"C_rows": ["xx","yy","zz","yz","xz","xy"],
        #"C_cols": [int(i) for i in cfg.components],
        #"C": C.tolist(),  # GPa
        #"note": "Rows are stress Voigt components (6). Columns follow requested epsilon components.",
    #}
    #(cfg.workdir / out_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
#
