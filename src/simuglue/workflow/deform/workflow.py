from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Iterator

import numpy as np

from simuglue.transform.linear import apply_transform
from simuglue.mechanics.voigt import (
    normalize_components_to_voigt1,
    stress_tensor_to_voigt6,
)

from .config import Config, load_config
from .registry import get_backend, is_done, is_running, RelaxResult, make_case_id


# -------------------- helpers --------------------

def _validate(cfg: Config, components: list[int], strains: list[float]) -> None:
    if not components:
        raise ValueError("Config 'components' is empty.")
    if not strains:
        raise ValueError("Config 'strains' is empty.")

def F_from_component(dir_idx: int, s: float) -> np.ndarray:
    """Build F for uniaxial strain: 1=xx, 2=yy, 3=zz."""
    F = np.eye(3)
    if dir_idx == 1: F[0, 0] += s
    elif dir_idx == 2: F[1, 1] += s
    elif dir_idx == 3: F[2, 2] += s
    else: raise ValueError("Deformation dir_idx must be 1..3 for uniaxial")
    return F

def _copy_common_files(cfg: Config) -> None:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    if not cfg.common_files:
        return
    target_base = cfg.workdir / cfg.common_path
    target_base.mkdir(parents=True, exist_ok=True)
    for src in cfg.common_files:
        src = Path(src)
        dst = target_base / src.name
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)


def _dump_result_json(case_dir: Path, kind: str, i: int | None, eps: float | None, res: RelaxResult) -> None:
    """Writes the result.json file that post_deformation.py reads."""
    s6 = stress_tensor_to_voigt6(res.stress)
    payload = {
        "kind": kind,
        "i": i,
        "eps": eps,
        "energy": float(res.energy),
        "stress6": [float(x) for x in s6],
        "cell": res.cell.tolist(),
        "units": {"stress": "eV/\u00c5^3", "energy": "eV", "strain": "-"},
    }
    (case_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -------------------- 1) init --------------------

def init_deformation(config_path: str) -> None:
    """Creates folders and prepares LAMMPS input files for each strain step."""
    cfg = load_config(config_path)
    components = normalize_components_to_voigt1(cfg.components)
    strains = sorted([float(eps) for eps in cfg.strains]) 
    _validate(cfg, components, strains)

    _copy_common_files(cfg)


    backend = get_backend(cfg.backend)
    atoms_ref = backend.read_data(cfg)

    # Prepare Reference
    ref_dir = cfg.workdir / "run.ref"
    ref_dir.mkdir(parents=True, exist_ok=True)
    backend.prepare_case(ref_dir, atoms_ref, cfg)

    # Prepare Deformed Cases
    for i in components:
        cfg.active_component = i  # Store the current pull direction (1, 2, or 3)
        for eps in strains:
            cid = make_case_id(i, eps)
            case_dir = cfg.workdir / cid
            case_dir.mkdir(parents=True, exist_ok=True)
            
            F = F_from_component(i, eps)
            atoms_def = apply_transform(atoms_ref, F)
            backend.prepare_case(case_dir, atoms_def, cfg)


# -------------------- 2) run & parse --------------------

def run_deformation(config_path: str) -> None:
    """
    Executes simulations sequentially. 
    Implements 'Early Exit' if backend detects a non-physical state (NaN).
    """
    cfg = load_config(config_path)
    components = normalize_components_to_voigt1(cfg.components)
    strains = sorted([float(eps) for eps in cfg.strains]) 
    backend = get_backend(cfg.backend)

    # 1. Reference Run
    ref_dir = cfg.workdir / "run.ref"
    if not is_done(ref_dir):
        print("[deformation/run] Running reference...")
        backend.run_case(ref_dir, cfg)
        res_ref = backend.parse_case(ref_dir, cfg)
        _dump_result_json(ref_dir, "ref", None, None, res_ref)

    # 2. Deformed Loop with Early Exit
    for i in components:
        print(f"\n[deformation/run] --- Starting Direction {i} ---")
        
        for eps in strains:
            cid = make_case_id(i, eps)
            case_dir = cfg.workdir / cid

            if is_done(case_dir):
                # Check if this done-case was already a failure to decide if we skip direction
                res = backend.parse_case(case_dir, cfg)
                if np.isnan(res.energy): break 
                continue

            print(f"[deformation/run] Executing {cid}...")
            backend.run_case(case_dir, cfg)
            
            # Parse immediately after run to check if material failed
            try:
                res = backend.parse_case(case_dir, cfg)
                _dump_result_json(case_dir, "sample", i, eps, res)

                # Check for NaN energy signal from lammps.py
                if np.isnan(res.energy):
                    print(f"[deformation/run] Lattice failed at eps={eps}. Breaking loop.")
                    break 

            except Exception as e:
                print(f"[deformation/run] Simulation error at {cid}: {e}")
                break

    print("\n[deformation/run] All directions finished or reached failure.")
