from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Iterator
import numpy as np

from scipy.linalg import logm, expm # Required for matrix math

from simuglue.transform.linear import apply_transform
from simuglue.mechanics.voigt import stress_tensor_to_voigt6

from .config import Config, load_config
from .registry import get_backend, is_done, RelaxResult

# -------------------- helpers --------------------
def _validate(cfg: Config) -> None:
    if not cfg.target_matrix:
        raise ValueError("Config 'target_matrix' is missing.")
    if not cfg.steps or cfg.steps < 1:
        raise ValueError("Config 'steps' must be >= 1.")

def _copy_common_files(cfg: Config) -> None:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    if not cfg.common_files: return
    target_base = cfg.workdir / cfg.common_path
    target_base.mkdir(parents=True, exist_ok=True)
    for src in cfg.common_files:
        src = Path(src)
        dst = target_base / src.name
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)

def _dump_result_json(case_dir: Path, step: int, res: RelaxResult) -> None:
    s6 = stress_tensor_to_voigt6(res.stress)
    payload = {
        "step": step,
        "energy": float(res.energy),
        "stress6": [float(x) for x in s6],
        "cell": res.cell.tolist(),
        "units": {"stress": "eV/A^3", "energy": "eV"}
    }
    (case_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

# -------------------- The Main Loop --------------------

def init_deformation(config_path: str) -> None:
    """Simple setup check."""
    cfg = load_config(config_path)
    _validate(cfg)
    _copy_common_files(cfg)
    print(f"[init] Workspace {cfg.workdir} initialized.")


def run_deformation(config_path: str) -> None:
    """
    Runs a sequential True Strain simulation.
    Step i+1 starts from the EQUILIBRIUM geometry of Step i.
    """
    cfg = load_config(config_path)
    _validate(cfg)
    _copy_common_files(cfg)
    
    backend = get_backend(cfg.backend)
    
    # 1. Read Initial Reference
    # We maintain 'current_atoms' in memory. It updates after every step.
    current_atoms = backend.read_data(cfg) 

    # 2. Calculate Incremental Deformation Gradient (F_inc)
    # Logic: F_total = (F_inc)^N  =>  ln(F_total) = N * ln(F_inc)
    # So: F_inc = exp( ln(F_total) / N )
    
    target_F = np.array(cfg.target_matrix, dtype=float)
    steps = int(cfg.steps)
    
    # Matrix Logarithm -> Divide by Steps -> Matrix Exponential
    # This gives the small deformation matrix to apply at each step
    F_inc = expm(logm(target_F) / steps) 
    
    print(f"Target F:\n{target_F}")
    print(f"Incremental F (will be applied {steps} times):\n{F_inc}")

    # 3. The Sequential Loop
    for i in range(1, steps + 1):
        cid = f"step_{i}"
        case_dir = cfg.workdir / cid
        case_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[deformation] --- Starting Step {i}/{steps} ---")

        # A. DEFORM: Apply F_inc to the CURRENT (previously relaxed) atoms
        # This is the definition of True Strain (updating reference frame)
        deformed_atoms = apply_transform(current_atoms, F_inc)

        # B. PREPARE: Write files
        backend.prepare_case(case_dir, deformed_atoms, cfg)

        # C. RUN: Execute LAMMPS
        if not is_done(case_dir):
            try:
                backend.run_case(case_dir, cfg)
            except Exception as e:
                print(f"Simulation crashed at step {i}: {e}")
                break
        
        # D. PARSE: Get energy/stress
        try:
            res = backend.parse_case(case_dir, cfg)
            _dump_result_json(case_dir, i, res)
            
            if np.isnan(res.energy):
                print("Structure failed (NaN energy). Stopping.")
                break
                
        except Exception as e:
            print(f"Failed to parse results at step {i}: {e}")
            break

        # E. UPDATE STATE: The Most Important Part
        # We read the 'final_str.data' created by LAMMPS to capture the RELAXED state.
        final_str_path = case_dir / "final_str.data"
        
        if final_str_path.exists():
            # We create a temporary config to tell backend where to look
            cfg.lammps["data_file"] = str(final_str_path)
            current_atoms = backend.read_data(cfg)
            
            # This updates 'current_atoms' to the relaxed structure
            # ready for the next iteration.
            current_atoms = backend.read_data(cfg)
            print(f"Step {i} complete. Updated reference structure from {final_str_path.name}")
        else:
            print(f"Critical Error: {final_str_path} not found. Cannot proceed to next step.")
            break

    print("\n[deformation] Sequence finished.")

