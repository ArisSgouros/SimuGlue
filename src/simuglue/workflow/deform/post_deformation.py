from __future__ import annotations

import json
import sys
import numpy as np
from pathlib import Path
from ase import units
from scipy.linalg import logm

from .config import load_config

def _load_s6(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Path not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    s6 = np.array(data["stress6"], float)
    return s6

def post_deformation(config_path: str, *, outfile: str | None = None) -> dict:
    """
    Scans step_1, step_2... folders.
    Aggregates True Stress vs True Strain for XX, YY, and XY directions.
    """
    cfg = load_config(config_path)

    # 1. Setup True Strain Tensor
    target_F = np.array(cfg.target_matrix, dtype=float)
    E_total = logm(target_F)  # Full 3x3 Logarithmic Strain Tensor

    steps = int(cfg.steps)

    # 2. Get thickness for 2D units
    from .registry import get_backend
    backend = get_backend(cfg.backend)
    atoms_ref = backend.read_data(cfg)
    thickness_angstrom = np.linalg.norm(atoms_ref.cell[2])

    # Data lists to store all 2D in-plane components
    # Voigt indices: XX = 0, YY = 1, XY = 5
    results = {
        "strain_xx": [], "strain_yy": [], "strain_xy": [],
        "stress_xx": [], "stress_yy": [], "stress_xy": []
    }

    print(f"[post] Processing {steps} steps for multi-directional analysis...")

    # 3. Loop through steps 1..N
    for i in range(1, steps + 1):
        cid = f"step_{i}"
        case_path = cfg.workdir / cid / "result.json"

        if not case_path.exists():
            print(f"[post] Warning: Missing {cid}", file=sys.stderr)
            continue

        try:
            # Current True Strain Tensor
            E_current = (i / steps) * E_total
            
            # Load Stress (eV/A^3)
            s6 = _load_s6(case_path)

            # Store Strains (XX=0,0 | YY=1,1 | XY=0,1)
            results["strain_xx"].append(E_current[0, 0])
            results["strain_yy"].append(E_current[1, 1])
            # For engineering shear strain, you might multiply by 2. We keep true tensor shear here.
            results["strain_xy"].append(E_current[0, 1]) 

            # Store Stresses
            results["stress_xx"].append(s6[0])
            results["stress_yy"].append(s6[1])
            results["stress_xy"].append(s6[5])

        except Exception as exc:
            print(f"[post] Error parsing {cid}: {exc}", file=sys.stderr)
            continue

    # 4. Handle YAML Custom Units
    req_unit = cfg.output.get("units_cij", "gpa").lower()
    
    converters = {
        "gpa":  1.0 / units.GPa,
        "pa":   1.0 / units.Pascal,
        "kbar": 1.0 / (1000.0 * units.bar),
        "pa m": (1.0 / units.Pascal) * thickness_angstrom * 1e-10,
    }
    
    if req_unit not in converters:
        print(f"[post] Warning: Unknown unit '{req_unit}'. Defaulting to GPa.")
        req_unit = "gpa"
        
    conv_factor = converters[req_unit]

    # Convert stress arrays
    s_xx_final = (np.array(results["stress_xx"]) * conv_factor).tolist()
    s_yy_final = (np.array(results["stress_yy"]) * conv_factor).tolist()
    s_xy_final = (np.array(results["stress_xy"]) * conv_factor).tolist()

    # 5. Build Final Payload
    out = {
        "Strain_XX": results["strain_xx"],
        "Strain_YY": results["strain_yy"],
        "Strain_XY": results["strain_xy"],
        f"Stress_XX_{req_unit}": s_xx_final,
        f"Stress_YY_{req_unit}": s_yy_final,
        f"Stress_XY_{req_unit}": s_xy_final,
        "meta": {
             "workdir": str(cfg.workdir),
             "thickness_angstrom": thickness_angstrom,
             "target_matrix": target_F.tolist()
        }
    }

    # 6. Save File
    out_name = outfile or cfg.output.get("deform_json", "deform.json")
    (cfg.workdir / out_name).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"[post] Saved multi-axial curve to {out_name}")
    return out
