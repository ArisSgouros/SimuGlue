from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from ase import units

from .config import load_config
from .registry import make_case_id

def _vpos(j_1based: int) -> int:
    """Map Voigt index 1..6 → 0..5."""
    return j_1based - 1

def _load_s6(path: Path) -> np.ndarray:
    """Loads the 6-component stress vector from result.json."""
    if not path.is_file():
        raise FileNotFoundError(f"Path not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.array(data["stress6"], float)

def _load_cell(path: Path) -> np.ndarray:
    """Loads the 3x3 cell matrix from result.json."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(data["cell"], float)

def post_deformation(config_path: str, *, outfile: str = "stress_strain_curve.json") -> Dict:
    """
    Scans all results.json in the workdir.
    Aggregates raw stress values into arrays for each pulling direction.
    Saves a JSON file containing the full curve data.
    """
    cfg = load_config(config_path)
    # Get components (e.g., 1 for XX, 2 for YY) and sort strains for a smooth curve
    components = [int(c) for c in cfg.components]
    strains = sorted([float(eps) for eps in cfg.strains])

    # 1. Get thickness from the reference state for 2D conversion
    ref_path = cfg.workdir / "run.ref" / "result.json"
    cell_ref = _load_cell(ref_path)
    thickness_angstrom = float(np.linalg.norm(cell_ref[2]))

    curve_results = {}

    for i in components:
        strain_axis = []
        stress_gpa_axis = []
        tension_nm_axis = []
        
        # Voigt index for the pulling direction
        idx = _vpos(i)

        for eps in strains:
            # Skip the zero-strain reference in the loop if handled separately
            if abs(eps) < 1e-12:
                continue

            cid = make_case_id(i, eps)
            case_path = cfg.workdir / cid / "result.json"

            if not case_path.exists():
                print(f"[deformation/post] Warning: Missing {cid}", file=sys.stderr)
                continue

            try:
                # Load deformed stress (eV/Å^3)
                s6_def = _load_s6(case_path)
                val_evA3 = s6_def[idx]

                # Convert to 3D GPa
                val_gpa = val_evA3 / units.GPa
                
                # Convert to 2D Tension (N/m)
                # Formula: Stress(eV/Å^3) * Thickness(Å) * 1.6021766
                val_nm = val_evA3 * thickness_angstrom * 16.021766

                # --- FAILURE HANDLING ---
                # If stress becomes negative or drops by more than 80% suddenly,
                # we stop recording this curve as the material has failed.
                if len(stress_gpa_axis) > 0:
                    prev_stress = stress_gpa_axis[-1]
                    if val_gpa < 0 or val_gpa < (0.2 * prev_stress):
                        print(f"[deformation] Failure detected at strain {eps}. Stopping curve {i}.")
                        break
                # -------------------------

                strain_axis.append(eps)
                stress_gpa_axis.append(float(val_gpa))
                tension_nm_axis.append(float(val_nm))

            except (KeyError, ValueError) as exc:
                print(f"[deformation/post] Error parsing {cid}: {exc}", file=sys.stderr)
                continue

        # Store the synchronized arrays for this component
        curve_results[f"direction_{i}"] = {
            "strain": strain_axis,
            "stress_3d_gpa": stress_gpa_axis,
            "tension_2d_Nm": tension_nm_axis,
            "units": {"strain": "-", "stress": "GPa", "tension": "N/m"}
        }

    # 2. Export the final dataset to the workdir
    output_path = cfg.workdir / outfile
    output_path.write_text(json.dumps(curve_results, indent=2), encoding="utf-8")
    
    print(f"[deformation/post] Success! Curve data saved to {output_path}")
    return curve_results
