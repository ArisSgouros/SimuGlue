from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

from simuglue.mechanics.voigt import normalize_components_to_voigt1
from ase import units

from .config import load_config
from .registry import make_case_id



def _vpos(j_1based: int) -> int:
    """Map Voigt index 1..6 → 0..5."""
    return j_1based - 1

def _load_s6(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Path not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "stress6" not in data:
        raise KeyError(f"'stress6' missing in {path}")
    s6 = np.array(data["stress6"], float)
    if s6.shape != (6,):
        raise ValueError(f"Invalid stress6 shape in {path}: {s6.shape}")
    if not np.isfinite(s6).all():
        raise ValueError(f"Non-finite stress6 values in {path}")
    return s6

def _load_cell(path: Path) -> np.ndarray:
    cell = np.asarray(json.loads(Path(path).read_text(encoding="utf-8"))["cell"], float)
    if cell.shape != (3, 3): raise ValueError(f"cell shape {cell.shape}, expected 3x3")
    return cell

def post_deformation(config_path: str, *, outfile: str | None = None) -> Dict[str, object]:

    """
    Scans all results.json in the workdir.
    Aggregates raw stress values into arrays for each pulling direction.
    Saves a JSON file containing the full curve data.
    """
    cfg = load_config(config_path)
    # Get components (e.g., 1 for XX, 2 for YY) and sort strains for a smooth curve
    components = normalize_components_to_voigt1(cfg.components)
    strains = sorted([float(eps) for eps in cfg.strains])

    # 1. Get thickness from the reference state for 2D conversion
    ref_path = cfg.workdir / "run.ref" / "result.json"
    cell_ref = _load_cell(ref_path)
    
    # For 2D Pa·m conversion
    thickness_angstrom = float(np.linalg.norm(cell_ref[2]))
    out = {}
    for i in components:
        strain_axis = []
        stress_gpa_axis = []


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
                # val_gpa = val_evA3 / units.GPa

                  # --- FAILURE HANDLING ---
                # If stress becomes negative or drops by more than 80% suddenly,
                # we stop recording this curve as the material has failed.
                #if len(stress_gpa_axis) > 0:
                #    prev_stress = stress_gpa_axis[-1]
                #    if val_gpa < 0 or val_gpa < (0.2 * prev_stress):
                #        print(f"[deformation] Failure detected at strain {eps}. Stopping curve {i}.")
                #        break
                # -------------------------

                strain_axis.append(eps)
                stress_gpa_axis.append(float(val_evA3))
    

            except (KeyError, ValueError) as exc:
                print(f"[deformation/post] Error parsing {cid}: {exc}", file=sys.stderr)
                continue

        stress_results = np.array(stress_gpa_axis, float)
     

        # ------------------------------------------------------------------
        # Unit conversion for export
        # ------------------------------------------------------------------
        units_deform = cfg.output.get("units_deform", "GPa")

        converters = {
            "gpa":  1.0 / units.GPa,
            "pa":   1.0 / units.Pascal,
            "kbar": 1.0 / (1000.0 * units.bar),
            "pa m": (1.0 / units.Pascal) * thickness_angstrom * 1e-10,
        }

        key = units_deform.lower()
        try:
            conv = converters[key]
        except KeyError:
            raise ValueError(
                f"Unsupported units_deform={units_deform!r}; supported: GPa, Pa, kbar, pa m."
            )

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        
        stress_array = stress_results * conv
        stress_custom_list = stress_array.tolist()

        out[f"direction_{i}"] = {
            "strain": strain_axis,
            "stress_3d_gpa": stress_results.tolist(),
            "stress_custom_units": stress_custom_list,
            "units": {"strain": "-", "stress": "GPa", "stress_custom": units_deform},
            "meta": {
                "workdir": str(cfg.workdir),
                "backend": cfg.backend,
                "base_units": "eV/Å^3",
            },
        }

    out_name = outfile or cfg.output.get("deform_json", "deform.json")
    (cfg.workdir / out_name).write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out


