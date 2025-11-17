# src/simuglue/workflows/cij/post.py
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from simuglue.mechanics.voigt import normalize_components_to_voigt1
from ase import units

from .config import Config, load_config
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

def post_cij(config_path: str, *, outfile: str | None = None) -> Dict[str, object]:
    """
    Post-process finished Cij cases:
    - Reads run.ref/result.json and run.c{i}_eps{eps}/result.json
    - Computes C_ij = (S_j(def) - S_j(ref)) / eps_i
    - Aggregates over ±eps, etc.
    - Emits cfg.workdir / outfile (default from config['output']['cij_json'])

    Returns the JSON-able dict.
    """
    cfg = load_config(config_path)
    components = normalize_components_to_voigt1(cfg.components)
    strains = [float(eps) for eps in cfg.strains]

    if not components:
        raise ValueError("Config 'components' is empty.")
    if not strains:
        raise ValueError("Config 'strains' is empty.")

    ref_path = cfg.workdir / "run.ref" / "result.json"
    s6_ref = _load_s6(ref_path)
    cell_ref = _load_cell(ref_path)

    # Accumulator: (i,j) -> list of samples
    CC_all: Dict[Tuple[int, int], list[float]] = {}
    for i in components:
        for j in components:
            CC_all[i, j] = []

    for eps in strains:
        if abs(eps) < 1e-12:
            print(f"[cij/post] skip near-zero strain eps={eps}", file=sys.stderr)
            continue
        for i in components:
            cid = make_case_id(i, eps)
            case_path = cfg.workdir / cid / "result.json"

            try:
                s6_def = _load_s6(case_path)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                print(f"[cij/post] skip {cid}: {exc}", file=sys.stderr)
                continue

            for j in components:
                jpos = _vpos(j)
                C_ij = (s6_def[jpos] - s6_ref[jpos]) / eps
                CC_all[i, j].append(float(C_ij))

    # Compute mean & SEM
    CC_mean: Dict[Tuple[int, int], float] = {}
    CC_sem: Dict[Tuple[int, int], float] = {}

    for i in components:
        for j in components:
            samples = np.array(CC_all[i, j], float)
            if samples.size == 0:
                print(f"[cij/post] no samples for C[{i},{j}]", file=sys.stderr)
                CC_mean[i, j] = float("nan")
                CC_sem[i, j] = float("nan")
                continue
            CC_mean[i, j] = float(samples.mean())
            if samples.size > 1:
                CC_sem[i, j] = float(samples.std(ddof=1) / np.sqrt(samples.size))
            else:
                CC_sem[i, j] = 0.0

    # Symmetry checks (soft warning)
    for i in components:
        for j in components:
            if (j, i) in CC_mean:
                diff = abs(CC_mean[i, j] - CC_mean[j, i])
                if diff > 1e-3:
                    print(
                        f"[cij/post] note: C[{i},{j}] != C[{j},{i}] "
                        f"(diff={diff:.3f} GPa)",
                        file=sys.stderr,
                    )

    # Calculate the compliance matrix
    S6 = None  # default: no compliance

    if sorted(components) != [1, 2, 3, 4, 5, 6]:
        print(
            "Compliance calculation requires all Voigt components [1..6]. "
            f"Got components={components!r}. Skipping compliance (Sij).",
            file=sys.stderr,
        )
    else:
        # Build 6x6 stiffness matrix C in Voigt (engineering) form
        C6 = np.zeros((6, 6), float)
        for i in components:
            for j in components:
                C6[i - 1, j - 1] = CC_mean[i, j]

        # apply factor of 2 to columns of off-diagonal (shear) elements: j=3..5 -> Voigt 4..6
        for i in range(6):
            for j in range(3, 6):
                C6[i, j] *= 2.0

        # Invert to get compliance in that mixed convention
        S6 = np.linalg.inv(C6)

        # Undo shear scaling on compliance (columns 4..6)
        for i in range(6):
            for j in range(3, 6):
                S6[i, j] *= 0.5

    from ase import units

    units_cij = cfg.output.get("units_cij", "GPa")

    converters = {
        "gpa":  1.0 / units.GPa,
        "pa":   1.0 / units.Pascal,
        "kbar": 1.0 / (1000.0 * units.bar),
        "pa m": (1.0 / units.Pascal)*cell_ref[2][2]*1e-10
    }

    key = units_cij.lower()
    try:
        conv = converters[key]
    except KeyError:
        raise ValueError(
            f"Unsupported units_cij={units_cij!r}; supported: GPa, Pa, kbar, pa m."
        )

    # ------------------------------------------------------------------
    # Flatten C and S back into dict form for JSON export
    # ------------------------------------------------------------------
    C_mean_out = {}
    C_sem_out = {}
    S_out = {}

    for i in components:
        for j in components:
            key_ij = f"{i}-{j}"
            C_mean_out[key_ij] = CC_mean[i, j] * conv
            C_sem_out[key_ij] = CC_sem[i, j] * conv

            if S6 is not None:
                # Only export S if it was actually computed
                S_out[key_ij] = S6[i - 1, j - 1] / conv

    out = {
        "components": components,
        "strains": strains,
        "C_mean": C_mean_out,
        "C_sem": C_sem_out,
        "S": S_out,
        "units": {"stress": units_cij, "strain": "-"},
        "meta": {
            "workdir": str(cfg.workdir),
            "backend": cfg.backend,
            "base_units": "eV/Å^3",
        },
    }

    out_name = outfile or cfg.output.get("cij_json", "cij.json")
    (cfg.workdir / out_name).write_text(json.dumps(out, indent=2), encoding="utf-8")

    return out

