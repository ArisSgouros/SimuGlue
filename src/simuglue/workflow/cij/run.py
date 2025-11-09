# src/simuglue/workflows/cij/run.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from simuglue.transform.linear import apply_transform
from simuglue.mechanics.voigt import normalize_components_to_voigt1, stress_tensor_to_voigt6

from .config import Config, load_config
from .registry import get_backend, is_done, mark_done, RelaxResult


def _validate(cfg: Config, components: list[int], strains: list[float]) -> None:
    if not components:
        raise ValueError("Config 'components' is empty.")
    if not strains:
        raise ValueError("Config 'strains' is empty.")
    if any(abs(eps) < 1e-12 for eps in strains):
        raise ValueError("Config 'strains' contains near-zero values.")


def F_from_component(dir_idx: int, s: float) -> np.ndarray:
    """
    Build F for Voigt dir: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy.
    s = ±up (engineering strain / shear).
    """
    F = np.eye(3)
    if dir_idx == 1:      # xx
        F[0, 0] += s
    elif dir_idx == 2:    # yy
        F[1, 1] += s
    elif dir_idx == 3:    # zz
        F[2, 2] += s
    elif dir_idx == 4:    # yz: y' += s * z
        F[1, 2] += s
    elif dir_idx == 5:    # xz: x' += s * z
        F[0, 2] += s
    elif dir_idx == 6:    # xy: x' += s * y
        F[0, 1] += s
    else:
        raise ValueError("dir_idx must be 1..6")
    return F


def make_case_id(i: int, eps: float) -> str:
    # Shared between run.py and post.py
    return f"run.c{i}_eps{eps:g}"


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


def _dump_result_json(case_dir: Path, *, kind: str, i: int | None,
                      eps: float | None, res: RelaxResult) -> None:
    s6 = stress_tensor_to_voigt6(res.stress)
    payload = {
        "kind": kind,        # "ref" or "sample"
        "i": i,
        "eps": eps,
        "energy": float(res.energy),
        "stress6": [float(x) for x in s6],
        "units": {"stress": "eV/\u00c5^3", "energy": "eV", "strain": "-"},
    }
    (case_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def run_cij(config_path: str) -> None:
    """
    Generate + run all Cij deformation cases.

    Side effects:
    - Creates subdirs in cfg.workdir:
        run.ref/
        run.c{i}_eps{eps}/
    - Runs backend in each dir.
    - Writes per-case result.json with stress6, energy, etc.
    - Marks finished cases with .done

    This function does NOT compute the final Cij; use post_cij() for that.
    """
    cfg = load_config(config_path)
    components = normalize_components_to_voigt1(cfg.components)
    strains = [float(eps) for eps in cfg.strains]
    _validate(cfg, components, strains)

    _copy_common_files(cfg)

    backend = get_backend(cfg.backend)

    # --- reference case ---
    atoms_ref = backend.read_data(cfg.data_file, cfg)
    ref_dir = cfg.workdir / "run.ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    if not is_done(ref_dir):
        backend.prepare_case(ref_dir, atoms_ref, cfg)
        backend.run_case(ref_dir, atoms_ref, cfg)
        res_ref = backend.parse_case(ref_dir, atoms_ref, cfg)
        _dump_result_json(ref_dir, kind="ref", i=None, eps=None, res=res_ref)
        mark_done(ref_dir)

    # --- deformed cases ---
    total = len(components) * len(strains)
    k = 0

    for eps in strains:
        for i in components:
            k += 1
            cid = make_case_id(i, eps)
            case_dir = cfg.workdir / cid
            case_dir.mkdir(parents=True, exist_ok=True)

            print(f"[cij/run] ({k}/{total}) i={i} eps={eps:g} -> {cid}")

            if is_done(case_dir):
                print(f"[cij/run] skip {cid} (already done)")
                continue

            F = F_from_component(i, eps)
            atoms_def = apply_transform(atoms_ref, F)

            if cfg.output.get("save_traj", True):
                # backend decides real format; xyz is just a conventional name here
                backend.write_data(case_dir / "deformed.xyz", atoms_def, cfg)

            backend.prepare_case(case_dir, atoms_def, cfg)
            backend.run_case(case_dir, atoms_def, cfg)
            res = backend.parse_case(case_dir, atoms_def, cfg)
            _dump_result_json(case_dir, kind="sample", i=i, eps=eps, res=res)
            mark_done(case_dir)

