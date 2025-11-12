# src/simuglue/workflows/cij/run.py
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np

from simuglue.transform.linear import apply_transform
from simuglue.mechanics.voigt import (
    normalize_components_to_voigt1,
    stress_tensor_to_voigt6,
)

from .config import Config, load_config
from .registry import get_backend, is_done, is_running, RelaxResult, make_case_id


# -------------------- core helpers --------------------


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

def _dump_result_json(
    case_dir: Path,
    *,
    kind: str,            # "ref" or "sample"
    i: int | None,
    eps: float | None,
    res: RelaxResult,
) -> None:
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
    (case_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _iter_cases(
    cfg: Config, components: list[int], strains: list[float]
) -> Iterator[tuple[int, float, str, Path]]:
    for eps in strains:
        for i in components:
            cid = make_case_id(i, eps)
            case_dir = cfg.workdir / cid
            yield i, eps, cid, case_dir


# -------------------- 1) init: generate cases --------------------


def init_cij(config_path: str) -> None:
    """
    Prepare reference + deformed cases.

    - Copies common files.
    - Reads reference structure.
    - Creates:
        workdir/run.ref/
        workdir/run.c{i}_eps{eps}/
    - Writes deformed structures (if enabled).
    - Calls backend.prepare_case(...) for each case.

    Does NOT:
    - run backend jobs
    - mark .done
    - write result.json
    """
    cfg = load_config(config_path)
    components = normalize_components_to_voigt1(cfg.components)
    strains = [float(eps) for eps in cfg.strains]
    _validate(cfg, components, strains)

    _copy_common_files(cfg)

    backend = get_backend(cfg.backend)

    # reference
    atoms_ref = backend.read_data(cfg)
    ref_dir = cfg.workdir / "run.ref"
    ref_dir.mkdir(parents=True, exist_ok=True)

    # we allow re-init; backend.prepare_case should be idempotent/overwrite-safe
    backend.prepare_case(ref_dir, atoms_ref, cfg)

    # deformed cases
    for i, eps, cid, case_dir in _iter_cases(cfg, components, strains):
        case_dir.mkdir(parents=True, exist_ok=True)
        F = F_from_component(i, eps)
        atoms_def = apply_transform(atoms_ref, F)
        backend.prepare_case(case_dir, atoms_def, cfg)


# -------------------- 2) run: execute jobs only --------------------


def run_cij(config_path: str) -> None:
    """
    Run all prepared cases that are not yet marked done.

    Assumes:
    - `init_cij` has been called and inputs exist.

    Behavior:
    - For each case directory (ref + deformed):
        if not .done:
            backend.run_case(case_dir, cfg)
            mark_done(case_dir)
    """
    cfg = load_config(config_path)
    components = normalize_components_to_voigt1(cfg.components)
    strains = [float(eps) for eps in cfg.strains]
    _validate(cfg, components, strains)

    backend = get_backend(cfg.backend)

    ref_dir = cfg.workdir / "run.ref"
    if not ref_dir.is_dir():
        raise RuntimeError(
            f"[cij/run] missing {ref_dir}; run init_cij() before run_cij()."
        )

    # reference
    if is_done(ref_dir):
        print(f"[cij/run] (ref) skip run.ref (done)")
    elif is_running(ref_dir):
        print(f"[cij/run] (ref) skip run.ref (running)")
    else:
        print(f"[cij/run] (ref) run  run.ref")
    backend.run_case(ref_dir, cfg)

    # deformed
    total = len(components) * len(strains)
    for k, (i, eps, cid, case_dir) in enumerate(
        _iter_cases(cfg, components, strains), start=1
    ):
        if not case_dir.is_dir():
            raise RuntimeError(
                f"[cij/run] missing case dir {case_dir}; run init_cij() first."
            )

        if is_done(case_dir):
            print(f"[cij/run] ({k}/{total}) skip {cid} (done)")
            continue
        elif is_running(case_dir):
            print(f"[cij/run] ({k}/{total}) skip {cid} (running)")
            continue

        print(f"[cij/run] ({k}/{total}) run  {cid}")
        backend.run_case(case_dir, cfg)


# -------------------- 3) parse: outputs → result.json --------------------


def parse_cij(config_path: str) -> None:
    """
    Parse all finished cases and write per-case result.json files.

    - Only parses directories where .done is present.
    - Calls backend.parse_case(case_dir, cfg) -> RelaxResult.
    - Writes:
        run.ref/result.json          (kind="ref")
        run.c{i}_eps{eps}/result.json (kind="sample")

    Safe to re-run (idempotent over the same outputs).
    """
    cfg = load_config(config_path)
    components = normalize_components_to_voigt1(cfg.components)
    strains = [float(eps) for eps in cfg.strains]
    _validate(cfg, components, strains)

    backend = get_backend(cfg.backend)

    # reference
    ref_dir = cfg.workdir / "run.ref"
    if is_done(ref_dir):
        try:
            res_ref = backend.parse_case(ref_dir, cfg)
        except Exception as exc:
            print(f"[cij/parse] skip ref: {exc}", file=sys.stderr)
        else:
            _dump_result_json(ref_dir, kind="ref", i=None, eps=None, res=res_ref)
    else:
        print("[cij/parse] skip ref (not done)", file=sys.stderr)

    # deformed
    for i, eps, cid, case_dir in _iter_cases(cfg, components, strains):
        if not is_done(case_dir):
            print(f"[cij/parse] skip {cid} (not done)", file=sys.stderr)
            continue

        try:
            res = backend.parse_case(case_dir, cfg)
        except Exception as exc:
            print(f"[cij/parse] skip {cid}: {exc}", file=sys.stderr)
            continue

        _dump_result_json(case_dir, kind="sample", i=i, eps=eps, res=res)
