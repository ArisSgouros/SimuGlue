# src/simuglue/workflows/cij/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any

import yaml


@dataclass(slots=True)
class Config:
    backend: str
    workdir: Path
    file_type: str
    common_files: List[Path]
    common_path: Path
    target_matrix: List[List[float]]
    steps: int 
    relax: Dict[str, Any]
    qe: Dict[str, Any]
    lammps: Dict[str, Any]
    output: Dict[str, Any]
    active_component: int | None = None


def load_config(path: str | Path) -> Config:
    p = Path(path)
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))

    common_files_raw = cfg.get("common_files", [])
    if isinstance(common_files_raw, (str, Path)):
        # allow single string
        common_files = [Path(common_files_raw)]
    else:
        common_files = [Path(x) for x in common_files_raw]

    return Config(
        backend=str(cfg["backend"]),
        workdir=Path(cfg.get("workdir", ".")),
        file_type=str(cfg.get("file_type","")),
        common_files=common_files,
        common_path=Path(cfg.get("common_path", ".")),
        target_matrix=list(cfg.get("target_matrix", [])),
        steps=int(cfg.get("steps", 0)),
        relax=dict(cfg.get("relax", {})),
        qe=dict(cfg.get("qe", {})),
        lammps=dict(cfg.get("lammps", {})),
        output=dict(cfg.get("output", {})),
        active_component=None,
    )


