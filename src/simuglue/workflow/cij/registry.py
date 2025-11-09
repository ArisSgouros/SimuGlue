# src/simuglue/workflows/cij/registry.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Protocol, Any

import numpy as np


@dataclass(slots=True)
class RelaxResult:
    energy: float
    stress: np.ndarray  # 3x3 tensor in GPa (canonical)


class Backend(Protocol):
    def read_data(self, path: Path, cfg) -> Any: ...
    def write_data(self, path: Path, atoms, cfg) -> None: ...
    def prepare_case(self, case_dir: Path, atoms, cfg) -> None: ...
    def run_case(self, case_dir: Path, atoms, cfg) -> None: ...
    def parse_case(self, case_dir: Path, atoms, cfg) -> RelaxResult: ...


_BACKENDS: Dict[str, Backend] = {}


def register_backend(name: str):
    """Class decorator to register a backend under a given name."""
    def deco(cls):
        _BACKENDS[name] = cls()
        return cls
    return deco


def get_backend(name: str) -> Backend:
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {sorted(_BACKENDS)}"
        ) from exc


def is_done(case_dir: Path) -> bool:
    return (case_dir / ".done").exists()


def mark_done(case_dir: Path) -> None:
    (case_dir / ".done").write_text("ok\n", encoding="utf-8")

