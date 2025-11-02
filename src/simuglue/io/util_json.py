# simuglue/qe_cli.py
from __future__ import annotations
import sys, json, argparse, math
import copy as cp
from pathlib import Path
from glob import glob
from typing import Iterable, Callable, Optional

# Write json file to path
def write_json(path: Path, obj, indent: int | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

# Enforce values rounding/snap:
def _snap_and_round(x: float, ndigits: int, snap_tol: float) -> float:
    """Snap values very close to an integer, then round; avoid -0.0."""
    if not math.isfinite(x):
        return x
    n = round(x)
    if math.isclose(x, n, rel_tol=0.0, abs_tol=snap_tol):
        x = float(n)
    x = round(x, ndigits)
    return 0.0 if x == 0 else x  # normalize -0.0 -> 0.0

def _map_numbers(obj, fn):
    """Recursively apply `fn` to all floats in nested dict/list structures."""
    if isinstance(obj, dict):
        return {k: _map_numbers(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_numbers(v, fn) for v in obj]
    if isinstance(obj, float):
        return fn(obj)
    return obj

def sanitize_json(obj, ndigits, snap_tol):
    obj_sanitized = cp.copy(obj)
    return _map_numbers(obj_sanitized, lambda v: _snap_and_round(v, ndigits, snap_tol))
