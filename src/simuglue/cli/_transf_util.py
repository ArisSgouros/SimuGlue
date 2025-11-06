from __future__ import annotations
import numpy as np

def _parse_3x3(s: str) -> np.ndarray:
    vals = [float(x) for x in s.replace(";", " ").split()]
    if len(vals) != 9:
        raise ValueError("3×3 input needs 9 numbers")
    M = np.array(vals, float).reshape(3, 3)
    return M
