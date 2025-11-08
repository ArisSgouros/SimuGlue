from __future__ import annotations
from typing import Iterable
import numpy as np

def parse_3x3(text: str) -> np.ndarray:
    """
    Parse 9 numbers (with optional ';' between rows) into a 3x3 matrix.
    """
    vals = [float(x) for x in text.replace(";", " ").split()]
    if len(vals) != 9:
        raise ValueError("3x3 matrix input needs 9 numbers.")
    return np.array(vals, float).reshape(3, 3)


def ensure_symmetric(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Return symmetrized matrix if nearly symmetric, else raise.
    """
    M = np.array(M, float)
    if not np.allclose(M, M.T, atol=tol):
        raise ValueError(
            "Tensor must be symmetric; got noticeably asymmetric input."
        )
    return 0.5 * (M + M.T)


def format_3x3(M: np.ndarray, precision: int = 12) -> str:
    """
    Format a 3x3 matrix as: 'M11 M12 M13; M21 M22 M23; M31 M32 M33'
    """
    M = np.array(M, float)
    M = np.round(M, precision)
    rows = [" ".join(f"{x:.{precision}g}" for x in row) for row in M]
    return "; ".join(rows)

