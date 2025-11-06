

from __future__ import annotations
import numpy as np

def _sqrtm_spd(A: np.ndarray) -> np.ndarray:
    """Principal square root of a symmetric positive-(semi)definite matrix."""
    w, V = np.linalg.eigh(A)
    if np.any(w < -1e-12):
        raise ValueError("Matrix for sqrt has negative eigenvalues.")
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T

def _expm_sym(A: np.ndarray) -> np.ndarray:
    """Matrix exponential for symmetric matrix via eigendecomposition."""
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T
