

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

def _logm_spd(U: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    """
    Matrix logarithm for symmetric positive-definite matrices.

    Inverse of _expm_sym for SPD inputs:
      if U = exp(E) with E symmetric, then logm_spd(U) = E.
    """
    U = 0.5 * (U + U.T)
    w, v = np.linalg.eigh(U)

    if np.any(w <= tol):
        raise ValueError(f"_logm_spd: matrix not SPD (eigs: {w})")

    L = (v * np.log(w)) @ v.T
    return 0.5 * (L + L.T)
