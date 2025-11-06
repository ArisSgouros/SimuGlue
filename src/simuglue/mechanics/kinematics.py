# src/simuglue/mechanics/kinematics.py
import numpy as np
from simuglue.mechanics.linalg import _sqrtm_spd, _expm_sym

def defgrad_from_strain(E: np.ndarray, measure: str) -> np.ndarray:
    """
    Convert symmetric strain tensor E to deformation gradient F.
    Conventions:
      - engineering: F = I + E (small-strain approximation, no rotation)
      - green-lagrange: E = 0.5*(C - I), C = F^T F ⇒ F = sqrtm(I + 2E) (no rotation)
      - hencky: E = log U, U = exp(E), F = U (no rotation)
    """
    E = 0.5 * (E + E.T)  # ensure symmetric
    I = np.eye(3)
    if measure == "engineering":
        return I + E
    elif measure == "green-lagrange":
        C = I + 2.0 * E
        return _sqrtm_spd(C)
    elif measure == "hencky":
        U = _expm_sym(E)
        return U
    else:
        raise ValueError(f"Unknown measure: {measure}")

