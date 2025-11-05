# src/simuglue/workflows/cij_run.py
from __future__ import annotations
import numpy as np

__all__ = ["deformation_gradient_from_voigt"]

def deformation_gradient_from_voigt(dir_idx: int, s: float) -> np.ndarray:
    """
    Build F for Voigt dir: 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy.
    s = ±up (engineering strain / shear).
    """
    F = np.eye(3)
    if   dir_idx == 1:  # xx
        F[0, 0] += s
    elif dir_idx == 2:  # yy
        F[1, 1] += s
    elif dir_idx == 3:  # zz
        F[2, 2] += s
    elif dir_idx == 4:  # yz: y' += s * z
        F[1, 2] += s
    elif dir_idx == 5:  # xz: x' += s * z
        F[0, 2] += s
    elif dir_idx == 6:  # xy: x' += s * y
        F[0, 1] += s
    else:
        raise ValueError("dir_idx must be 1..6")
    return F
