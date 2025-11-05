from __future__ import annotations
import numpy as np

from simuglue.mechanics.voigt import voigt_to_cart

def parse_F_from_voigt_str(s: str) -> np.ndarray:
    """
    Build a 3x3 deformation gradient F from a string.

    - "xx yy zz yz xz xy"  (engineering shear components)
      Interprets as small strain E_voigt, returns F ≈ I + E (in cart).
    """
    s = s.strip()
    parts = [float(x) for x in s.replace(",", " ").split()]
    if len(parts) != 6:
        raise ValueError("With --voigt, provide exactly 6 numbers: 'xx yy zz yz xz xy'.")
    E = np.asarray(parts, dtype=float)
    I_voigt = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    Fv = E + I_voigt
    F = voigt_to_cart(Fv)

    if F.shape != (3, 3):
        raise ValueError(f"Transformer must be 3x3, got {F.shape}")
    return F

def parse_F_from_tensor_str(s: str) -> np.ndarray:
    """
    Build a 3x3 deformation gradient F from a string.

    - "a b c; d e f; g h i" (rows separated by ';')
      Interprets as F directly.
    """
    # Non-voigt: strictly 'a b c; d e f; g h i'
    rows = [r.strip() for r in s.split(";") if r.strip()]
    if len(rows) != 3:
        raise ValueError("Provide exactly 3 rows separated by ';', e.g. '1 0 0; 0 1 0; 0 0 1'.")
    mat: list[float] = []
    for r in rows:
        parts = [float(x) for x in r.replace(",", " ").split()]
        if len(parts) != 3:
            raise ValueError("Each row must contain exactly 3 numbers.")
        mat.extend(parts)

    strain = np.asarray(mat, dtype=float).reshape(3, 3)
    F = strain + np.eye(3)
    return F
