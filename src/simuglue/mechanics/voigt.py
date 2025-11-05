from __future__ import annotations
from typing import Iterable, Union, List
import numpy as np

# 1-based Voigt: 1=xx,2=yy,3=zz,4=yz,5=xz,6=xy
_NAME_TO_VOIGT1 = {"xx":1, "yy":2, "zz":3, "yz":4, "xz":5, "xy":6}
_VOIGT1_TO_NAME = {v:k for k,v in _NAME_TO_VOIGT1.items()}

__all__ = [
    "normalize_components_to_voigt1",
    "stress_tensor_to_voigt6",
    "voigt1_to_name",
    "name_to_voigt1",
    "voigt_to_cart",
]

def name_to_voigt1(name: str) -> int:
    s = name.strip().lower()
    if s not in _NAME_TO_VOIGT1:
        raise ValueError(f"Unknown component '{name}' (use xx,yy,zz,xy,xz,yz)")
    return _NAME_TO_VOIGT1[s]

def voigt1_to_name(i: int) -> str:
    if i not in _VOIGT1_TO_NAME:
        raise ValueError("Voigt index must be 1..6")
    return _VOIGT1_TO_NAME[i]

def normalize_components_to_voigt1(components: Iterable[Union[int,str]]) -> List[int]:
    out: List[int] = []
    for c in components:
        if isinstance(c, int):
            if 1 <= c <= 6: out.append(c)
            else: raise ValueError(f"Voigt index must be 1..6, got {c}")
        else:
            s = str(c).strip().lower()
            if s.isdigit():
                v = int(s)
                if 1 <= v <= 6: out.append(v)
                else: raise ValueError(f"Voigt index must be 1..6, got {c}")
            else:
                out.append(name_to_voigt1(s))
    return out

def stress_tensor_to_voigt6(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    return np.array([S[0,0], S[1,1], S[2,2], S[1,2], S[0,2], S[0,1]], float)

# FIXIT: rm redudant functions
def voigt_to_cart(v: Sequence[float]) -> np.ndarray:
    """
    Convert a 6×1 array [xx, yy, zz, yz, xz, xy] to a 3×3 matrix:
        [[xx, xy, xz],
         [0 , yy, yz],
         [0 , 0 , zz]]
    """
    if len(v) != 6:
        raise ValueError("Expected 6 elements: [xx, yy, zz, yz, xz, xy].")
    xx, yy, zz, yz, xz, xy = map(float, v)
    return np.array([
        [xx, xy, xz],
        [0.0, yy, yz],
        [0.0, 0.0, zz],
    ], dtype=float)


# FIXIT: rm redudant functions
def voigt_to_cart_test(v: Sequence[float]) -> np.ndarray:
    """
    Convert a 6×1 array [xx, yy, zz, yz, xz, xy] to a 3×3 matrix:
        [[xx, xy, xz],
         [0 , yy, yz],
         [0 , 0 , zz]]
    """
    if len(v) != 6:
        raise ValueError("Expected 6 elements: [xx, yy, zz, yz, xz, xy].")
    xx, yy, zz, yz, xz, xy = map(float, v)
    return np.array([
        [xx  , xy/2, xz/2],
        [xy/2, yy  , yz/2],
        [xz/2, yz/2, zz  ],
    ], dtype=float)
    #return np.array([
    #    [xx  , xy, xz],
    #    [0., yy  , yz],
    #    [0., 0., zz  ],
    #], dtype=float)

def parse_F_from_voigt_str_sym(s: str) -> np.ndarray:
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
    F = voigt_to_cart_test(Fv)

    print("strain: ", Fv)
    print("F: ", F)

    if F.shape != (3, 3):
        raise ValueError(f"Transformer must be 3x3, got {F.shape}")
    return F


import numpy as np
from scipy.linalg import sqrtm

def symmetric_equivalent_from_upper_tri(F_ut):
    # Right Cauchy–Green
    C = F_ut.T @ F_ut
    # Symmetric stretch tensor U
    U = sqrtm(C).real
    return U


import numpy as np
from scipy.linalg import sqrtm, qr

def symmetric_to_upper_tri(U):
    C = U @ U
    C = 0.5 * (C + C.T)
    L = np.linalg.cholesky(C)   # C = L L^T, L lower with diag > 0
    return L.T                  # upper with diag > 0



