from __future__ import annotations
from typing import Iterable, Union, List
import numpy as np


def parse_voigt6(s: str) -> np.ndarray:
    """
    Parse Voigt: exx eyy ezz gyz gxz gxy -> symmetric 3x3 strain tensor.

    Convention: no factor 2 on shear in input; we divide by 2 internally.
    """
    vals = [float(x) for x in s.replace(";", " ").split()]
    if len(vals) != 6:
        raise ValueError("Voigt strain needs 6 numbers: exx eyy ezz gyz gxz gxy.")
    exx, eyy, ezz, gyz, gxz, gxy = vals
    return np.array(
        [
            [exx,       gxy / 2.0, gxz / 2.0],
            [gxy / 2.0, eyy,       gyz / 2.0],
            [gxz / 2.0, gyz / 2.0, ezz],
        ],
        float,
    )

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

def voigt6_to_stress_tensor(v: Sequence[float]) -> np.ndarray:
    if len(v) != 6:
        raise ValueError("Voigt-6 input must have length 6: [xx, yy, zz, yz, xz, xy].")
    xx, yy, zz, yz, xz, xy = map(float, v)
    S = np.array([
        [xx, xy, xz],
        [xy, yy, yz],
        [xz, yz, zz],
    ], dtype=float)
    return S


# REFACTOR: rm redudant
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



