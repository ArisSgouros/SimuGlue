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

