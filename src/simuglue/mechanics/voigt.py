# src/simuglue/workflows/cij_run.py
from __future__ import annotations
from typing import Iterable, Union, List
import numpy as np

# Internal 1-based Voigt mapping:
# 1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
_NAME_TO_VOIGT1 = {
    "xx": 1, "yy": 2, "zz": 3,
    "yz": 4, "xz": 5, "xy": 6,
}

__all__ = [
    "normalize_components_to_voigt1",
    "stress_tensor_to_voigt6",
    "voigt1_to_name",
    "name_to_voigt1",
]

def normalize_components_to_voigt1(components: Iterable[Union[int, str]]) -> list[int]:
    """
    Accepts user-supplied components as ints (1..6) or strings ('xx','yy','zz','xy','xz','yz')
    and returns a validated list of 1-based Voigt indices following:
      1=xx, 2=yy, 3=zz, 4=yz, 5=xz, 6=xy
    Preserves user order, removes accidental whitespace, and is case-insensitive for names.
    """
    out: list[int] = []
    for c in components:
        if isinstance(c, int):
            if 1 <= c <= 6:
                out.append(c)
            else:
                raise ValueError(f"Voigt index out of range (must be 1..6): {c}")
        else:
            s = str(c).strip().lower()
            if s.isdigit():
                # If user passed "1","2",... as strings, honor them (still 1-based)
                v = int(s)
                if 1 <= v <= 6:
                    out.append(v)
                else:
                    raise ValueError(f"Voigt index (string) out of range 1..6: {c}")
            else:
                if s not in _NAME_TO_VOIGT1:
                    raise ValueError(
                        f"Unknown component name '{c}'. "
                        f"Allowed: xx, yy, zz, xy, xz, yz (case-insensitive), or 1..6."
                    )
                out.append(_NAME_TO_VOIGT1[s])
    return out

def stress_tensor_to_voigt6(S: np.ndarray) -> np.ndarray:
    # symmetrize to be safe
    S = 0.5 * (S + S.T)
    return np.array([S[0,0], S[1,1], S[2,2], S[1,2], S[0,2], S[0,1]], float)
