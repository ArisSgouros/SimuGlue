from __future__ import annotations
from pathlib import Path
from typing import Literal
import sys
import numpy as np

from simuglue.mechanics.kinematics import defgrad_from_strain

StrainKind = Literal["auto", "full", "voigt"]
MeasureKind = Literal["engineering", "green-lagrange", "hencky"]


def _parse_3x3(s: str, tol: float = 1e-10) -> np.ndarray:
    """
    Parse 9 numbers (with optional ';' between rows) into a symmetric 3x3.

    Rejects noticeably asymmetric input: this helper is for true strain tensors.
    """
    vals = [float(x) for x in s.replace(";", " ").split()]
    if len(vals) != 9:
        raise ValueError("3x3 strain input needs 9 numbers.")
    M = np.array(vals, float).reshape(3, 3)

    if not np.allclose(M, M.T, atol=tol):
        raise ValueError(
            "Strain tensor must be symmetric; got asymmetric input. "
            "If you want to prescribe rotation, provide the deformation "
            "gradient F directly in your deformation tool."
        )

    return 0.5 * (M + M.T)


def _parse_voigt6(s: str) -> np.ndarray:
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


def parse_strain(text: str, kind: StrainKind = "auto") -> np.ndarray:
    """
    Parse a 3x3 strain tensor E from plain text.

    kind:
      - 'full'  : 3x3 (9 numbers, ';' optional between rows), must be symmetric
      - 'voigt' : exx eyy ezz gyz gxz gxy
      - 'auto'  : 9 -> full, 6 -> voigt, else error
    """
    s = text.strip()
    if not s:
        raise ValueError("Empty strain input.")

    if kind == "full":
        return _parse_3x3(s)

    if kind == "voigt":
        return _parse_voigt6(s)

    # auto-detect
    vals = s.replace(";", " ").split()
    if len(vals) == 9:
        return _parse_3x3(s)
    if len(vals) == 6:
        return _parse_voigt6(s)

    raise ValueError(
        "Could not infer strain format in auto mode: expected 9 (3x3) or 6 (Voigt) numbers."
    )

# REFACTOR: fmt_3x3
def format_F(F: np.ndarray, precision: int = 12) -> str:
    """
    Format F as: 'F11 F12 F13; F21 F22 F23; F31 F32 F33'
    """
    F = np.array(F, float)
    F = np.round(F, precision)
    rows = [" ".join(f"{x:.{precision}g}" for x in row) for row in F]
    return "; ".join(rows)


def convert_strain_to_F_stream(
    input_source: str,
    input_kind: StrainKind,
    measure: MeasureKind,
    precision: int,
    output_target: str,
) -> None:
    """
    CLI-facing wrapper: read E, compute F, write F.

    Assumptions:
      - E is a proper symmetric strain tensor.
      - 'green-lagrange' and 'hencky' are inverted as pure stretches (R = I).
      - 'engineering' uses F = I + E (small-strain approximation).
    """
    # read
    if input_source == "-":
        text = sys.stdin.read()
    else:
        text = Path(input_source).read_text(encoding="utf-8")

    # parse strain
    E = parse_strain(text, kind=input_kind)

    # compute F using the kinematics core (which also symmetrizes defensively)
    F = defgrad_from_strain(E, measure=measure)

    # format
    out = format_F(F, precision=precision)
    if not out.endswith("\n"):
        out += "\n"

    # write
    if output_target == "-":
        sys.stdout.write(out)
    else:
        Path(output_target).write_text(out, encoding="utf-8")

