from __future__ import annotations
from pathlib import Path
from typing import Literal
import sys
import numpy as np

from simuglue.mechanics.kinematics import strain_from_F

MeasureKind = Literal["engineering", "green-lagrange", "hencky"]
StrainOutKind = Literal["full", "voigt"]


def _parse_F_3x3(text: str) -> np.ndarray:
    """
    Parse deformation gradient F from plain text:
      'F11 F12 F13; F21 F22 F23; F31 F32 F33'
    or 9 space-separated numbers.

    No symmetry constraints: F may include rotation, shear, etc.
    """
    s = text.strip()
    if not s:
        raise ValueError("Empty deformation gradient input.")

    vals = [float(x) for x in s.replace(";", " ").split()]
    if len(vals) != 9:
        raise ValueError("F input must have 9 numbers.")
    F = np.array(vals, float).reshape(3, 3)
    return F


def _format_strain_full(E: np.ndarray, precision: int) -> str:
    """
    Format E as:
      'exx exy exz; eyx eyy eyz; ezx ezy ezz'
    """
    E = np.array(E, float)
    E = 0.5 * (E + E.T)
    E = np.round(E, precision)
    rows = [" ".join(f"{x:.{precision}g}" for x in row) for row in E]
    return "; ".join(rows)


def _format_strain_voigt(E: np.ndarray, precision: int) -> str:
    """
    Format symmetric E as Voigt:
      'exx eyy ezz gyz gxz gxy'
    using:
      exx = E[0,0], eyy = E[1,1], ezz = E[2,2],
      gyz = 2 E[1,2], gxz = 2 E[0,2], gxy = 2 E[0,1]
    """
    E = np.array(E, float)
    E = 0.5 * (E + E.T)
    E = np.round(E, precision)

    exx = E[0, 0]
    eyy = E[1, 1]
    ezz = E[2, 2]
    gyz = 2.0 * E[1, 2]
    gxz = 2.0 * E[0, 2]
    gxy = 2.0 * E[0, 1]

    vals = [exx, eyy, ezz, gyz, gxz, gxy]
    return " ".join(f"{x:.{precision}g}" for x in vals)


def convert_F_to_strain_stream(
    input_source: str,
    measure: MeasureKind,
    out_kind: StrainOutKind,
    precision: int,
    output_target: str,
) -> None:
    """
    Read F, compute strain E via chosen measure, and write E.

    - input_source: path or '-' for stdin
    - measure: 'engineering' | 'green-lagrange' | 'hencky'
    - out_kind: 'full' -> 3x3 text, 'voigt' -> Voigt 6
    - precision: decimals
    - output_target: path or '-' for stdout
    """
    # read
    if input_source == "-":
        text = sys.stdin.read()
    else:
        text = Path(input_source).read_text(encoding="utf-8")

    # parse F
    F = _parse_F_3x3(text)

    # compute strain
    E = strain_from_F(F, measure=measure)

    # format
    if out_kind == "full":
        out = _format_strain_full(E, precision)
    else:
        out = _format_strain_voigt(E, precision)

    if not out.endswith("\n"):
        out += "\n"

    # write
    if output_target == "-":
        sys.stdout.write(out)
    else:
        Path(output_target).write_text(out, encoding="utf-8")

