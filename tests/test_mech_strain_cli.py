from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
import pytest

from simuglue.mechanics.kinematics import strain_from_F


def run_strain(tmp_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run `sgl mech strain` and fail loudly if it errors."""
    cmd = ["sgl", "mech", "strain", *args]
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        input=stdin,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def _parse_full(s: str) -> np.ndarray:
    """Parse 'a b c; d e f; g h i' into 3x3."""
    parts = s.strip().split(";")
    rows = []
    for row in parts:
        row = row.strip()
        if not row:
            continue
        rows.append([float(x) for x in row.split()])
    arr = np.array(rows, float)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3, got {arr.shape}")
    return arr


def _parse_voigt_to_full(s: str) -> np.ndarray:
    """Parse 'exx eyy ezz gyz gxz gxy' into symmetric 3x3."""
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != 6:
        raise ValueError(f"Expected 6 Voigt components, got {len(vals)}")
    exx, eyy, ezz, gyz, gxz, gxy = vals
    E = np.array(
        [
            [exx,     gxy / 2.0, gxz / 2.0],
            [gxy / 2.0, eyy,     gyz / 2.0],
            [gxz / 2.0, gyz / 2.0, ezz],
        ],
        float,
    )
    return E


def _approx(A: np.ndarray, B: np.ndarray, r=1e-10, a=1e-10):
    assert np.allclose(A, B, rtol=r, atol=a), f"\nA=\n{A}\nB=\n{B}"


# Define a few representative deformation gradients

F_CASES = {
    # small uniaxial stretch in x
    "uniax_small": np.diag([1.01, 1.0, 1.0]),
    # simple shear in xy
    "shear_xy": np.array(
        [
            [1.0, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    # mixed normal + shear, but still reasonable
    "mixed": np.array(
        [
            [1.02, 0.03, 0.0],
            [0.00, 0.98, 0.02],
            [0.00, 0.00, 1.01],
        ]
    ),
}


@pytest.mark.parametrize("name, F", F_CASES.items())
@pytest.mark.parametrize("measure", ["engineering", "green-lagrange", "hencky"])
def test_mech_strain_full_matches_core(tmp_path: Path, name: str, F: np.ndarray, measure: str):
    """
    For several deformation gradients and all measures, check that
    `sgl mech strain` with --out-kind full matches strain_from_F.
    """
    # Reference strain from core kinematics
    E_ref = strain_from_F(F, measure=measure)

    # Build F input text: 'F11 F12 F13; ...'
    F_text = "; ".join(" ".join(str(float(x)) for x in row) for row in F)

    res = run_strain(
        tmp_path,
        [
            "--measure",
            measure,
            "--out-kind",
            "full",
        ],
        stdin=F_text,
    )

    E_cli = _parse_full(res.stdout)

    # Symmetrize both (Hencky & GL are symmetric; eng is defined sym)
    E_ref_sym = 0.5 * (E_ref + E_ref.T)
    E_cli_sym = 0.5 * (E_cli + E_cli.T)

    _approx(E_cli_sym, E_ref_sym, r=1e-9, a=1e-9)


@pytest.mark.parametrize("name, F", F_CASES.items())
@pytest.mark.parametrize("measure", ["engineering", "green-lagrange", "hencky"])
def test_mech_strain_voigt_matches_core(tmp_path: Path, name: str, F: np.ndarray, measure: str):
    """
    Same as above, but using --out-kind voigt and reconstructing E
    from the Voigt output.
    """
    E_ref = strain_from_F(F, measure=measure)
    E_ref_sym = 0.5 * (E_ref + E_ref.T)

    F_text = "; ".join(" ".join(str(float(x)) for x in row) for row in F)

    res = run_strain(
        tmp_path,
        [
            "--measure",
            measure,
            "--out-kind",
            "voigt",
        ],
        stdin=F_text,
    )

    E_cli = _parse_voigt_to_full(res.stdout)
    E_cli_sym = 0.5 * (E_cli + E_cli.T)

    _approx(E_cli_sym, E_ref_sym, r=1e-9, a=1e-9)

