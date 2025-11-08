from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
import pytest

from simuglue.mechanics.kinematics import defgrad_from_strain


def run_defgrad(tmp_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run `sgl mech defgrad` inside tmp_path, fail loudly on error."""
    cmd = ["sgl", "mech", "defgrad", *args]
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


def _parse_F_text(s: str) -> np.ndarray:
    """Parse 'a b c; d e f; g h i' into 3x3 float array."""
    parts = s.strip().split(";")
    rows = []
    for row in parts:
        row = row.strip()
        if not row:
            continue
        rows.append([float(x) for x in row.split()])
    arr = np.array(rows, float)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 F, got shape {arr.shape}")
    return arr


def test_mech_defgrad_voigt_zero_strain(tmp_path: Path):
    """
    For zero strain in Voigt form, F must be the identity.
    This is a basic end-to-end sanity check.
    """
    res = run_defgrad(
        tmp_path,
        ["--kind", "voigt"],
        stdin="0 0 0 0 0 0",
    )
    out = res.stdout.strip()
    assert out == "1 0 0; 0 1 0; 0 0 1"


@pytest.mark.parametrize("measure", ["engineering", "green-lagrange", "hencky"])
@pytest.mark.parametrize(
    "E",
    [
        # diagonal only
        np.diag([0.01, 0.00, 0.00]),
        np.diag([0.00, -0.02, 0.00]),
        # shear only (symmetric tensor)
        np.array([[0.0, 0.01, 0.0],
                  [0.01, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]),
        # mixed diagonal + shear
        np.array([[0.02, 0.01, 0.0],
                  [0.01, 0.01, 0.0],
                  [0.0,  0.0, 0.00]]),
    ],
)
def test_mech_defgrad_cli_matches_core(tmp_path: Path, measure: str, E: np.ndarray):
    """
    For various simple strain tensors and all supported measures,
    the CLI output must match defgrad_from_strain from the mechanics core.
    """
    # Reference F from core implementation
    F_ref = defgrad_from_strain(E, measure=measure)

    # Build textual input for --kind full
    # Format: 'exx exy exz; eyx eyy eyz; ezx ezy ezz'
    rows = []
    for i in range(3):
        rows.append(" ".join(str(float(x)) for x in E[i]))
    E_text = "; ".join(rows)

    # Run CLI
    res = run_defgrad(
        tmp_path,
        ["--kind", "full", "--measure", measure, "--precision", "12"],
        stdin=E_text,
    )

    F_cli = _parse_F_text(res.stdout)

    # Compare
    assert F_cli == pytest.approx(F_ref, rel=1e-10, abs=1e-10)

# --- Analytic helper functions for benchmarks ---


def F_uniaxial(lmbda: float) -> np.ndarray:
    """Uniaxial stretch along x."""
    return np.diag([lmbda, 1.0, 1.0])


def E_engineering_from_F(F: np.ndarray) -> np.ndarray:
    """
    "Engineering" / small strain: eps = sym(F - I) for homogeneous F.
    """
    I = np.eye(3)
    return 0.5 * ((F - I) + (F - I).T)


def E_green_lagrange_from_F(F: np.ndarray) -> np.ndarray:
    """
    Green–Lagrange: E = 0.5 (C - I), C = F^T F.
    """
    I = np.eye(3)
    C = F.T @ F
    return 0.5 * (C - I)


def E_hencky_from_F_uniaxial(F: np.ndarray) -> np.ndarray:
    """
    Hencky (logarithmic) strain for diagonal F:
    E_H = 0.5 ln C, with principal values ln(lambda_i).
    Assumes F is diagonal (uniaxial / pure stretch).
    """
    # For diagonal F: C = diag(lambda_i^2), so 0.5 ln C = diag(ln lambda_i)
    diagF = np.diag(F)
    if not np.allclose(F, np.diag(diagF)):
        raise ValueError("E_hencky_from_F_uniaxial expects diagonal F.")
    lmbdas = diagF
    vals = np.log(lmbdas)
    return np.diag(vals)


# --- Tests: uniaxial stretch for all measures ---


@pytest.mark.parametrize("lmbda", [1.05, 1.10])
@pytest.mark.parametrize("measure", ["engineering", "green-lagrange", "hencky"])
def test_uniaxial_known_F(tmp_path: Path, lmbda: float, measure: str):
    """
    For uniaxial stretch, construct E from known F via the chosen measure,
    feed E to the CLI, and ensure we recover F.
    """
    F_target = F_uniaxial(lmbda)

    if measure == "engineering":
        E = E_engineering_from_F(F_target)
    elif measure == "green-lagrange":
        E = E_green_lagrange_from_F(F_target)
    elif measure == "hencky":
        # Only valid for diagonal F; that's our case.
        E = E_hencky_from_F_uniaxial(F_target)
    else:
        raise RuntimeError("Unexpected measure")

    # Build input text for --kind full
    E_text = "; ".join(
        " ".join(str(float(x)) for x in row) for row in E
    )

    res = run_defgrad(
        tmp_path,
        ["--kind", "full", "--measure", measure, "--precision", "12"],
        stdin=E_text,
    )
    F_cli = _parse_F_text(res.stdout)

    assert F_cli == pytest.approx(F_target, rel=1e-8, abs=1e-8)


# --- Tests: simple shear (engineering & Green–Lagrange) ---


@pytest.mark.parametrize("gamma", [0.05, 0.10])
def test_simple_shear_engineering(tmp_path: Path, gamma: float):
    """
    Simple shear: F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]]
    Engineering/small strain: E = sym(F - I) -> off-diagonal gamma/2.
    """
    F = np.array([[1.0, gamma, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    E = E_engineering_from_F(F)

    E_text = "; ".join(
        " ".join(str(float(x)) for x in row) for row in E
    )

    res = run_defgrad(
        tmp_path,
        ["--kind", "full", "--measure", "engineering", "--precision", "12"],
        stdin=E_text,
    )
    F_cli = _parse_F_text(res.stdout)

    assert F_cli == pytest.approx(F, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("gamma", [0.05, 0.10])
def test_simple_shear_green_lagrange(tmp_path: Path, gamma: float):
    """
    Simple shear: F = [[1, gamma, 0], [0, 1, 0], [0, 0, 1]]
    Green–Lagrange: E = 0.5(F^T F - I).
    """
    F = np.array([[1.0, gamma, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0]])

    E = E_green_lagrange_from_F(F)

    E_text = "; ".join(
        " ".join(str(float(x)) for x in row) for row in E
    )

    res = run_defgrad(
        tmp_path,
        ["--kind", "full", "--measure", "green-lagrange", "--precision", "12"],
        stdin=E_text,
    )
    F_cli = _parse_F_text(res.stdout)

    assert F_cli == pytest.approx(F, rel=1e-8, abs=1e-8)

