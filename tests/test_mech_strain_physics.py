from __future__ import annotations
import subprocess
from pathlib import Path

import numpy as np
import pytest


def run_strain(tmp_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
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


def _approx(A: np.ndarray, B: np.ndarray, r=1e-10, a=1e-10):
    assert np.allclose(A, B, rtol=r, atol=a), f"\nA=\n{A}\nB=\n{B}"


# Same deformation gradients as before

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
    # mixed normal + shear
    "mixed": np.array(
        [
            [1.02, 0.03, 0.0],
            [0.00, 0.98, 0.02],
            [0.00, 0.00, 1.01],
        ]
    ),
}


# --- Analytic strain definitions (no use of simuglue internals) ---


def E_engineering(F: np.ndarray) -> np.ndarray:
    """eps = 0.5((F - I) + (F - I)^T)."""
    F = np.asarray(F, float)
    I = np.eye(3)
    return 0.5 * ((F - I) + (F - I).T)


def E_green_lagrange(F: np.ndarray) -> np.ndarray:
    """E = 0.5(F^T F - I)."""
    F = np.asarray(F, float)
    I = np.eye(3)
    C = F.T @ F
    return 0.5 * (C - I)


def E_hencky(F: np.ndarray) -> np.ndarray:
    """
    Hencky (logarithmic) strain:
      E = log U,  U = sqrt(C),  C = F^T F.

    Implemented via eigen-decomposition:
      C = Q diag(w) Q^T, w > 0
      U = Q diag(sqrt(w)) Q^T
      E = Q diag(log(sqrt(w))) Q^T = 0.5 Q diag(log w) Q^T
    """
    F = np.asarray(F, float)
    C = F.T @ F
    # symmetrize numerically
    C = 0.5 * (C + C.T)
    w, Q = np.linalg.eigh(C)
    if np.any(w <= 0.0):
        raise ValueError(f"Non-SPD C in Hencky strain test (eigs: {w})")
    log_w = np.log(w)
    E = (Q * (0.5 * log_w)) @ Q.T
    # enforce symmetry
    return 0.5 * (E + E.T)


# --- Physics tests: CLI vs analytic formulas ---


@pytest.mark.parametrize("name,F", F_CASES.items())
def test_strain_physics_engineering(tmp_path: Path, name: str, F: np.ndarray):
    """
    Check that `sgl mech strain --measure engineering` matches
    eps = 0.5((F - I) + (F - I)^T) for several F cases.
    """
    E_expected = E_engineering(F)

    F_text = "; ".join(" ".join(str(float(x)) for x in row) for row in F)

    res = run_strain(
        tmp_path,
        ["--measure", "engineering", "--out-kind", "full"],
        stdin=F_text,
    )
    E_cli = _parse_full(res.stdout)

    # Both should be symmetric by definition; enforce before compare
    E_cli_sym = 0.5 * (E_cli + E_cli.T)
    E_exp_sym = 0.5 * (E_expected + E_expected.T)

    _approx(E_cli_sym, E_exp_sym, r=1e-10, a=1e-10)


@pytest.mark.parametrize("name,F", F_CASES.items())
def test_strain_physics_green_lagrange(tmp_path: Path, name: str, F: np.ndarray):
    """
    Check that `sgl mech strain --measure green-lagrange` matches
    E = 0.5(F^T F - I).
    """
    E_expected = E_green_lagrange(F)

    F_text = "; ".join(" ".join(str(float(x)) for x in row) for row in F)

    res = run_strain(
        tmp_path,
        ["--measure", "green-lagrange", "--out-kind", "full"],
        stdin=F_text,
    )
    E_cli = _parse_full(res.stdout)

    E_cli_sym = 0.5 * (E_cli + E_cli.T)
    E_exp_sym = 0.5 * (E_expected + E_expected.T)

    _approx(E_cli_sym, E_exp_sym, r=1e-10, a=1e-10)


@pytest.mark.parametrize("name,F", F_CASES.items())
def test_strain_physics_hencky(tmp_path: Path, name: str, F: np.ndarray):
    """
    Check that `sgl mech strain --measure hencky` matches
    Hencky strain definition: E = log U, U = sqrt(F^T F).
    """
    E_expected = E_hencky(F)

    F_text = "; ".join(" ".join(str(float(x)) for x in row) for row in F)

    res = run_strain(
        tmp_path,
        ["--measure", "hencky", "--out-kind", "full"],
        stdin=F_text,
    )
    E_cli = _parse_full(res.stdout)

    E_cli_sym = 0.5 * (E_cli + E_cli.T)
    E_exp_sym = 0.5 * (E_expected + E_expected.T)

    # Hencky involves eigen-decompositions; loosen tolerance slightly
    _approx(E_cli_sym, E_exp_sym, r=1e-9, a=1e-9)

