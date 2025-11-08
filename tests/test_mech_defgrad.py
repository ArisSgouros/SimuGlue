from __future__ import annotations
import numpy as np
import pytest
from simuglue.mechanics.linalg import _sqrtm_spd, _expm_sym

from simuglue.mechanics.kinematics import (
    defgrad_from_strain,
    _right_stretch_from_strain,
    strain_from_F,
)


def _approx(A, B, r=1e-10, a=1e-10):
    assert np.array(A).shape == np.array(B).shape
    assert np.allclose(A, B, rtol=r, atol=a)


# --- 1. Uniaxial stretch: all three measures, pure stretch (R = I) ---


@pytest.mark.parametrize("lmbda", [1.02, 1.10])
def test_uniaxial_green_lagrange_inverse(lmbda: float):
    F_true = np.diag([lmbda, 1.0, 1.0])

    # forward: E_GL = 0.5(F^T F - I)
    E = strain_from_F(F_true, "green-lagrange")

    # inverse: should recover F (pure stretch)
    F_back = defgrad_from_strain(E, "green-lagrange")
    _approx(F_back, F_true)


@pytest.mark.parametrize("lmbda", [1.02, 1.10])
def test_uniaxial_hencky_inverse(lmbda: float):
    F_true = np.diag([lmbda, 1.0, 1.0])

    # forward: Hencky strain
    E = strain_from_F(F_true, "hencky")

    # inverse: F = U = exp(E) for diagonal case
    F_back = defgrad_from_strain(E, "hencky")
    _approx(F_back, F_true)


@pytest.mark.parametrize("eps", [1e-3, -2e-3])
def test_uniaxial_engineering_small_strain(eps: float):
    # Here engineering is defined as F ≈ I + E (small strain, pure stretch)
    E = np.diag([eps, 0.0, 0.0])

    F_back = defgrad_from_strain(E, "engineering")
    F_expected = np.eye(3) + E

    _approx(F_back, F_expected)


# --- 2. _right_stretch_from_strain: consistency with defgrad_from_strain ---


@pytest.mark.parametrize("measure", ["engineering", "green-lagrange", "hencky"])
def test_right_stretch_matches_defgrad_no_rotation(measure: str):
    # Simple symmetric strain example (small + safe)
    E = np.array(
        [
            [0.01, 0.002, 0.0],
            [0.002, -0.005, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    U = _right_stretch_from_strain(E, measure)
    F = defgrad_from_strain(E, measure)

    _approx(F, U)


# --- 3. Rotation handling: GL inverse with prescribed R ---


def test_defgrad_from_strain_with_rotation_green_lagrange():
    # Build a deformation F = R U
    theta = 0.2  # some rotation around z
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,          1.0],
        ]
    )
    U = np.diag([1.05, 0.97, 1.0])
    F_true = R @ U

    # forward: E_GL from F_true
    E = strain_from_F(F_true, "green-lagrange")

    # inverse with R: should reconstruct F_true
    F_back = defgrad_from_strain(E, "green-lagrange", R=R)

    _approx(F_back, F_true, r=1e-9, a=1e-9)


# --- 4. Asymmetric E: defgrad_from_strain must ignore antisymmetric part ---


@pytest.mark.parametrize("measure", ["engineering", "green-lagrange", "hencky"])
def test_defgrad_ignores_antisymmetric_part(measure: str):
    # Symmetric "true" strain
    E_sym = np.array(
        [
            [0.01, 0.003, 0.0],
            [0.003, 0.0,   0.0],
            [0.0,   0.0,   0.0],
        ]
    )
    # Add a fake antisymmetric part (which should be ignored)
    W = np.array(
        [
            [0.0,  0.01, 0.0],
            [-0.01, 0.0, 0.0],
            [0.0,  0.0,  0.0],
        ]
    )
    E_bad = E_sym + W

    F_from_sym = defgrad_from_strain(E_sym, measure)
    F_from_bad = defgrad_from_strain(E_bad, measure)

    _approx(F_from_bad, F_from_sym)


import subprocess

@pytest.mark.parametrize("measure", ["green-lagrange"])
def test_unphysical_strain_spd_check(tmp_path: Path, measure: str):
    """
    Unphysical Green–Lagrange strain should be rejected by SPD check.

    For GL: C = I + 2E must be SPD.
    Choose E so that I + 2E has a negative eigenvalue, e.g.
      E = diag(-1, 0, 0) -> C = diag(-1, 1, 1), not SPD.
    """
    # This E is 3x3 full format: exx exy exz; eyx eyy eyz; ezx ezy ezz
    E_text = "-1 0 0; 0 0 0; 0 0 0"

    cmd = [
        "sgl",
        "mech",
        "defgrad",
        "--kind",
        "full",
        "--measure",
        measure,
        "--precision",
        "12",
    ]
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
        input=E_text,
        capture_output=True,
        text=True,
    )

    # Must fail
    assert res.returncode != 0

    # Error message should indicate SPD incompatibility (from our kinematics logic)
    assert "Green-Lagrange strain incompatible with SPD" in res.stderr or "SPD" in res.stderr


import subprocess


def test_unphysical_green_lagrange_spd_check(tmp_path: Path):
    """
    Unphysical GL strain: C = I + 2E not SPD -> must fail.
    """
    # E = diag(-1, 0, 0) -> C = diag(-1, 1, 1), not SPD
    E_text = "-1 0 0; 0 0 0; 0 0 0"

    cmd = [
        "sgl",
        "mech",
        "defgrad",
        "--kind",
        "full",
        "--measure",
        "green-lagrange",
        "--precision",
        "12",
    ]
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
        input=E_text,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    assert "Green-Lagrange strain incompatible with SPD".lower() in res.stderr.lower() \
        or "SPD" in res.stderr


def test_unphysical_hencky_nonsymmetric(tmp_path: Path):
    """
    For Hencky, any symmetric E is mathematically valid (log/exp bijection).
    The physically invalid case is a non-symmetric 'strain' tensor.

    Our CLI enforces symmetry at parse time, so an asymmetric 3x3 input
    must be rejected before reaching Hencky logic.
    """
    # Asymmetric "strain" matrix
    E_text = "0 1 0; 0 0 0; 0 0 0"

    cmd = [
        "sgl",
        "mech",
        "defgrad",
        "--kind",
        "full",
        "--measure",
        "hencky",
        "--precision",
        "12",
    ]
    res = subprocess.run(
        cmd,
        cwd=tmp_path,
        input=E_text,
        capture_output=True,
        text=True,
    )

    assert res.returncode != 0
    # This comes from _parse_3x3 symmetry check
    assert "must be symmetric" in res.stderr.lower()

