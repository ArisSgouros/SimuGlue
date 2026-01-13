from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")

import simuglue.workflow.cij.post as post


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_result_json(path: Path, *, stress6: np.ndarray, cell: np.ndarray) -> None:
    payload = {
        "stress6": [float(x) for x in np.asarray(stress6, float).ravel()],
        "cell": np.asarray(cell, float).reshape(3, 3).tolist(),
        "energy": 0.0,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _cij_matrix_from_out(out: dict) -> np.ndarray:
    comps = out["components"]
    C = np.zeros((len(comps), len(comps)), float)
    for a, i in enumerate(comps):
        for b, j in enumerate(comps):
            C[a, b] = float(out["C_mean"][f"{i}-{j}"])
    return C


def _S_matrix_from_out(out: dict) -> np.ndarray:
    comps = out["components"]
    S = np.zeros((len(comps), len(comps)), float)
    for a, i in enumerate(comps):
        for b, j in enumerate(comps):
            S[a, b] = float(out["S"][f"{i}-{j}"])
    return S


def test_post_cij_symmetrization_averages_offdiagonal(tmp_path: Path):
    workdir = tmp_path / "work"

    # Minimal reference
    cell = np.eye(3)
    (workdir / "run.ref").mkdir(parents=True, exist_ok=True)
    _write_result_json(workdir / "run.ref" / "result.json", stress6=np.zeros(6), cell=cell)

    # Construct a non-symmetric 2x2 stiffness in base units (eV/Å^3)
    eps = 0.1
    C11, C12 = 10.0, 20.0
    C21, C22 = 40.0, 50.0

    # i = 1 case
    cid1 = "run.c1_eps0.1"
    (workdir / cid1).mkdir(parents=True, exist_ok=True)
    s6_1 = np.zeros(6)
    s6_1[0] = C11 * eps
    s6_1[1] = C12 * eps
    _write_result_json(workdir / cid1 / "result.json", stress6=s6_1, cell=cell)

    # i = 2 case
    cid2 = "run.c2_eps0.1"
    (workdir / cid2).mkdir(parents=True, exist_ok=True)
    s6_2 = np.zeros(6)
    s6_2[0] = C21 * eps
    s6_2[1] = C22 * eps
    _write_result_json(workdir / cid2 / "result.json", stress6=s6_2, cell=cell)

    # symmetrize = false
    cfg0 = tmp_path / "cij0.yaml"
    _write_yaml(
        cfg0,
        f"""
backend: dummy
workdir: {workdir}
components: [1, 2]
strains: [{eps}]
output:
  cij_json: cij.json
  units_cij: gpa
  symmetrize_cij: false
""",
    )
    out0 = post.post_cij(str(cfg0))
    assert out0["C_mean"]["1-2"] == pytest.approx(C12 / post.units.GPa)
    assert out0["C_mean"]["2-1"] == pytest.approx(C21 / post.units.GPa)

    # symmetrize = true
    cfg1 = tmp_path / "cij1.yaml"
    _write_yaml(
        cfg1,
        f"""
backend: dummy
workdir: {workdir}
components: [1, 2]
strains: [{eps}]
output:
  cij_json: cij.json
  units_cij: gpa
  symmetrize_cij: true
""",
    )
    out1 = post.post_cij(str(cfg1))

    avg = 0.5 * (C12 + C21) / post.units.GPa
    assert out1["C_mean"]["1-2"] == pytest.approx(avg)
    assert out1["C_mean"]["2-1"] == pytest.approx(avg)


def test_post_cij_compliance_is_inverse_in_export_units(tmp_path: Path):
    workdir = tmp_path / "work"
    cell = np.eye(3)

    (workdir / "run.ref").mkdir(parents=True, exist_ok=True)
    _write_result_json(workdir / "run.ref" / "result.json", stress6=np.zeros(6), cell=cell)

    # Make a symmetric positive-definite 6x6 stiffness matrix (base units)
    rng = np.random.default_rng(0)
    A = rng.random((6, 6))
    C_base = A.T @ A + 2.0 * np.eye(6)

    strains = [0.01, -0.01]
    for eps in strains:
        for i in range(1, 7):
            cid = f"run.c{i}_eps{eps:g}"
            (workdir / cid).mkdir(parents=True, exist_ok=True)
            s6 = C_base[i - 1] * eps  # matches post.py convention (i=strain, j=stress)
            _write_result_json(workdir / cid / "result.json", stress6=s6, cell=cell)

    cfg = tmp_path / "cij.yaml"
    _write_yaml(
        cfg,
        f"""
backend: dummy
workdir: {workdir}
components: [1, 2, 3, 4, 5, 6]
strains: {strains}
output:
  cij_json: cij.json
  units_cij: gpa
  symmetrize_cij: true
""",
    )

    out = post.post_cij(str(cfg))
    assert out["S"], "S (compliance) should be present for components [1..6]"

    C_gpa = _cij_matrix_from_out(out)
    S_1_per_gpa = _S_matrix_from_out(out)

    I = C_gpa @ S_1_per_gpa
    assert np.allclose(I, np.eye(6), atol=1e-8, rtol=1e-8)


def test_post_cij_units_pa_m_matches_gpa_times_thickness(tmp_path: Path):
    workdir = tmp_path / "work"

    # Thickness = |c| = 12 Å
    cell = np.diag([1.0, 1.0, 12.0])
    (workdir / "run.ref").mkdir(parents=True, exist_ok=True)
    _write_result_json(workdir / "run.ref" / "result.json", stress6=np.zeros(6), cell=cell)

    # Simple diagonal stiffness (base units)
    C_base = np.diag([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    strains = [0.02, -0.02]

    for eps in strains:
        for i in range(1, 7):
            cid = f"run.c{i}_eps{eps:g}"
            (workdir / cid).mkdir(parents=True, exist_ok=True)
            s6 = C_base[i - 1] * eps
            _write_result_json(workdir / cid / "result.json", stress6=s6, cell=cell)

    cfg_gpa = tmp_path / "cij_gpa.yaml"
    _write_yaml(
        cfg_gpa,
        f"""
backend: dummy
workdir: {workdir}
components: [1, 2, 3, 4, 5, 6]
strains: {strains}
output:
  cij_json: cij.json
  units_cij: gpa
  symmetrize_cij: true
""",
    )
    out_gpa = post.post_cij(str(cfg_gpa))

    cfg_pm = tmp_path / "cij_pm.yaml"
    _write_yaml(
        cfg_pm,
        f"""
backend: dummy
workdir: {workdir}
components: [1, 2, 3, 4, 5, 6]
strains: {strains}
output:
  cij_json: cij.json
  units_cij: "pa m"
  symmetrize_cij: true
""",
    )
    out_pm = post.post_cij(str(cfg_pm))

    thickness_m = 12.0e-10
    for k, v_gpa in out_gpa["C_mean"].items():
        v_pm = out_pm["C_mean"][k]
        expected = float(v_gpa) * 1.0e9 * thickness_m
        assert v_pm == pytest.approx(expected, rel=1e-10, abs=0.0)
