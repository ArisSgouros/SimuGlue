from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("ase")
from ase import Atoms

# Import submodules directly.
import simuglue.workflow.cij.config as cfgmod
import simuglue.workflow.cij.registry as reg
import simuglue.workflow.cij.workflow as wf


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class DummyBackend:
    """Backend used to test init/run/parse logic without external engines."""

    def __init__(self):
        self.prepared: dict[str, Atoms] = {}
        self.ran: list[str] = []
        self.parsed: list[str] = []

    # Be permissive on signatures because the project evolved.
    def read_data(self, *args, **kwargs):
        # Return a deterministic reference structure.
        return Atoms(
            symbols=["Si", "Si"],
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=np.eye(3),
            pbc=True,
        )

    def prepare_case(self, case_dir: Path, atoms: Atoms, *args, **kwargs) -> None:
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "prepared.txt").write_text("ok\n", encoding="utf-8")
        self.prepared[str(case_dir)] = atoms.copy()

    def run_case(self, case_dir: Path, *args, **kwargs) -> None:
        self.ran.append(str(case_dir))
        # emulate successful completion
        (case_dir / ".done").write_text("done\n", encoding="utf-8")

    def parse_case(self, case_dir: Path, *args, **kwargs):
        self.parsed.append(str(case_dir))
        # Return a fixed stress tensor and cell.
        # Stress is in the units expected by the workflow's converters (base units).
        stress = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0], [3.0, 5.0, 6.0]], float)
        cell = np.eye(3)
        return reg.RelaxResult(energy=1.23, stress=stress, cell=cell)


def _register_dummy_backend(monkeypatch) -> DummyBackend:
    dummy = DummyBackend()
    # registry uses a module-level dict
    monkeypatch.setitem(reg.__dict__.get("_BACKENDS"), "dummy", dummy)
    return dummy


def test_F_from_component_mapping():
    F = wf.F_from_component(1, 0.1)
    assert np.allclose(F, np.diag([1.1, 1.0, 1.0]))

    F = wf.F_from_component(6, 0.2)
    expected = np.eye(3)
    expected[0, 1] += 0.2
    assert np.allclose(F, expected)

    with pytest.raises(ValueError):
        wf.F_from_component(0, 0.1)


def test_make_case_id_formatting():
    assert reg.make_case_id(1, 1e-5) == "run.c1_eps1e-05"
    assert reg.make_case_id(2, -1e-5) == "run.c2_eps-1e-05"
    assert reg.make_case_id(3, 0.001) == "run.c3_eps0.001"


def test_load_config_common_files_accepts_string(tmp_path: Path):
    p = tmp_path / "cij.yaml"
    _write_yaml(
        p,
        """
backend: dummy
workdir: wd
file_type: extxyz
common_files: somefile.txt
common_path: '.'
components: [1]
strains: [0.01]
output: {}
""",
    )
    cfg = cfgmod.load_config(p)
    assert isinstance(cfg.common_files, list)
    assert len(cfg.common_files) == 1
    assert cfg.common_files[0].name == "somefile.txt"


def test_init_cij_creates_cases_and_copies_common(tmp_path: Path, monkeypatch):
    # Arrange
    dummy = _register_dummy_backend(monkeypatch)

    # Patch apply_transform to a deterministic implementation.
    def _apply_transform(atoms: Atoms, F: np.ndarray):
        out = atoms.copy()
        out.set_positions(atoms.positions @ F.T)
        out.set_cell(atoms.cell @ F.T, scale_atoms=False)
        return out

    monkeypatch.setattr(wf, "apply_transform", _apply_transform)

    workdir = tmp_path / "work"
    common = tmp_path / "common.dat"
    common.write_text("x", encoding="utf-8")

    cfg_path = tmp_path / "cij.yaml"
    _write_yaml(
        cfg_path,
        f"""
backend: dummy
workdir: {workdir}
file_type: extxyz
common_files: [{common}]
common_path: '.'
components: [1, 6]
strains: [0.1]
output: {{}}
""",
    )

    # Act
    wf.init_cij(str(cfg_path))

    # Assert
    assert (workdir / common.name).is_file()

    ref_dir = workdir / "run.ref"
    assert (ref_dir / "prepared.txt").is_file()

    case_dir = workdir / reg.make_case_id(1, 0.1)
    assert (case_dir / "prepared.txt").is_file()

    # Ensure deformation was applied (x coordinate changed for atom 2)
    atoms_ref = dummy.prepared[str(ref_dir)]
    atoms_def = dummy.prepared[str(case_dir)]
    assert np.isclose(atoms_ref.positions[1, 0], 1.0)
    assert np.isclose(atoms_def.positions[1, 0], 1.1)


def test_parse_cij_writes_result_json_for_done_cases_only(tmp_path: Path, monkeypatch):
    dummy = _register_dummy_backend(monkeypatch)

    workdir = tmp_path / "work"
    cfg_path = tmp_path / "cij.yaml"
    _write_yaml(
        cfg_path,
        f"""
backend: dummy
workdir: {workdir}
file_type: extxyz
components: [1]
strains: [0.01]
output: {{}}
""",
    )

    # Prepare directory structure expected by parse_cij.
    ref_dir = workdir / "run.ref"
    case_dir = workdir / reg.make_case_id(1, 0.01)
    ref_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    # Only mark the deformed case as done.
    (case_dir / ".done").write_text("done\n", encoding="utf-8")

    wf.parse_cij(str(cfg_path))

    assert not (ref_dir / "result.json").exists()
    assert (case_dir / "result.json").exists()

    data = json.loads((case_dir / "result.json").read_text(encoding="utf-8"))
    assert data["kind"] == "sample"
    assert data["i"] == 1
    assert np.isfinite(np.array(data["stress6"], float)).all()

    assert str(case_dir) in dummy.parsed
    assert str(ref_dir) not in dummy.parsed


def test_run_cij_skips_done_deformed_cases(tmp_path: Path, monkeypatch):
    dummy = _register_dummy_backend(monkeypatch)

    workdir = tmp_path / "work"
    cfg_path = tmp_path / "cij.yaml"
    _write_yaml(
        cfg_path,
        f"""
backend: dummy
workdir: {workdir}
file_type: extxyz
components: [1]
strains: [0.01]
output: {{}}
""",
    )

    ref_dir = workdir / "run.ref"
    case_dir = workdir / reg.make_case_id(1, 0.01)
    ref_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)

    # Mark deformed case as done; it should be skipped.
    (case_dir / ".done").write_text("done\n", encoding="utf-8")

    wf.run_cij(str(cfg_path))

    assert str(ref_dir) in dummy.ran
    assert str(case_dir) not in dummy.ran


def test_run_cij_skips_done_reference_case(tmp_path: Path, monkeypatch):
    dummy = _register_dummy_backend(monkeypatch)

    workdir = tmp_path / "work"
    cfg_path = tmp_path / "cij.yaml"
    _write_yaml(
        cfg_path,
        f"""
backend: dummy
workdir: {workdir}
file_type: extxyz
components: [1]
strains: [0.01]
output: {{}}
""",
    )

    ref_dir = workdir / "run.ref"
    case_dir = workdir / reg.make_case_id(1, 0.01)
    ref_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)  # required by run_cij
    (case_dir / ".done").write_text("done\n", encoding="utf-8")  # avoid running deformed
    (ref_dir / ".done").write_text("done\n", encoding="utf-8")

    wf.run_cij(str(cfg_path))


    assert str(ref_dir) not in dummy.ran
