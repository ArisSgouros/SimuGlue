from __future__ import annotations

import numpy as np
import pytest

from ase import Atoms

from simuglue.topology.topo import Topo


def _import_cli_module():
    """Import the CLI module.

    In the full SimuGlue repository, all dependencies should exist.
    This helper avoids hard failures in minimal test environments by
    allowing users to provide stubs if desired.
    """
    import importlib
    return importlib.import_module("simuglue.cli.build_lmp_topo")


def _dummy_atoms() -> Atoms:
    atoms = Atoms(
        symbols=["C", "C", "C"],
        positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        cell=[20.0, 20.0, 20.0],
        pbc=[True, True, True],
    )
    atoms.arrays["type"] = np.array([1, 1, 1], dtype=int)
    return atoms


def test_cli_requires_rc_if_bonds_implied(tmp_path, monkeypatch):
    cli = _import_cli_module()

    inp = tmp_path / "in.data"
    inp.write_text("dummy")
    out = tmp_path / "out.data"

    # Avoid touching real IO/inference; we only want the argument validation.
    monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: _dummy_atoms())
    monkeypatch.setattr(cli, "get_lmp_type_table", lambda *a, **k: {"tag": {1: "C"}})

    with pytest.raises(SystemExit, match=r"--rc is required"):
        cli.main(["-i", str(inp), "-o", str(out), "--angles"])


def test_cli_angles_imply_bonds_and_writes(tmp_path, monkeypatch):
    cli = _import_cli_module()

    inp = tmp_path / "in.data"
    inp.write_text("dummy")
    out = tmp_path / "out.data"
    types_out = tmp_path / "o.types"

    atoms = _dummy_atoms()

    called = {"build": None, "write": 0, "export": 0}

    def fake_read_lammps_data(*args, **kwargs):
        return atoms

    def fake_get_lmp_type_table(_atoms):
        return {"n_types": 1, "mass": {1: 12.011}, "tag": {1: "C"}}

    def fake_build_topology_from_atoms(
        atoms_in,
        *,
        calc_bonds: bool,
        calc_angles: bool,
        calc_diheds: bool,
        rc_list,
        drc: float,
        deduplicate: bool = True,
    ):
        called["build"] = (calc_bonds, calc_angles, calc_diheds, list(rc_list), float(drc), bool(deduplicate))
        topo = Topo(bonds=[(0, 1), (1, 2)])
        if calc_angles:
            topo.angles = [(0, 1, 2)]
        return topo, [[] for _ in range(len(atoms_in))]

    def fake_type_topology_inplace(*args, **kwargs):
        # Populate type arrays minimally so the downstream attach/writer has what it needs.
        topo: Topo = args[1]
        if topo.bonds:
            topo.bond_types = [1] * len(topo.bonds)
            topo.meta["bond_type_table"] = {1: {"tag": "C C"}}
        if topo.angles:
            topo.angle_types = [1] * len(topo.angles)
            topo.meta["angle_type_table"] = {1: {"tag": "C C C"}}

    def fake_attach_topology_arrays_to_atoms(*args, **kwargs):
        return

    def fake_write_lammps_data(*args, **kwargs):
        called["write"] += 1

    def fake_export_types_topo(*args, **kwargs):
        called["export"] += 1

    monkeypatch.setattr(cli, "read_lammps_data", fake_read_lammps_data)
    monkeypatch.setattr(cli, "get_lmp_type_table", fake_get_lmp_type_table)
    monkeypatch.setattr(cli, "build_topology_from_atoms", fake_build_topology_from_atoms)
    monkeypatch.setattr(cli, "type_topology_inplace", fake_type_topology_inplace)
    monkeypatch.setattr(cli, "attach_topology_arrays_to_atoms", fake_attach_topology_arrays_to_atoms)
    monkeypatch.setattr(cli, "write_lammps_data", fake_write_lammps_data)
    monkeypatch.setattr(cli, "ExportTypesTopo", fake_export_types_topo)

    ret = cli.main(
        [
            "-i",
            str(inp),
            "-o",
            str(out),
            "--angles",
            "--rc",
            "1.0",
            "--types-out",
            str(types_out),
        ]
    )
    assert ret == 0

    # angles imply bonds in main()
    assert called["build"] is not None
    calc_bonds, calc_angles, calc_diheds, rc_list, drc, dedup = called["build"]
    assert calc_bonds is True
    assert calc_angles is True
    assert calc_diheds is False
    assert rc_list == [1.0]

    # non-dry-run writes outputs
    assert called["write"] == 1
    assert called["export"] == 1


def test_cli_dry_run_does_not_write(tmp_path, monkeypatch, capsys):
    cli = _import_cli_module()

    inp = tmp_path / "in.data"
    inp.write_text("dummy")
    out = tmp_path / "out.data"

    atoms = _dummy_atoms()

    monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: atoms)
    monkeypatch.setattr(cli, "get_lmp_type_table", lambda *_: {"n_types": 1, "mass": {1: 12.011}, "tag": {1: "C"}})

    # Minimal topology to make dry-run print counts.
    def fake_build(*args, **kwargs):
        topo = Topo(bonds=[(0, 1), (1, 2)])
        topo.bond_types = [1, 1]
        return topo, [[] for _ in range(len(atoms))]

    monkeypatch.setattr(cli, "build_topology_from_atoms", fake_build)
    monkeypatch.setattr(cli, "type_topology_inplace", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_topology_arrays_to_atoms", lambda *a, **k: None)

    wrote = {"called": 0}
    monkeypatch.setattr(cli, "write_lammps_data", lambda *a, **k: wrote.__setitem__("called", wrote["called"] + 1))

    ret = cli.main(["-i", str(inp), "-o", str(out), "--bonds", "--rc", "1.0", "--dry-run"])
    assert ret == 0

    # ensure no output is written
    assert wrote["called"] == 0

    captured = capsys.readouterr().out
    assert "atoms:" in captured
    assert "bonds:" in captured


def test_cli_refuses_overwrite(tmp_path, monkeypatch):
    cli = _import_cli_module()

    inp = tmp_path / "in.data"
    inp.write_text("dummy")
    out = tmp_path / "out.data"
    out.write_text("existing")

    # avoid touching real IO/inference
    monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: _dummy_atoms())
    monkeypatch.setattr(cli, "get_lmp_type_table", lambda *_: {"tag": {1: "C"}})

    with pytest.raises(SystemExit, match=r"Refusing to overwrite"):
        cli.main(["-i", str(inp), "-o", str(out), "--bonds", "--rc", "1.0"])
