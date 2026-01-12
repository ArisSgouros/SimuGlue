from __future__ import annotations

import numpy as np
import pytest

from ase import Atoms


def _import_cli():
    import importlib
    return importlib.import_module("simuglue.cli.build_lmp_topo")


def _dummy_atoms() -> Atoms:
    atoms = Atoms(
        symbols=["C"],
        positions=[(0.0, 0.0, 0.0)],
        cell=[10.0, 10.0, 10.0],
        pbc=[False, False, False],
    )
    atoms.arrays["type"] = np.array([1], dtype=int)
    return atoms


def _touch(path):
    path.write_text("dummy")
    return path


def test_cli_pbc_override_is_applied_before_inference(tmp_path, monkeypatch):
    cli = _import_cli()

    inp = _touch(tmp_path / "in.data")
    out = tmp_path / "out.data"

    atoms = _dummy_atoms()

    monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: atoms)
    monkeypatch.setattr(cli, "get_lmp_type_table", lambda *_: {"tag": {1: "C"}, "mass": {1: 12.0}, "n_types": 1})

    seen = {}

    def fake_build(atoms_in, **kwargs):
        # pbc override should already be applied
        seen["pbc"] = tuple(bool(x) for x in atoms_in.pbc)
        from simuglue.topology.topo import Topo

        return Topo(bonds=[]), [[] for _ in range(len(atoms_in))]

    monkeypatch.setattr(cli, "build_topology_from_atoms", fake_build)
    monkeypatch.setattr(cli, "type_topology_inplace", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_topology_arrays_to_atoms", lambda *a, **k: None)
    monkeypatch.setattr(cli, "write_lammps_data", lambda *a, **k: None)

    cli.main(["-i", str(inp), "-o", str(out), "--bonds", "--rc", "1.0", "--pbc", "1,0,1"])

    assert seen["pbc"] == (True, False, True)


def test_cli_out_atom_style_default_and_override(tmp_path, monkeypatch):
    cli = _import_cli()

    inp = _touch(tmp_path / "in.data")
    out = tmp_path / "out.data"

    atoms = _dummy_atoms()

    monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: atoms)
    monkeypatch.setattr(cli, "get_lmp_type_table", lambda *_: {"tag": {1: "C"}, "mass": {1: 12.0}, "n_types": 1})

    from simuglue.topology.topo import Topo

    monkeypatch.setattr(cli, "build_topology_from_atoms", lambda *a, **k: (Topo(bonds=[]), [[]]))
    monkeypatch.setattr(cli, "type_topology_inplace", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_topology_arrays_to_atoms", lambda *a, **k: None)

    called = {"atom_style": []}

    def fake_write(_path, _atoms, *, atom_style, **kwargs):
        called["atom_style"].append(atom_style)

    monkeypatch.setattr(cli, "write_lammps_data", fake_write)

    # default: out_atom_style = in_atom_style
    cli.main(["-i", str(inp), "-o", str(out), "--bonds", "--rc", "1.0", "--in-atom-style", "full"])

    # override: out_atom_style explicit
    out2 = tmp_path / "out2.data"
    cli.main(
        [
            "-i",
            str(inp),
            "-o",
            str(out2),
            "--bonds",
            "--rc",
            "1.0",
            "--in-atom-style",
            "full",
            "--out-atom-style",
            "atomic",
        ]
    )

    assert called["atom_style"] == ["full", "atomic"]


@pytest.mark.parametrize(
    "argv, expected",
    [
        (
            [
                "--diff-bond-len",
                "--bond-len-fmt",
                "%.3f",
                "--type-delimiter",
                "_",
            ],
            {"diff_bond_len": True, "diff_bond_fmt": "%.3f", "type_delimiter": "_"},
        ),
        (
            [
                "--angle-symmetry",
                "--diff-angle-theta",
                "--angle-theta-fmt",
                "%.1f",
            ],
            {"angle_symmetry": True, "diff_angle_theta": True, "diff_angle_theta_fmt": "%.1f"},
        ),
        (
            [
                "--cis-trans",
                "--diff-dihed-theta",
                "--no-dihed-abs",
                "--dihed-theta-fmt",
                "%.0f",
            ],
            {"cis_trans": True, "diff_dihed_theta": True, "diff_dihed_theta_abs": False, "diff_dihed_theta_fmt": "%.0f"},
        ),
    ],
)
def test_cli_typing_options_are_passed_through(tmp_path, monkeypatch, argv, expected):
    cli = _import_cli()

    inp = _touch(tmp_path / "in.data")
    out = tmp_path / ("out_" + str(abs(hash(tuple(argv)))) + ".data")

    atoms = _dummy_atoms()

    monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: atoms)
    monkeypatch.setattr(cli, "get_lmp_type_table", lambda *_: {"tag": {1: "C"}, "mass": {1: 12.0}, "n_types": 1})

    from simuglue.topology.topo import Topo

    monkeypatch.setattr(cli, "build_topology_from_atoms", lambda *a, **k: (Topo(bonds=[]), [[]]))

    captured = {}

    def fake_type_topology_inplace(_atoms, _topo, *, opts, **kwargs):
        captured["opts"] = opts

    monkeypatch.setattr(cli, "type_topology_inplace", fake_type_topology_inplace)
    monkeypatch.setattr(cli, "attach_topology_arrays_to_atoms", lambda *a, **k: None)
    monkeypatch.setattr(cli, "write_lammps_data", lambda *a, **k: None)

    cli.main(["-i", str(inp), "-o", str(out), "--bonds", "--rc", "1.0", *argv])

    opts = captured["opts"]
    for k, v in expected.items():
        assert getattr(opts, k) == v


def test_cli_dry_run_ignores_types_out(tmp_path, monkeypatch, capsys):
    cli = _import_cli()

    inp = _touch(tmp_path / "in.data")
    out = tmp_path / "out.data"

    atoms = _dummy_atoms()

    monkeypatch.setattr(cli, "read_lammps_data", lambda *a, **k: atoms)
    monkeypatch.setattr(cli, "get_lmp_type_table", lambda *_: {"tag": {1: "C"}, "mass": {1: 12.0}, "n_types": 1})

    from simuglue.topology.topo import Topo

    monkeypatch.setattr(cli, "build_topology_from_atoms", lambda *a, **k: (Topo(bonds=[]), [[]]))
    monkeypatch.setattr(cli, "type_topology_inplace", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_topology_arrays_to_atoms", lambda *a, **k: None)

    # Should not be called in dry-run
    monkeypatch.setattr(cli, "write_lammps_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError("write should not happen")))

    cli.main(
        [
            "-i",
            str(inp),
            "-o",
            str(out),
            "--bonds",
            "--rc",
            "1.0",
            "--dry-run",
            "--types-out",
            str(tmp_path / "o.types"),
        ]
    )

    txt = capsys.readouterr().out
    assert "note: --types-out ignored in --dry-run" in txt
