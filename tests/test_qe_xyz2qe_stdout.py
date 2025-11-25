import os
from pathlib import Path
from textwrap import dedent

import pytest
from ase import Atoms
from ase.io import write

from simuglue.cli.io_xyz2pwi import main  # adjust import to your actual CLI module


PWI_TEMPLATE = dedent(
    """\
    &control
        calculation = 'scf',
        prefix='tmpl',
        outdir='./out',
    /
    &system
        ibrav=0
    /
    CELL_PARAMETERS {angstrom}
    1.0000000000000000 0.0000000000000000 0.0000000000000000
    0.0000000000000000 1.0000000000000000 0.0000000000000000
    0.0000000000000000 0.0000000000000000 1.0000000000000000

    ATOMIC_POSITIONS {angstrom}
    Si 0.0000000000000000 0.0000000000000000 0.0000000000000000 
    """
)


def _make_single_frame_xyz(path: Path):
    atoms = Atoms(
        "Si",
        positions=[[0.1, 0.2, 0.3]],
        cell=[[2.0, 0.0, 0.0],
              [0.0, 2.0, 0.0],
              [0.0, 0.0, 2.0]],
        pbc=True,
    )
    write(path, atoms, format="extxyz")


@pytest.mark.usefixtures("tmp_path")
def test_stdout_default_single_frame(tmp_path, capsys):
    """Default behavior (no -o, no --frames): single frame -> stdout, no files, no summary."""
    pwi = tmp_path / "tmpl.pwi"
    xyz = tmp_path / "traj.xyz"

    pwi.write_text(PWI_TEMPLATE, encoding="utf-8")
    _make_single_frame_xyz(xyz)

    # Run CLI: expect stdout output, return code 0
    rc = main([
        "--pwi", str(pwi),
        "--xyz", str(xyz),
    ])
    assert rc == 0

    out = capsys.readouterr().out

    # QE content should be printed
    assert "CELL_PARAMETERS {angstrom}" in out
    assert "ATOMIC_POSITIONS {angstrom}" in out
    # Updated values from xyz should appear
    assert "2.0000000000000000 0.0000000000000000 0.0000000000000000" in out
    assert "Si 0.1000000000000000 0.2000000000000000 0.3000000000000000 " in out

    # No .in files should have been created
    in_files = list(tmp_path.glob("*.in"))
    assert in_files == []


@pytest.mark.usefixtures("tmp_path")
def test_stdout_single_frame_explicit_dash_o(tmp_path, capsys):
    """With -o - and single frame: write QE input to stdout, no files, no summary."""
    pwi = tmp_path / "tmpl.pwi"
    xyz = tmp_path / "frame.xyz"

    pwi.write_text(PWI_TEMPLATE, encoding="utf-8")
    _make_single_frame_xyz(xyz)

    rc = main([
        "--pwi", str(pwi),
        "--xyz", str(xyz),
        "-o", "-",
    ])
    assert rc == 0

    out = capsys.readouterr().out

    assert "CELL_PARAMETERS {angstrom}" in out
    assert "ATOMIC_POSITIONS {angstrom}" in out

    in_files = list(tmp_path.glob("*.in"))
    assert in_files == []

