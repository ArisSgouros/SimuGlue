from __future__ import annotations
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write


def run_deform_xyz(tmp_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["sgl", "transform", "xyz", *args]
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


def _make_simple_xyz(path: Path) -> None:
    """Create a tiny extxyz with 2 atoms at (0,0,0) and (1,0,0)."""
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    write(path, atoms, format="extxyz")


def test_deform_xyz_file_to_file_single_frame(tmp_path: Path):
    """
    File in -> file out, single frame.
    Apply F = diag(2,1,1): x positions should double.
    """
    src = tmp_path / "in.xyz"
    dst = tmp_path / "out.xyz"
    _make_simple_xyz(src)

    F = "2 0 0; 0 1 0; 0 0 1"

    run_deform_xyz(
        tmp_path,
        [
            "-i",
            str(src),
            "-o",
            str(dst),
            "--F",
            F,
        ],
    )

    out = read(dst, format="extxyz")
    pos = out.get_positions()

    assert np.allclose(pos[0], [0.0, 0.0, 0.0])
    assert np.allclose(pos[1], [2.0, 0.0, 0.0])  # was 1.0 -> now 2.0


def test_deform_xyz_stdin_stdout_single_frame(tmp_path: Path):
    """
    stdin -> stdout, single frame.
    Same scaling F = diag(2,1,1), but fully streamed.
    """
    # Prepare input xyz text
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=[5, 5, 5], pbc=True)
    buf = StringIO()
    write(buf, atoms, format="extxyz")
    xyz_text = buf.getvalue()

    F = "2 0 0; 0 1 0; 0 0 1"

    res = run_deform_xyz(
        tmp_path,
        [
            "-i",
            "-",
            "-o",
            "-",
            "--F",
            F,
        ],
        stdin=xyz_text,
    )

    out_atoms = read(StringIO(res.stdout), format="extxyz")
    pos = out_atoms.get_positions()

    assert np.allclose(pos[0], [0.0, 0.0, 0.0])
    assert np.allclose(pos[1], [2.0, 0.0, 0.0])


def test_deform_xyz_multiframe_all(tmp_path: Path):
    """
    Multi-frame file in -> file out with --frames all.
    Check that both frames are deformed.
    """
    src = tmp_path / "multi.xyz"
    dst = tmp_path / "multi_out.xyz"

    # Build a 2-frame extxyz
    atoms1 = Atoms("H", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    atoms2 = Atoms("H", positions=[[1, 1, 0]], cell=[5, 5, 5], pbc=True)
    write(src, [atoms1, atoms2], format="extxyz")

    # Simple translation-like deformation in x,y (linear map)
    F = "1 0 0; 0 1 0; 0 0 1"  # identity (for structure), then tweak to see effect
    # Let's use a shear in x from y: F = [[1, 0.5, 0],[0,1,0],[0,0,1]]
    F = "1 0.5 0; 0 1 0; 0 0 1"

    run_deform_xyz(
        tmp_path,
        [
            "-i",
            str(src),
            "-o",
            str(dst),
            "--F",
            F,
            "--frames",
            "all",
        ],
    )

    frames_out = list(read(dst, format="extxyz", index=":"))
    assert len(frames_out) == 2

    pos1 = frames_out[0].get_positions()
    pos2 = frames_out[1].get_positions()

    # Frame 1: (0,0,0) stays (0,0,0)
    assert np.allclose(pos1[0], [0.0, 0.0, 0.0])

    # Frame 2: original (1,1,0) -> x' = 1 + 0.5*1 = 1.5, y' = 1
    assert np.allclose(pos2[0], [1.5, 1.0, 0.0])

