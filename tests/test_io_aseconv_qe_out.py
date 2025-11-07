from __future__ import annotations
import subprocess
from pathlib import Path

import pytest
from ase.io import read


def run_aseconv(tmp_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run `sgl io aseconv` inside tmp_path and fail loudly on error."""
    cmd = ["sgl", "io", "aseconv", *args]
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


CASES_QE = Path(__file__).parent / "cases" / "io_aseconv_qe_out"


@pytest.mark.skipif(
    not (CASES_QE / "qe_long.out").exists() or not (CASES_QE / "gold.xyz").exists(),
    reason="QE gold files not available",
)
def test_qe_out_to_extxyz_matches_gold(tmp_path: Path):
    """
    Convert a real QE output (espresso-out) to extxyz (last frame)
    and compare against a trusted gold.xyz.
    """
    src_qe = CASES_QE / "qe_long.out"
    gold_xyz = CASES_QE / "gold.xyz"

    # Copy to tmp_path to avoid touching fixtures
    qe_local = tmp_path / "qe_long.out"
    gold_local = tmp_path / "gold.xyz"
    qe_local.write_text(src_qe.read_text(encoding="utf-8"), encoding="utf-8")
    gold_local.write_text(gold_xyz.read_text(encoding="utf-8"), encoding="utf-8")

    out_xyz = tmp_path / "out.xyz"

    # Use only the last frame from QE output
    run_aseconv(
        tmp_path,
        [
            "-i",
            str(qe_local),
            "--iformat",
            "espresso-out",
            "--frames",
            "-1",
            "-o",
            str(out_xyz),
            "--oformat",
            "extxyz",
            "--overwrite",
        ],
    )

    got = read(out_xyz, format="extxyz")
    ref = read(gold_local, format="extxyz")

    # Basic invariants
    assert len(got) == len(ref), "Number of atoms differs"

    # Symbols: exact match
    assert got.get_chemical_symbols() == ref.get_chemical_symbols()

    # Cell: small numerical tolerance
    assert got.get_cell().array == pytest.approx(
        ref.get_cell().array, rel=1e-8, abs=1e-8
    )

    # Positions: small numerical tolerance
    assert got.get_positions() == pytest.approx(
        ref.get_positions(), rel=1e-8, abs=1e-8
    )

