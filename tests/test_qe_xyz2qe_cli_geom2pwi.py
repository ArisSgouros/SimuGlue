# tests/test_geom2pwi_cli.py
from __future__ import annotations

from pathlib import Path
from _utils import check_cli_or_skip
import subprocess
import shutil
import yaml
import pytest

CASES = Path(__file__).parent / "cases" / "qe_xyz2qe_cli"


def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]


@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_geom2pwi_cli(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Expects case.yaml like:
      inputs:  [pwi_file, struct_file]
      outputs: [list of produced files]
      gold:    [list of gold files]
      frames:  None | "all" | int   (optional)
      output:  base name or "-"     (optional, maps to -o/--output)
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    exe = "sgl"
    cli = [exe, "io", "geom2pwi"]
    check_cli_or_skip(cli)

    args = cli.copy()

    input_list = cfg["inputs"]
    if len(input_list) < 2:
        pytest.fail(f"Expected at least 2 inputs (pwi, structure) in case {case_dir.name}")

    # Stage PWI template
    src = case_dir / "input" / input_list[0]
    dst = tmp_path_cwd / input_list[0]
    shutil.copy(src, dst)
    args += ["--pwi", str(dst)]

    # Stage structure file (extxyz, lammps-data, traj, ...)
    src = case_dir / "input" / input_list[1]
    dst = tmp_path_cwd / input_list[1]
    shutil.copy(src, dst)
    args += ["--input", str(dst)]
    # Let geom2pwi infer --iformat from suffix; if you ever need explicit control,
    # add "iformat" to case.yaml and pass it here.

    frames = cfg.get("frames", None)
    if frames is not None:
        args += ["--frames", str(frames)]

    output = cfg.get("output", None)
    if output is not None:
        args += ["--output", str(output)]

    # Run CLI; on failure, show stdout/stderr
    try:
        subprocess.run(args, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"CLI failed (code {e.returncode}).\n"
            f"CMD: {' '.join(args)}\n"
            f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
        )

    output_list = cfg["outputs"]
    gold_list = cfg["gold"]
    for output_file, gold_file in zip(output_list, gold_list):
        out_path = tmp_path_cwd / output_file

        assert out_path.exists(), f"Expected output file not found: {out_path}"

        gold_path = case_dir / "gold" / gold_file

        if update_gold:
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            existed = gold_path.exists()
            shutil.copy(out_path, gold_path)
            print(f"[{'UPDATED' if existed else 'CREATED'} GOLD] {gold_path}")
            return

        # Compare to gold (as text)
        new_txt = out_path.read_text(encoding="utf-8")
        gold_txt = gold_path.read_text(encoding="utf-8")
        assert new_txt == gold_txt, f"QE input mismatch for case {case_dir.name}"
