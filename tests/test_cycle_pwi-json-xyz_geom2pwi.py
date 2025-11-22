from __future__ import annotations

from pathlib import Path
from _utils import check_cli_or_skip
import subprocess
import shutil
import yaml
import pytest

CASES = Path(__file__).parent / "cases" / "cycle_pwi-json-xyz"

def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]

@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_geom2pwi_cycle(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Generic cycle:

        pwi2json → json2xyz → geom2pwi

    The case.yaml format mirrors the xyz2qe cycle tests:

      inputs:  [list of files to stage]
      outputs: [expected produced files]
      gold:    [gold files to compare against]
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    # --- Stage inputs ---
    for filename in cfg["inputs"]:
        src = case_dir / "input" / filename
        dst = tmp_path_cwd / filename
        shutil.copy(src, dst)

    exe = "sgl"
    cli_pwi2json = [exe, "qe", "pwi2json"]
    cli_json2xyz = [exe, "qe", "json2xyz"]
    cli_geom2pwi   = [exe, "io", "geom2pwi"]

    # Ensure CLI is available
    for cli in [cli_pwi2json, cli_json2xyz, cli_geom2pwi]:
        check_cli_or_skip(cli)

    # --- Build command sequences ---
    # 1. pwi2json
    args1 = cli_pwi2json + ["i.01.in", "--pretty", "-o", "o.02.json"]

    # 2. json2xyz
    args2 = cli_json2xyz + ["o.02.json", "-o", "o.03.xyz"]

    # 3. geom2pwi (instead of xyz2qe)
    #    input = o.03.xyz
    #    template pwi = i.01.header
    args3 = cli_geom2pwi + [
        "--pwi", "i.01.header",
        "--input", "o.03.xyz",
        "--iformat", "extxyz",
        "-o", "o.04.in",
    ]

    # --- Run the commands, fail loudly ---
    for args in [args1, args2, args3]:
        try:
            subprocess.run(args, text=True, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            pytest.fail(
                f"CLI failed (code {e.returncode}).\n"
                f"CMD: {' '.join(args)}\n"
                f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
            )

    # --- Compare outputs to gold ---
    outputs = cfg["outputs"]
    golds = cfg["gold"]

    for out_file, gold_file in zip(outputs, golds):
        out_path = tmp_path_cwd / out_file
        assert out_path.exists(), f"Expected output file not found: {out_path}"

        gold_path = case_dir / "gold" / gold_file

        if update_gold:
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            existed = gold_path.exists()
            shutil.copy(out_path, gold_path)
            print(f"[{'UPDATED' if existed else 'CREATED'} GOLD] {gold_path}")
            return

        new_txt = out_path.read_text(encoding="utf-8")
        gold_txt = gold_path.read_text(encoding="utf-8")
        assert new_txt == gold_txt, f"Output mismatch for case {case_dir.name}"

