# tests/test_pwi2json.py
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
def test_json2xyz_cli(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Expects case.yaml like:
      pwi:    *.in
      xyz:       *.xyz
      gold:      *.in
      frames:    None|"all"|int
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    input_list = cfg["inputs"]

    # Stage inputs
    for input_ in input_list:
        src = case_dir / "input" / input_
        dst = tmp_path_cwd / input_
        shutil.copy(src, dst)

    exe = "sgl"
    cli_pwi2json = [exe, "qe", "pwi2json"]
    cli_json2xyz = [exe, "qe", "json2xyz"]
    cli_xyz2qe   = [exe, "io", "xyz2pwi"]
 
    for cli in [cli_pwi2json, cli_json2xyz, cli_xyz2qe]:
        check_cli_or_skip(cli)

    args1 = cli_pwi2json + ["i.01.in", "--pretty", "-o", "o.02.json"]
    args2 = cli_json2xyz + ["o.02.json", "-o", "o.03.xyz"]
    args3 = cli_xyz2qe + ["--pwi", "i.01.header", "--xyz", "o.03.xyz", "-o", "o.04.in"]

    # Run CLI; on failure, show stdout/stderr
    for args in [args1, args2, args3]:
        try:
            result = subprocess.run(args, text=True, capture_output=True, check=True)
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
        assert new_txt == gold_txt, f"XYZ mismatch for case {case_dir.name}"
