# tests/test_pwi2json.py
from pathlib import Path
import subprocess
import shutil
import yaml
import pytest

CASES = Path(__file__).parent / "cases" / "xyz2qe_cli"

def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]

@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_json2xyz_cli(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Expects case.yaml like:
      header:    *.in
      xyz:       *.xyz
      gold:      *.in
      frames:    None|"all"|int
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    input_list = cfg["inputs"]

    cli = "sgl-xyz2qe"
    if shutil.which(cli) is None:
        pytest.skip(f"CLI '{cli}' not found in PATH — is it installed in the environment?")

    # Build CLI args
    args = [cli]

    # Stage input
    src = case_dir / "input" / input_list[0]
    dst = tmp_path_cwd / input_list[0]
    shutil.copy(src, dst)
    args += ["--header", str(dst)]

    src = case_dir / "input" / input_list[1]
    dst = tmp_path_cwd / input_list[1]
    shutil.copy(src, dst)
    args += ["--xyz", str(dst)]

    frames = cfg.get("frames", None)
    if frames:
        args += ["--frames", str(frames)]

    prefix = cfg.get("prefix", None)
    if prefix:
        args += ["--prefix", str(prefix)]

    # Run CLI; on failure, show stdout/stderr
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
