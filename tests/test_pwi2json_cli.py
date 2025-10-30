# tests/test_pwi2json.py
from pathlib import Path
import subprocess
import shutil
import yaml
import json
import pytest

CASES = Path(__file__).parent / "cases" / "pwi2json_cli"

def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]

@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_pwi2json_cli(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Expects case.yaml like:
      input:  log.out | pattern: log*.out
      output: o.json
      gold:   o.json
      pretty: true
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    cli = "sgl-pwi2json"
    if shutil.which(cli) is None:
        pytest.skip(f"CLI '{cli}' not found in PATH — is it installed in the environment?")

    # Stage inputs
    pattern = cfg.get("pattern")
    input_dir = case_dir / "input"

    if pattern:
        # Copy all matching input files to tmp_path
        for src in input_dir.glob(pattern):
            shutil.copy(src, tmp_path_cwd / src.name)
        # We'll pass an expanded glob to the CLI (portable)
        files = sorted(tmp_path_cwd.glob(pattern))
        args = [cli, *map(str, files)]
    else: # single input
        src = (case_dir / "input" / cfg["input"])
        dst = tmp_path_cwd / cfg["input"]
        shutil.copy(src, dst)
        shutil.copy(src, dst)
        args = [cli, str(dst)]

    # Build CLI args
    out_path = tmp_path_cwd / cfg.get("output", "o.json")
    args += ["-o", str(out_path)]
    if cfg.get("pretty", False):
        args.append("--pretty")

    # Run CLI; on failure, show stdout/stderr
    try:
        result = subprocess.run(
            args, text=True, capture_output=True, check=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"CLI failed (code {e.returncode}).\n"
            f"CMD: {' '.join(args)}\n"
            f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
        )

    assert out_path.exists(), f"Expected output file not found: {out_path}"

    # Compare to gold (as JSON objects)
    gold_file = case_dir / "gold" / cfg["gold"]

    if update_gold:
        gold_file.parent.mkdir(parents=True, exist_ok=True)
        existed = gold_file.exists()
        new_obj = json.loads(out_path.read_text(encoding="utf-8"))
        # write canonical JSON (stable keys/newlines)
        gold_file.write_text(
            json.dumps(new_obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )
        print(f"[{'UPDATED' if existed else 'CREATED'} GOLD] {gold_file}")
        return

    # Robust comparison: parse both sides
    new_obj = json.loads(out_path.read_text(encoding="utf-8"))
    gold_obj = json.loads(gold_file.read_text(encoding="utf-8"))
    assert new_obj == gold_obj, f"JSON mismatch for case {case_dir.name}"

