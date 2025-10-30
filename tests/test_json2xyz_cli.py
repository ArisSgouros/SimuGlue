# tests/test_pwi2json.py
from pathlib import Path
import subprocess
import shutil
import yaml
import pytest

CASES = Path(__file__).parent / "cases" / "json2xyz_cli"

def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]

@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_pwi2json_cli(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Expects case.yaml like:
      input:     file.json
      output:    o.xyz
      gold:      o.xyz
      no_forces: true                # optional, default false
      info_keys: [energy, virial]    # optional, default []
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    cli = "sgl-json2xyz"
    if shutil.which(cli) is None:
        pytest.skip(f"CLI '{cli}' not found in PATH — is it installed in the environment?")

    # Stage input
    src = case_dir / "input" / cfg["input"]
    dst = tmp_path_cwd / cfg["input"]
    shutil.copy(src, dst)

    # Build CLI args
    args = [cli, str(dst)]

    # Optional flags
    if cfg.get("no_forces", False):
        args.append("--no-forces")

    info_keys = cfg.get("info_keys", [])
    if info_keys:
        # Supports multiple values: --info-keys energy virial ...
        args += ["--info-keys", *map(str, info_keys)]

    out_path = tmp_path_cwd / cfg.get("output", "o.xyz")
    args += ["-o", str(out_path)]

    print(args)
    print(args)
    print(args)
    print(args)

    # Run CLI; on failure, show stdout/stderr
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(
            f"CLI failed (code {e.returncode}).\n"
            f"CMD: {' '.join(args)}\n"
            f"STDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}"
        )

    assert out_path.exists(), f"Expected output file not found: {out_path}"

    # Compare to gold (as text)
    gold_file = case_dir / "gold" / cfg["gold"]

    if update_gold:
        gold_file.parent.mkdir(parents=True, exist_ok=True)
        existed = gold_file.exists()
        shutil.copy(out_path, gold_file)
        print(f"[{'UPDATED' if existed else 'CREATED'} GOLD] {gold_file}")
        return

    new_txt = out_path.read_text(encoding="utf-8")
    gold_txt = gold_file.read_text(encoding="utf-8")
    assert new_txt == gold_txt, f"XYZ mismatch for case {case_dir.name}"

