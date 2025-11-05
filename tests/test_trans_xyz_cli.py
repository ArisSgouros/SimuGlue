# tests/test_transf_xyz_cli.py
from pathlib import Path
import subprocess
import shutil
import yaml
import pytest

CASES = Path(__file__).parent / "cases" / "transf_xyz_cli"

def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]

@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_transf_xyz_cli(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Expects case.yaml like:
      inputs:
        - single.xyz              # required (input extxyz)
      transformer: "a b c; d e f; g h i"   # required
      voigt: false                # optional, default false
      frames: None|"all"|int      # optional
      output: transf.xyz          # required (output filename)
      gold:   transf.xyz          # required (gold filename)
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    exe = "sgl"
    cli = [exe, "transform","xyz"]

    # launcher on PATH
    if shutil.which(exe) is None:
        pytest.skip(f"'{exe}' not found on PATH")

    # subcommand exists
    probe = subprocess.run(
        cli + ["--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        command_str = " ".join(cli)
        pytest.skip(f"Subcommand '{command_str}' not available")

    args = cli

    # Stage input xyz
    in_xyz_name = cfg["inputs"][0]
    src_xyz = case_dir / "input" / in_xyz_name
    dst_xyz = tmp_path_cwd / in_xyz_name
    shutil.copy(src_xyz, dst_xyz)

    # Build CLI args
    out_name = cfg["output"]
    out_path = tmp_path_cwd / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args += [
        str(dst_xyz),
        str(cfg["transformer"]),
        "--output", str(out_path),
    ]

    # Optional flags
    if cfg.get("voigt", False):
        args.append("--voigt")
    frames = cfg.get("frames", None)
    if frames is not None:
        args += ["--frames", str(frames)]

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

    gold_name = cfg["gold"]
    gold_path = case_dir / "gold" / gold_name

    if update_gold:
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        existed = gold_path.exists()
        shutil.copy(out_path, gold_path)
        print(f"[{'UPDATED' if existed else 'CREATED'} GOLD] {gold_path}")
        return

    # Compare as text (exact match)
    new_txt = out_path.read_text(encoding="utf-8")
    gold_txt = gold_path.read_text(encoding="utf-8")
    assert new_txt == gold_txt, f"XYZ mismatch for case {case_dir.name}"
