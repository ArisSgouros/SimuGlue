# tests/test_pwi2json.py
from pathlib import Path
from _utils import check_cli_or_skip
import subprocess
import shutil
import yaml
import pytest

CASES = Path(__file__).parent / "cases" / "build_lmp_topo_cli"

def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]

@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_build_lmp_topo_cli(case_dir: Path, tmp_path_cwd: Path, update_gold: bool):
    """
    Expects case.yaml like:
      pwi:    *.in
      xyz:       *.xyz
      gold:      *.in
      frames:    None|"all"|int
    """
    cfg = yaml.safe_load((case_dir / "case.yaml").read_text(encoding="utf-8"))

    exe = "sgl"
    cli = [exe, "build","lmp-topo"]
    check_cli_or_skip(cli)

    args = cli

    input_list = cfg["inputs"]

    # Stage input
    src = case_dir / "input" / input_list[0]
    dst = tmp_path_cwd / input_list[0]
    shutil.copy(src, dst)
    args += ["-i", str(dst)]

    args += ["-o", "o.supercell_topo.dat"]
    args += ["--types-out", "o.types"]

    rc = cfg.get("rc", False)
    if rc:
        args += ["--rc", str(rc)]

    bonds = cfg.get("bonds", False)
    if bonds:
        args += ["--bonds"]
    diff_bond_length = cfg.get("diff-bond-len", False)
    if diff_bond_length:
        args += ["--diff-bond-len"]


    angles = cfg.get("angles", False)
    if angles:
        args += ["--angles"]
    angle_symmetry = cfg.get("angle-symmetry", False)
    if angle_symmetry:
        args += ["--angle-symmetry"]
    diff_angle_theta = cfg.get("diff-angle-theta", False)
    if diff_angle_theta:
        args += ["--diff-angle-theta"]
    angle_theta_fmt = cfg.get("angle-theta-fmt", False)
    if angle_theta_fmt:
        args += ["--angle-theta-fmt", angle_theta_fmt]

    dihedrals = cfg.get("dihedrals", False)
    if dihedrals:
        args += ["--dihedrals"]
    cistrans = cfg.get("cis-trans", False)
    if cistrans:
        args += ["--cis-trans"]
    diff_dihed_theta = cfg.get("diff-dihed-theta", False)
    if diff_dihed_theta:
        args += ["--diff-dihed-theta"]
    dihed_theta_fmt = cfg.get("dihed-theta-fmt", False)
    if dihed_theta_fmt:
        args += ["--dihed-theta-fmt", dihed_theta_fmt]

    type_delimiter = cfg.get("type-delimiter", False)
    if type_delimiter:
        args += ["--type-delimiter", type_delimiter]

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
