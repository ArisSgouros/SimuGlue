# tests/test_cli_wf_cij.py
import json
from pathlib import Path

import yaml

def test_sgl_wf_cij_help(capsys):
    """
    Smoke test: ensure three-level CLI wiring for `sgl wf cij` is alive.
    Does NOT run any workflow; just checks help runs without error.
    """
    from simuglue.cli.main import main as sgl_main

    try:
        # `sgl wf cij` (no action) → dispatcher forwards ["--help"] to wf_cij.main
        code = sgl_main(["wf", "cij"])
    except SystemExit as e:
        code = e.code

    assert code == 0

    out, err = capsys.readouterr()
    # A loose check to confirm we actually hit the wf_cij parser:
    assert "Cij workflow" in out


def test_cij_post_smoke(tmp_path):
    """
    Smoke test for workflow.cij.post:
    - uses synthetic tiny data (no backend)
    - ensures post_cij runs and writes a sane cij.json
    """
    from simuglue.workflow.cij.post import post_cij

    workdir = tmp_path
    cfg_path = workdir / "cij.yaml"

    # Minimal config: backend is unused in post_cij, so set any string
    cfg = {
        "backend": "dummy",
        "workdir": str(workdir),
        "components": [1],
        "strains": [0.01],
        "output": {
            "units_cij": "GPa",
            "precision": 4,
            "cij_json": "cij.json",
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    # Reference case: zero stress
    ref_dir = workdir / "run.ref"
    ref_dir.mkdir()
    ref_data = {"stress6": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
               }
    (ref_dir / "result.json").write_text(json.dumps(ref_data), encoding="utf-8")

    # Deformed case: simple non-zero σ_xx so C11 is well-defined
    case_dir = workdir / "run.c1_eps0.01"
    case_dir.mkdir()
    def_data = {"stress6": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
               }
    (case_dir / "result.json").write_text(json.dumps(def_data), encoding="utf-8")

    # Run post-processing
    out = post_cij(str(cfg_path))

    # Check returned dict structure
    assert "C_mean" in out
    assert "1-1" in out["C_mean"]
    assert out["units"]["stress"] == "GPa"

    # And check file was written
    cij_file = workdir / "cij.json"
    assert cij_file.is_file()

    on_disk = json.loads(cij_file.read_text(encoding="utf-8"))
    assert "C_mean" in on_disk
    assert "1-1" in on_disk["C_mean"]
    assert on_disk["units"]["stress"] == "GPa"

