from pathlib import Path
import sys
import os
from simuglue.workflow.cij.workflow import init_cij, run_cij, parse_cij
from simuglue.workflow.cij.post import post_cij

def run_parse_post(cfg, ref):
    os.chdir(Path(__file__).parent)
    config_path = Path(cfg)
    ref_path = Path(ref)
    if ref_path.is_file():
        ref_path.unlink()
    parse_cij(config_path=config_path)
    post_cij(config_path=config_path)
    assert ref_path.is_file(), f"File not found: {ref_path}"

def test_parse_post_2d():
    run_parse_post(cfg="cij_2d.yaml", ref="o.res_3d/cij.json")
