# tests/test_thermo_merge.py
from pathlib import Path
import pytest
import yaml
from simuglue.template import template_function

CASES = Path(__file__).parent / "cases" / "template"

def discover():
    return [p for p in CASES.iterdir() if (p / "case.yaml").exists()]

@pytest.mark.parametrize("case_dir", discover(), ids=lambda p: p.name)
def test_template(case_dir: Path):

    cfg = yaml.safe_load((case_dir / "case.yaml").read_text())

    input_msg = cfg.get("input_msg")
    gold_msg = cfg.get("gold_msg")

    output_msg = template_function(input_msg)

    assert output_msg == gold_msg
