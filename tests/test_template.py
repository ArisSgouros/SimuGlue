# tests/test_thermo_merge.py
from pathlib import Path
from simuglue.template import template_function

def test_template(tmp_path: Path):

    # paths to test data in the repo
    gold_dir = Path(__file__).parent / "data" / "template"
    gold_file = gold_dir / "gold"

    # run code under test
    new_data = template_function()

    # compare new output with reference output
    gold_data = gold_file.read_text().strip()

    assert new_data == gold_data
