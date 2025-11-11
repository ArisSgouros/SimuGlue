import re
from textwrap import dedent

from simuglue.quantum_espresso.pwi_update import update_qe_input  # adjust to your actual module path


BASE_QE_INPUT = dedent(
    """\
    &control
        calculation = 'scf',
        prefix='old_prefix',
        outdir='/old/path',
    /
    &system
        ibrav=0
    /
    CELL_PARAMETERS {angstrom}
    1.0000000000000000 0.0000000000000000 0.0000000000000000
    0.0000000000000000 1.0000000000000000 0.0000000000000000
    0.0000000000000000 0.0000000000000000 1.0000000000000000

    ATOMIC_POSITIONS {angstrom}
    Si 0.0000000000000000 0.0000000000000000 0.0000000000000000 
    """
)


def _get_block(text: str, header: str) -> str:
    """Return the card block starting at `header` up to next blank line or EOF."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(header):
            start = i
            break
    if start is None:
        return ""
    j = start + 1
    while j < len(lines) and lines[j].strip() != "":
        j += 1
    return "\n".join(lines[start:j])


def test_update_cell_parameters_only():
    new_cell = [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    out = update_qe_input(BASE_QE_INPUT, cell=new_cell)

    cell_block = _get_block(out, "CELL_PARAMETERS")
    assert "CELL_PARAMETERS {angstrom}" in cell_block
    assert "2.0000000000000000 0.0000000000000000 0.0000000000000000" in cell_block
    assert "0.0000000000000000 3.0000000000000000 0.0000000000000000" in cell_block
    assert "0.0000000000000000 0.0000000000000000 4.0000000000000000" in cell_block

    # Old cell entries should be gone
    assert "1.0000000000000000 0.0000000000000000 0.0000000000000000" not in cell_block

    # Prefix / outdir / positions unchanged
    assert "prefix='old_prefix'" in out
    assert "outdir='/old/path'" in out
    assert "Si 0.0000000000000000 0.0000000000000000 0.0000000000000000 " in out


def test_update_atomic_positions_only():
    symbols = ["Si", "Si"]
    positions = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    out = update_qe_input(BASE_QE_INPUT, positions=positions, symbols=symbols)

    pos_block = _get_block(out, "ATOMIC_POSITIONS")
    assert "ATOMIC_POSITIONS {angstrom}" in pos_block
    assert "Si 0.1000000000000000 0.2000000000000000 0.3000000000000000 " in pos_block
    assert "Si 0.4000000000000000 0.5000000000000000 0.6000000000000000 " in pos_block

    # Old position line should be gone
    assert (
        "Si 0.0000000000000000 0.0000000000000000 0.0000000000000000 "
        not in pos_block
    )

    # Cell / prefix / outdir unchanged
    assert "1.0000000000000000 0.0000000000000000 0.0000000000000000" in out
    assert "prefix='old_prefix'" in out
    assert "outdir='/old/path'" in out


def test_update_prefix_only():
    out = update_qe_input(BASE_QE_INPUT, prefix="new_prefix")

    # Updated prefix line
    assert re.search(r"prefix='new_prefix'\s*,?\s*$", out, re.MULTILINE)

    # Old prefix should be gone
    assert "prefix='old_prefix'" not in out

    # Everything else unchanged (spot checks)
    assert "outdir='/old/path'" in out
    assert "CELL_PARAMETERS {angstrom}" in out
    assert "ATOMIC_POSITIONS {angstrom}" in out


def test_update_outdir_only():
    out = update_qe_input(BASE_QE_INPUT, outdir="/new/outdir")

    # Updated outdir line
    assert re.search(r"outdir='/new/outdir'\s*,?\s*$", out, re.MULTILINE)

    # Old outdir should be gone
    assert "outdir='/old/path'" not in out

    # Everything else unchanged (spot checks)
    assert "prefix='old_prefix'" in out
    assert "CELL_PARAMETERS {angstrom}" in out
    assert "ATOMIC_POSITIONS {angstrom}" in out

def test_update_all_fields():
    new_cell = [
        [2.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    new_symbols = ["Si", "Si"]
    new_positions = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    new_prefix = "new_prefix_all"
    new_outdir = "/scratch/new_all"

    out = update_qe_input(
        BASE_QE_INPUT,
        cell=new_cell,
        positions=new_positions,
        symbols=new_symbols,
        prefix=new_prefix,
        outdir=new_outdir,
    )

    # Check CELL_PARAMETERS block
    cell_block = _get_block(out, "CELL_PARAMETERS")
    assert "CELL_PARAMETERS {angstrom}" in cell_block
    assert "2.0000000000000000 0.0000000000000000 0.0000000000000000" in cell_block
    assert "0.0000000000000000 3.0000000000000000 0.0000000000000000" in cell_block
    assert "0.0000000000000000 0.0000000000000000 4.0000000000000000" in cell_block
    assert "1.0000000000000000 0.0000000000000000 0.0000000000000000" not in cell_block

    # Check ATOMIC_POSITIONS block
    pos_block = _get_block(out, "ATOMIC_POSITIONS")
    assert "ATOMIC_POSITIONS {angstrom}" in pos_block
    assert "Si 0.1000000000000000 0.2000000000000000 0.3000000000000000 " in pos_block
    assert "Si 0.4000000000000000 0.5000000000000000 0.6000000000000000 " in pos_block
    assert (
        "Si 0.0000000000000000 0.0000000000000000 0.0000000000000000 "
        not in pos_block
    )

    # Check prefix and outdir updated
    assert "prefix='old_prefix'" not in out
    assert f"prefix='{new_prefix}'" in out

    assert "outdir='/old/path'" not in out
    assert f"outdir='{new_outdir}'" in out


def test_noop_is_idempotent():
    """Calling update_qe_input with no changes should return the same text."""
    out = update_qe_input(BASE_QE_INPUT)
    # Strict equality: structure and whitespace preserved
    assert out == BASE_QE_INPUT

