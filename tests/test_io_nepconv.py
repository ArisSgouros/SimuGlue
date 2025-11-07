from __future__ import annotations
import subprocess
from pathlib import Path

import pytest


def run_nepconv(tmp_path: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """Run `sgl io nepconv` inside tmp_path, fail loudly on nonzero exit."""
    cmd = ["sgl", "io", "nepconv", *args]
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        input=stdin,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def test_nep_to_ase_single_frame(tmp_path: Path):
    """NEP-style header tokens are converted to ASE/extxyz-style for a single frame."""
    inp = tmp_path / "nep.xyz"
    outp = tmp_path / "ase.xyz"

    # Minimal NEP-like frame:
    # - lowercase lattice=
    # - lowercase properties=
    # - 'force:' in Properties
    inp.write_text(
        "2\n"
        "lattice=\"5 0 0 0 5 0 0 0 5\" properties=species:S:1:pos:R:3:force:R:3\n"
        "H 0.0 0.0 0.0  0.1 0.2 0.3\n"
        "He 1.0 1.0 1.0 -0.1 -0.2 -0.3\n",
        encoding="utf-8",
    )

    run_nepconv(
        tmp_path,
        [
            str(inp),
            "--to",
            "ase",
            "-o",
            str(outp),
        ],
    )

    text = outp.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Structure preserved
    assert lines[0].strip() == "2"

    header = lines[1]
    # lattice -> Lattice
    assert "Lattice=" in header
    assert "lattice=" not in header

    # properties -> Properties
    assert "Properties=" in header
    assert "properties=" not in header

    # force: -> forces:
    assert "forces:" in header
    assert "force:" not in header

    # Atom lines untouched (simple spot check)
    assert "H 0.0 0.0 0.0  0.1 0.2 0.3" in text
    assert "He 1.0 1.0 1.0 -0.1 -0.2 -0.3" in text


def test_ase_nep_roundtrip_multiframe(tmp_path: Path):
    """ASE -> NEP -> ASE roundtrip on a 2-frame XYZ keeps headers consistent."""
    inp = tmp_path / "ase_multi.xyz"
    mid = tmp_path / "nep_multi.xyz"
    outp = tmp_path / "ase_multi_back.xyz"

    inp.write_text(
        # frame 1
        "1\n"
        "Lattice=\"5 0 0 0 5 0 0 0 5\" Properties=species:S:1:pos:R:3:forces:R:3\n"
        "H 0.0 0.0 0.0  0.1 0.2 0.3\n"
        # frame 2
        "1\n"
        "Lattice=\"6 0 0 0 6 0 0 0 6\" Properties=species:S:1:pos:R:3:forces:R:3\n"
        "He 1.0 1.0 1.0 -0.1 -0.2 -0.3\n",
        encoding="utf-8",
    )

    # ASE -> NEP
    run_nepconv(
        tmp_path,
        [
            str(inp),
            "--to",
            "nep",
            "-o",
            str(mid),
        ],
    )

    mid_txt = mid.read_text(encoding="utf-8")
    mid_lines = mid_txt.splitlines()

    # Headers are at lines 1 and 4
    nep_headers = [mid_lines[1], mid_lines[4]]
    for h in nep_headers:
        assert "lattice=" in h
        assert "Lattice=" not in h
        assert "properties=" in h
        assert "Properties=" not in h
        assert "force:" in h
        assert "forces:" not in h

    # NEP -> ASE (back)
    run_nepconv(
        tmp_path,
        [
            str(mid),
            "--to",
            "ase",
            "-o",
            str(outp),
        ],
    )

    back = outp.read_text(encoding="utf-8")
    back_lines = back.splitlines()

    # Headers again at lines 1 and 4
    ase_headers = [back_lines[1], back_lines[4]]
    for h in ase_headers:
        assert "Lattice=" in h
        assert "lattice=" not in h
        assert "Properties=" in h
        assert "properties=" not in h
        assert "forces:" in h
        assert "force:" not in h

def test_nepconv_roundtrip_stdin_stdout(tmp_path: Path):
    """
    ASE -> NEP -> ASE using only stdin/stdout.

    No files passed directly to nepconv; everything is piped through '-'.
    """

    original = (
        "1\n"
        "Lattice=\"5 0 0 0 5 0 0 0 5\" Properties=species:S:1:pos:R:3:forces:R:3\n"
        "H 0.0 0.0 0.0  0.1 0.2 0.3\n"
        "1\n"
        "Lattice=\"6 0 0 0 6 0 0 0 6\" Properties=species:S:1:pos:R:3:forces:R:3\n"
        "He 1.0 1.0 1.0 -0.1 -0.2 -0.3\n"
    )

    # ASE -> NEP via stdin/stdout
    res_nep = run_nepconv(
        tmp_path,
        ["-", "--to", "nep", "-o", "-"],
        stdin=original,
    )
    nep_text = res_nep.stdout

    # NEP -> ASE via stdin/stdout
    res_ase = run_nepconv(
        tmp_path,
        ["-", "--to", "ase", "-o", "-"],
        stdin=nep_text,
    )
    back = res_ase.stdout

    # Roundtrip must preserve content (ignoring trailing whitespace/newlines)
    assert back.strip() == original.strip()

