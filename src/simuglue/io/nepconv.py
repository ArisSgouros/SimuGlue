from __future__ import annotations
from pathlib import Path
from typing import List
import sys


def _split_frames(lines: List[str]) -> List[tuple[int, int, int]]:
    """
    Split extended XYZ-style content into frames.

    Returns list of (count_idx, header_idx, atoms_start_idx).

    Assumes:
      - count line is an integer natoms
      - next line is the header/comment line
      - followed by natoms atom lines
    """
    frames = []
    i = 0
    n = len(lines)

    while i < n:
        # skip leading blank lines between frames
        if not lines[i].strip():
            i += 1
            continue

        try:
            natoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError(
                f"Expected atom count at line {i+1}, got: {lines[i]!r}"
            ) from exc

        if i + 1 >= n:
            raise ValueError("Truncated frame: missing header line.")

        header_idx = i + 1
        atoms_start = i + 2
        atoms_end = atoms_start + natoms

        if atoms_end > n:
            raise ValueError("Truncated frame: not enough atom lines.")

        frames.append((i, header_idx, atoms_start))
        i = atoms_end

    return frames


def _nep_to_ase_header(header: str) -> str:
    """
    Convert NEP-style header tokens to ASE/extxyz style.

    NEP conventions (as used in practice):
      lattice=   -> Lattice=
      properties=-> Properties=
      force:     -> forces:
    """
    out = header
    out = out.replace("lattice=", "Lattice=")
    out = out.replace("properties=", "Properties=")
    out = out.replace("force:", "forces:")
    return out


def _ase_to_nep_header(header: str) -> str:
    """
    Convert ASE/extxyz-style header tokens to NEP style.

      Lattice=   -> lattice=
      Properties=-> properties=
      forces:    -> force:
    """
    out = header
    out = out.replace("Lattice=", "lattice=")
    out = out.replace("Properties=", "properties=")
    out = out.replace("forces:", "force:")
    return out


def convert_text(text: str, to: str) -> str:
    """
    Convert between NEP-style and ASE/extxyz-style extended XYZ headers.

    Parameters
    ----------
    text : str
        Input XYZ text.
    to : {'ase', 'nep'}
        Target style.
    """
    target = to.lower()
    if target not in {"ase", "nep"}:
        raise ValueError("Argument 'to' must be 'ase' or 'nep'.")

    lines = text.splitlines(keepends=True)
    frames = _split_frames(lines)

    for _, header_idx, _ in frames:
        header = lines[header_idx]
        if target == "ase":
            lines[header_idx] = _nep_to_ase_header(header)
        else:
            lines[header_idx] = _ase_to_nep_header(header)

    return "".join(lines)


def convert_stream(
    input_path: str,
    output_path: str,
    to: str,
) -> None:
    """
    High-level I/O wrapper:
      - input_path: filename or '-' for stdin
      - output_path: filename or '-' for stdout
      - to: 'ase' or 'nep'
    """
    # read
    if input_path == "-":
        text = sys.stdin.read()
    else:
        text = Path(input_path).read_text(encoding="utf-8")

    out = convert_text(text, to=to)

    # write
    if output_path == "-":
        sys.stdout.write(out)
    else:
        Path(output_path).write_text(out, encoding="utf-8")

