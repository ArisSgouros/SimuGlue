from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, List, OrderedDict
import sys

from ase import Atoms
from ase.io import read, write
from ase.io.lammpsdata import read_lammps_data, write_lammps_data
from io import StringIO

SUPPORTED_INPUTS = {
    "extxyz",
    "traj",
    "espresso-in",
    "espresso-out",
    "lammps-data",
    "lammps-dump-text",
}

SUPPORTED_OUTPUTS = SUPPORTED_INPUTS


def _infer_format_from_suffix(path: Path) -> str | None:
    suf = path.suffix.lower()
    if suf == ".xyz":
        return "extxyz"
    if suf == ".traj":
        return "traj"
    if suf in {".pwi", ".in"}:
        return "espresso-in"
    if suf in {".pwo", ".out"}:
        return "espresso-out"
    if suf in {".data", ".lmp", ".lammps"}:
        return "lammps-data"
    if suf in {".dump", ".lammpstrj"}:
        return "lammps-dump-text"
    return None


def _get_fmt_opts(fmt: str, options: Dict[str, Dict[str, Any]] | None) -> Dict[str, Any]:
    """
    Fetch per-format options. If you later want families, you can:
      - also check a prefix key like 'lammps', etc.
    For now: exact match only.
    """
    if not options:
        return {}
    return options.get(fmt, {})

def _parse_index(frames: str | None):
    """Convert CLI --frames into an ASE index object."""
    if frames is None:
        return ":"

    s = frames.strip()

    # If it looks like a slice or complex expression, let ASE handle the string
    if any(c in s for c in [":", ","]):
        return s

    # If it's a plain integer (including negative), use int
    try:
        return int(s)
    except ValueError:
        # Fallback: pass raw string to ASE (it may still understand it)
        return s


def _to_atoms_list(obj) -> List[Atoms]:
    """Normalize ASE read() output to List[Atoms]."""
    if isinstance(obj, Atoms):
        return [obj]
    # ASE already returns a list-like of Atoms for multi-frame; make it concrete.
    return list(obj)


def _make_source(src: str, fmt: str) -> Source:
    """Return a Path or a StringIO depending on src, with format-specific checks."""
    if src == "-":
        if fmt == "traj":
            raise ValueError("Reading traj from stdin is not supported.")
        text = sys.stdin.read()
        return StringIO(text)
    return Path(src)


def _read_from_source(
    source: Source,
    fmt: str,
    index,
    opts: Dict[str, Any],
) -> List[Atoms]:
    """Format-specific read logic from a Path or file-like."""
    # --- LAMMPS data ---
    if fmt == "lammps-data":
        style = opts.get("style", "full")
        units = opts.get("units", "metal")
        if units not in ("metal", "real"):
            raise ValueError(f"Unsupported LAMMPS units for lammps-data: {units}")
        atoms = read_lammps_data(source, style=style, units=units)
        return [atoms]

    # --- LAMMPS dump (text) ---
    if fmt == "lammps-dump-text":
        images = read(source, format="lammps-dump-text", index=index)
        return _to_atoms_list(images)

    # --- extxyz ---
    if fmt == "extxyz":
        # Let ASE infer from suffix for Path; for StringIO we must be explicit.
        if isinstance(source, Path):
            images = read(source, index=index)
        else:
            images = read(source, format="extxyz", index=index)
        return _to_atoms_list(images)

    # --- QE + traj + others handled by ASE ---
    images = read(source, format=fmt, index=index)
    return _to_atoms_list(images)


def _read_atoms(
    src: str,
    fmt: str,
    frames: str | None,
    options: Dict[str, Dict[str, Any]] | None,
) -> List[Atoms]:
    index = _parse_index(frames)
    opts = _get_fmt_opts(fmt, options)
    source = _make_source(src, fmt)
    return _read_from_source(source, fmt, index, opts)

def _write_atoms(
    atoms: Iterable[Atoms],
    dst: str,
    fmt: str,
    options: Dict[str, Dict[str, Any]] | None,
) -> None:
    atoms = list(atoms)
    opts = _get_fmt_opts(fmt, options)

    # stdout for text formats only
    if dst == "-":
        if fmt == "traj":
            raise ValueError("Writing traj to stdout is not supported.")

        if fmt == "lammps-data":
            if len(atoms) != 1:
                raise ValueError("lammps-data output supports a single frame.")
            style = opts.get("style", "full")
            symbols = atoms[0].get_chemical_symbols()
            specorder = opts.get(
                "specorder",
                list(OrderedDict.fromkeys(symbols)),
            )
            write_lammps_data(
                sys.stdout,
                atoms[0],
                atom_style=style,
                specorder=specorder,
                masses=True,      # ← also here
            )
            return

        if fmt == "lammps-dump-text":
            write(sys.stdout, atoms, format="lammps-dump-text")
            return

        if fmt == "espresso-in":
            # QE input requires pseudopotentials etc.; we don't guess here.
            raise ValueError(
                "espresso-in output is not supported by aseconv. "
                "Use a dedicated QE input generator with pseudopotential settings."
            )

        if fmt == "espresso-out":
            # ASE has no writer; keep this explicit for clarity.
            raise ValueError("espresso-out is not a supported output format.")

        # extxyz / espresso-in / espresso-out
        write(sys.stdout, atoms, format=fmt)
        return

    # normal file
    path = Path(dst)

    if fmt == "lammps-data":
        if len(atoms) != 1:
            raise ValueError("lammps-data output supports a single frame.")

        style = opts.get("style", "full")

        # Stable type -> symbol mapping
        symbols = atoms[0].get_chemical_symbols()
        specorder = opts.get(
            "specorder",
            list(OrderedDict.fromkeys(symbols)),
        )

        write_lammps_data(
            path,
            atoms[0],
            atom_style=style,
            specorder=specorder,
            masses=True,          # ← IMPORTANT
        )
        return

    if fmt == "lammps-dump-text":
        write(path, atoms, format="lammps-dump-text")
        return

    if fmt == "espresso-in":
        # QE input requires pseudopotentials etc.; we don't guess here.
        raise ValueError(
            "espresso-in output is not supported by aseconv. "
            "Use a dedicated QE input generator with pseudopotential settings."
        )

    if fmt == "espresso-out":
        # ASE has no writer; keep this explicit for clarity.
        raise ValueError("espresso-out is not a supported output format.")

    # extxyz / traj / espresso-in / espresso-out
    write(path, atoms, format=fmt)


def convert(
    input_path: str,
    output_path: str,
    iformat: str = "auto",
    oformat: str | None = None,
    frames: str | None = None,
    read_opts: Dict[str, Dict[str, Any]] | None = None,
    write_opts: Dict[str, Dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> None:
    # ---- resolve input format ----
    if iformat == "auto":
        if input_path == "-":
            raise ValueError("Cannot infer --iformat from stdin; please set --iformat.")
        inf = _infer_format_from_suffix(Path(input_path))
        if inf is None:
            raise ValueError("Could not infer --iformat from extension; please set it.")
        iformat = inf

    if iformat not in SUPPORTED_INPUTS:
        raise ValueError(f"Unsupported iformat: {iformat}")

    # ---- resolve output format ----
    if oformat is None:
        if output_path == "-":
            raise ValueError("When writing to stdout, --oformat is required.")
        outf = _infer_format_from_suffix(Path(output_path))
        if outf is None:
            raise ValueError("Could not infer --oformat from extension; please set it.")
        oformat = outf

    if oformat not in SUPPORTED_OUTPUTS:
        raise ValueError(f"Unsupported oformat: {oformat}")

    # ---- overwrite safety ----
    if output_path != "-" and Path(output_path).exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path} "
            f"(use --overwrite to allow)."
        )

    # ---- perform conversion ----
    atoms = _read_atoms(
        src=input_path,
        fmt=iformat,
        frames=frames,
        options=read_opts,
    )

    _write_atoms(
        atoms=atoms,
        dst=output_path,
        fmt=oformat,
        options=write_opts,
    )

