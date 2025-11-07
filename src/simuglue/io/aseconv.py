from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, List, OrderedDict
import sys

from ase import Atoms
from ase.io import read, write

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


def _read_atoms(
    src: str,
    fmt: str,
    frames: str | None,
    options: Dict[str, Dict[str, Any]] | None,
) -> List[Atoms]:
    index = frames if frames is not None else ":"
    opts = _get_fmt_opts(fmt, options)

    # stdin allowed for text formats only
    if src == "-":
        if fmt == "traj":
            raise ValueError("Reading traj from stdin is not supported.")

        from io import StringIO
        text = sys.stdin.read()
        fh = StringIO(text)

        if fmt == "lammps-data":
            style = opts.get("style", "full")
            from ase.io.lammpsdata import read_lammps_data
            atoms = read_lammps_data(fh, style=style)
            return [atoms]

        if fmt == "lammps-dump-text":
            return list(read(fh, format="lammps-dump-text", index=index))

        # extxyz / espresso-in / espresso-out
        return list(read(fh, format=fmt, index=index))

    # normal file
    path = Path(src)

    if fmt == "lammps-data":
        style = opts.get("style", "full")
        from ase.io.lammpsdata import read_lammps_data
        atoms = read_lammps_data(path, style=style)
        return [atoms]

    if fmt == "lammps-dump-text":
        return list(read(path, format="lammps-dump-text", index=index))

    if fmt == "extxyz":
        # ASE can infer from .xyz; specifying format is also fine
        return list(read(path, index=index))

    # traj / espresso-in / espresso-out
    return list(read(path, format=fmt, index=index))


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
            from ase.io.lammpsdata import write_lammps_data
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

        from ase.io.lammpsdata import write_lammps_data
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

