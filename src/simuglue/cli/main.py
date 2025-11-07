# src/simuglue/cli/main.py
from __future__ import annotations
import argparse, sys
from importlib import import_module
from typing import Dict, Tuple

try:
    from argcomplete import autocomplete  # optional; CLI still works without it
except Exception:
    autocomplete = None

# group -> { subcmd: (module_path, help_text, prog_string) }
COMMANDS: Dict[str, Dict[str, Tuple[str, str, str]]] = {
    "qe": {
        "pwo2json": ("simuglue.cli.pwo2json_cli", "QE output (.out/.pwo) → JSON", "sgl qe pwo2json"),
        "pwi2json": ("simuglue.cli.pwi2json_cli", "QE input  (.in/.pwi) → JSON", "sgl qe pwi2json"),
        "json2xyz": ("simuglue.cli.json2xyz_cli", "QE JSON → EXTXYZ",           "sgl qe json2xyz"),
        "xyz2qe":   ("simuglue.cli.xyz2qe_cli",   "EXTXYZ → QE input",           "sgl qe xyz2qe"),
    },
    "transform": {
        "xyz":    ("simuglue.cli.transf_xyz_cli",    "Transform XYZ",         "sgl transform xyz"),
        "lammps": ("simuglue.cli.transf_lammps_cli", "Transform LAMMPS data", "sgl transform lammps"),
    },
    "mech": {
        "strain2defgrad": ("simuglue.cli.strain2defgrad_cli",
                           "Transform strain to deformation gradient",
                           "sgl mech strain2defgrad"),
    },
}

def _run_leaf(mod_path: str, prog: str, extra: list[str]) -> int:
    """Import the leaf CLI and run it. With no args, show its help."""
    inner = import_module(mod_path).main
    argv = extra or ["--help"]
    return inner(argv=argv, prog=prog)

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="sgl", description="SimuGlue command line interface")
    subparsers = parser.add_subparsers(dest="group", required=True)

    # just register group+subcommand names so argcomplete can tab-complete them
    for group, table in COMMANDS.items():
        p_group = subparsers.add_parser(group, help=f"{group} commands")
        sub = p_group.add_subparsers(dest="cmd", required=True)
        for cmd, (mod_path, help_text, prog) in table.items():
            sp = sub.add_parser(cmd, help=help_text, add_help=False)  # no arg defs here
            sp.set_defaults(_mod_path=mod_path, _prog=prog)

    if autocomplete:
        autocomplete(parser)

    args, extra = parser.parse_known_args(argv)
    return _run_leaf(args._mod_path, args._prog, extra)

if __name__ == "__main__":
    raise SystemExit(main())

