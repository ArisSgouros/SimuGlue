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
    "io": [
        {
            "aseconv": ("simuglue.cli.io_aseconv", "Convert file types using ASE modules", "sgl io aseconv"),
            "nepconv": ("simuglue.cli.io_nepconv", "Convert file extxyz between NEP and ASE", "sgl io nepconv"),
            "xyz2pwi": ("simuglue.cli.io_xyz2pwi", "EXTXYZ → QE input", "sgl io xyz2pwi"),
        },
        'io operations / file converters',
    ],
    "build": [
        {
            "supercell": ("simuglue.cli.build_supercell", "Replicate EXTXYZ along a, b and c cell vectors", "sgl build supercell"),
        },
        'build operations',
    ],
    "qe": [
        {
            "pwo2json": ("simuglue.cli.qe_pwo2json", "QE output (.out/.pwo) → JSON", "sgl qe pwo2json"),
            "pwi2json": ("simuglue.cli.qe_pwi2json", "QE input  (.in/.pwi) → JSON", "sgl qe pwi2json"),
            "json2xyz": ("simuglue.cli.qe_json2xyz", "QE JSON → EXTXYZ",           "sgl qe json2xyz"),
        },
        'quantum espresso helpers',
    ],
    "mech": [
        {
            "defgrad": ("simuglue.cli.mech_defgrad", "Transform strain to deformation gradient", "sgl mech defgrad"),
            "strain": ("simuglue.cli.mech_strain", "Transform deformation gradient to strain", "sgl mech strain"),
            "xform": ("simuglue.cli.mech_xform", "Transform XYZ", "sgl mech xform"),
        },
        'mechanics',
    ],
    "wf": [
        {
            "cij": ("simuglue.cli.wf_cij", "Elastic constants (Cij) workflow", "sgl wf cij init|run|parse|post|all"),
        },
        'workflows'
    ]
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
    for group, [table, group_help] in COMMANDS.items():
        p_group = subparsers.add_parser(group, help=f"{group_help}")
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

