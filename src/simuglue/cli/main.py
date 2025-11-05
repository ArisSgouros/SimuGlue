# src/simuglue/cli/main.py
from __future__ import annotations
import argparse, sys

def _dispatch_qe(cmd: str, extra: list[str]) -> int:
    if cmd == "pwo2json":
        from .pwo2json_cli import main as inner
        return inner(argv=extra)
    elif cmd == "pwi2json":
        from .pwi2json_cli import main as inner
        return inner(argv=extra)
    elif cmd == "json2xyz":
        from .json2xyz_cli import main as inner
        if not extra or extra in (["-h"], ["--help"]):
            return inner(argv=["--help"], prog="sgl file json2xyz")
        return inner(argv=extra, prog="sgl file json2xyz")
    elif cmd == "xyz2qe":
        from .xyz2qe_cli import main as inner
        if not extra or extra in (["-h"], ["--help"]):
            return inner(argv=["--help"], prog="sgl file xyz2qe")
        return inner(argv=extra, prog="sgl file xyz2qe")
    else:
        print(f"Unknown qe subcommand: {cmd}", file=sys.stderr)
        return 2

def _dispatch_transform(cmd: str, extra: list[str]) -> int:
    if cmd == "xyz":
        from .transf_xyz_cli import main as inner
        return inner(argv=extra)
    elif cmd == "lammps":
        from .transf_lammps_cli import main as inner
        return inner(argv=extra)
    else:
        print(f"Unknown transform subcommand: {cmd}", file=sys.stderr)
        return 2

def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="sgl", description="SimuGlue command line interface")
    sub = parser.add_subparsers(dest="group", required=True)

    # Group: qe
    p_qe = sub.add_parser("qe", help="Quantum Espresso utilities")
    sub_qe = p_qe.add_subparsers(dest="cmd", required=True)
    sub_qe.add_parser("pwo2json", help="QE output (.out/.pwo) → JSON", add_help=False)
    sub_qe.add_parser("pwi2json", help="QE input (.in/.pwi) → JSON", add_help=False)
    sub_qe.add_parser("json2xyz", help="JSON → XYZ", add_help=False)
    sub_qe.add_parser("xyz2qe", help="XYZ → QE input", add_help=False)

    # Group: transform
    p_tr = sub.add_parser("transform", help="Structure transforms")
    sub_tr = p_tr.add_subparsers(dest="cmd", required=True)
    sub_tr.add_parser("xyz", help="Transform XYZ", add_help=False)
    sub_tr.add_parser("lammps", help="Transform LAMMPS data", add_help=False)

    # Parse top-level/group/subcommand; leave the rest for the inner CLIs
    args, extra = parser.parse_known_args(argv)

    if args.group == "qe":
        return _dispatch_qe(args.cmd, extra)
    elif args.group == "transform":
        return _dispatch_transform(args.cmd, extra)
    else:
        parser.error(f"Unknown group: {args.group}")
        return 2  # not reached

if __name__ == "__main__":
    raise SystemExit(main())
