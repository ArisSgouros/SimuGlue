# simuglue/qe_cli.py
from __future__ import annotations
import sys, json, argparse, math
from pathlib import Path
from glob import glob
from typing import Iterable, Callable, Optional
from simuglue.io.util_json import sanitize_json, write_json

def _resolve_inputs(specs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for s in specs:
        pats = glob(s) if any(c in s for c in "*?[]") else [s]
        files += [Path(p) for p in pats if Path(p).is_file()]
    seen, out = set(), []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); out.append(p)
    return out

def build_parser(prog: Optional[str] = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, description="Parse QE files to JSON.")
    ap.add_argument("input", nargs="+", help="Files or globs")
    ap.add_argument(
        "-o", "--out", default=None,
        help=("Output file. With many inputs: JSON array to file; "
              "otherwise NDJSON to stdout if omitted.")
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON (indent=2).")
    ap.add_argument("--round-digits", type=int, default=10)
    ap.add_argument("--snap-tol", type=float, default=1e-8)
    return ap

def run(parse_fn: Callable[[Path], dict | None],
        argv: Optional[list[str]] = None,
        prog: Optional[str] = None) -> int:
    """
    parse_fn: function that parses a single QE file -> dict
    argv:     list of CLI args (None => sys.argv[1:])
    prog:     program name to show in help/usage (optional)
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser(prog=prog)
    args = parser.parse_args(argv)

    inputs = _resolve_inputs(args.input)
    if not inputs:
        print("No input files found.", file=sys.stderr); return 2
    indent = 2 if args.pretty else None

    if len(inputs) == 1:
        data = parse_fn(inputs[0])
        if data is None: return 1
        data = sanitize_json(data, ndigits=args.round_digits, snap_tol=args.snap_tol)
        if args.out:
            outp = Path(args.out)
            if outp.is_dir(): outp = outp / f"{inputs[0].stem}.json"
            write_json(outp, data, indent)
        else:
            json.dump(data, sys.stdout, ensure_ascii=False, indent=indent)
            if indent: sys.stdout.write("\n")
        return 0

    frames, status = [], 0
    for p in inputs:
        data = parse_fn(p)
        if data is None:
            status = 1; frames.append({"file": p.name, "error": "parse_failed"})
        else:
            data = sanitize_json(data, ndigits=args.round_digits, snap_tol=args.snap_tol)
            frames.append({"file": p.name, "data": data})

    if args.out:
        write_json(Path(args.out), frames, indent)
    else:
        for fr in frames:
            sys.stdout.write(json.dumps(fr, ensure_ascii=False) + "\n")
    return status
