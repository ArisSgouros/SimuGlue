# simuglue/qe_cli.py
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from glob import glob
from typing import Iterable, Callable, Optional

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

def _write_json(path: Path, obj, indent: int | None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        if indent: f.write("\n")

def run(parse_fn: Callable[[Path], dict | None], argv: Optional[list[str]] = None, prog: Optional[str] = None) -> int:
    ap = argparse.ArgumentParser(prog=prog, description="Parse QE files to JSON.")
    ap.add_argument("input", nargs="+", help="Files or globs")
    ap.add_argument("-o","--out", default=None,
                    help="Output file. With many inputs: JSON array to file; "
                         "otherwise NDJSON to stdout if omitted.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON (indent=2).")
    args = ap.parse_args(argv)

    inputs = _resolve_inputs(args.input)
    if not inputs:
        print("No input files found.", file=sys.stderr); return 2
    indent = 2 if args.pretty else None

    if len(inputs) == 1:
        data = parse_fn(inputs[0])
        if data is None: return 1
        if args.out:
            outp = Path(args.out)
            if outp.is_dir(): outp = outp / f"{inputs[0].stem}.json"
            _write_json(outp, data, indent)
        else:
            json.dump(data, sys.stdout, ensure_ascii=False, indent=indent)
            if indent: sys.stdout.write("\n")
        return 0

    frames, status = [], 0
    for p in inputs:
        obj = parse_fn(p)
        if obj is None:
            status = 1; frames.append({"file": p.name, "error": "parse_failed"})
        else:
            frames.append({"file": p.name, "data": obj})

    if args.out:
        _write_json(Path(args.out), frames, indent)
    else:
        for fr in frames:
            sys.stdout.write(json.dumps(fr, ensure_ascii=False) + "\n")
    return status
