#!/usr/bin/env python3
from __future__ import annotations

import sys, json, argparse
from pathlib import Path
from glob import glob
from typing import Iterable

from simuglue.quantum_espresso import pwi2json


def resolve_inputs(specs: Iterable[str]) -> list[Path]:
    files: list[Path] = []
    for s in specs:
        patt = any(c in s for c in "*?[]")
        files += [Path(p) for p in (glob(s) if patt else [s]) if Path(p).is_file()]
    seen: set[Path] = set()
    out: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp); out.append(p)
    return out


def write_json(path: Path, obj, indent: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
        if indent: f.write("\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Parse Quantum ESPRESSO input (*.in) to JSON.")
    ap.add_argument("input", nargs="+", help="Files or globs, e.g. 'scf.in' or 'runs/*/scf.in'")
    ap.add_argument("-o", "--out", default=None,
                    help="Output file. With multiple inputs, writes a JSON array here. "
                         "If omitted, prints single JSON (1 input) or NDJSON (many) to stdout.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON (indent=2).")
    args = ap.parse_args(argv)

    inputs = resolve_inputs(args.input)
    if not inputs:
        print("No input files found.", file=sys.stderr)
        return 2

    indent = 2 if args.pretty else None

    if len(inputs) == 1:
        data = pwi2json(inputs[0])
        if data is None: return 1
        if args.out:
            outp = Path(args.out)
            if outp.is_dir(): outp = outp / f"{inputs[0].stem}.json"
            write_json(outp, data, indent)
        else:
            json.dump(data, sys.stdout, ensure_ascii=False, indent=indent)
            if indent: sys.stdout.write("\n")
        return 0

    # multiple inputs
    frames = []
    status = 0
    for p in inputs:
        obj = pwi2json(p)
        if obj is None:
            status = 1
            frames.append({"file": p.name, "error": "parse_failed"})
        else:
            frames.append({"file": p.name, "data": obj})

    if args.out:
        write_json(Path(args.out), frames, indent)
    else:
        for fr in frames:
            sys.stdout.write(json.dumps(fr, ensure_ascii=False) + "\n")

    return status


if __name__ == "__main__":
    raise SystemExit(main())

