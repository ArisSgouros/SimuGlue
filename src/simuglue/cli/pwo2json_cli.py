#!/usr/bin/env python3
import sys
import re
import json
import argparse
from pathlib import Path
from glob import glob
from simuglue.quantum_espresso import pwo2json

# -------------------------- CLI / main -------------------------- #
def _resolve_inputs(globs_or_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in globs_or_paths:
        matches = [Path(p) for p in (glob(pat) if any(c in pat for c in "*?[]") else [pat])]
        files.extend(m for m in matches if m.is_file())
    # Deduplicate while preserving order
    seen = set()
    out = []
    for p in files:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            out.append(p)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Parse Quantum ESPRESSO output (*.out) to JSON."
    )
    p.add_argument(
        "input",
        nargs="+",
        help="Input file(s) or glob(s), e.g. 'scf.out' or 'runs/*/scf.out'",
    )
    p.add_argument(
        "-o",
        "--out",
        dest="out",
        default=None,
        help="Output JSON file (single input) or output directory (multiple inputs). "
             "If omitted, prints to stdout.",
    )
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (indent=2).",
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress non-error warnings on stderr.",
    )

    args = p.parse_args(argv)

    inputs = _resolve_inputs(args.input)
    if not inputs:
        print("No input files found.", file=sys.stderr)
        return 2

    # Silence prints in parser if requested (we already routed warnings to stderr)
    if args.quiet:
        # Nothing to do; the parser already prints to stderr only.
        pass

    indent = 2 if args.pretty else None

    if len(inputs) == 1:
        # Single file mode
        data = pwo2json(inputs[0])
        if data is None:
            return 1

        if args.out:
            out_path = Path(args.out)
            if out_path.exists() and out_path.is_dir():
                # If user gave a dir, name it after input
                out_path = out_path / (inputs[0].stem + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
        else:
            json.dump(data, sys.stdout, ensure_ascii=False, indent=indent)
            if indent is not None:
                sys.stdout.write("\n")
        return 0

    # Multiple files
    if len(inputs) > 1 and not args.out:
        # Stream to stdout
        for path in inputs:
            data = pwo2json(path)
            if data is None:
                print(f'{{"file":"{path}","error":"parse_failed"}}')
            else:
                obj = {"file": path.name, "data": data}
                sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return 0

    # Write to directory or stdout as a combined JSON list
    if len(inputs) > 1 and args.out:
        all_objs = []
        for path in inputs:
            data = pwo2json(path)
            if data is None:
                continue
            all_objs.append({"file": path.name, "data": data})

        # Write to specified output file instead of stdout
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_objs, f, ensure_ascii=False, indent=indent)
            if indent is not None:
                f.write("\n")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())

