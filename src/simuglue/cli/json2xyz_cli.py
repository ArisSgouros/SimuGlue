#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.io import write
from simuglue.quantum_espresso.json2xyz import build_atoms_from_json

def main() -> None:
    p = argparse.ArgumentParser(description="Convert JSON (single or multi-snapshot) to EXTXYZ using ASE.")
    p.add_argument("input", help="Path to input JSON file.")
    p.add_argument("-o", "--output", help="Path to output .xyz (default: stdout).")
    p.add_argument("--no-cell", action="store_true", help="Exclude cell (no Lattice in header).")
    p.add_argument("--no-forces", action="store_true", help="Exclude per-atom forces array.")
    p.add_argument("--no-info", action="store_true", help="Exclude global info tags.")
    p.add_argument("--info-keys", nargs="+", help="Only include these info keys (e.g., energy stress virial).")
    p.add_argument("--pbc", choices=["auto", "on", "off"], default="auto",
                   help="Periodic boundary conditions. 'auto' uses PBC if cell is present.")
    args = p.parse_args()

    obj = json.loads(Path(args.input).read_text(encoding="utf-8"))

    frames: List[Atoms] = []
    if isinstance(obj, list):
        # Multi-snapshot: each entry may have {"file": ..., "data": {...}}
        for entry in obj:
            d = entry.get("data", entry) if isinstance(entry, dict) else entry
            src = entry.get("file") if isinstance(entry, dict) else None
            frames.append(
                build_atoms_from_json(
                    d,
                    include_cell=not args.no_cell,
                    include_forces=not args.no_forces,
                    include_info=not args.no_info,
                    info_keys=args.info_keys,
                    pbc_mode=args.pbc,
                    source=src,
                )
            )
    elif isinstance(obj, dict):
        # Single snapshot (old format)
        frames.append(
            build_atoms_from_json(
                obj,
                include_cell=not args.no_cell,
                include_forces=not args.no_forces,
                include_info=not args.no_info,
                info_keys=args.info_keys,
                pbc_mode=args.pbc,
            )
        )
    else:
        raise ValueError("Top-level JSON must be an object or a list of objects.")

    if args.output:
        write(args.output, frames if len(frames) > 1 else frames[0], format="extxyz")
    else:
        from sys import stdout
        write(stdout, frames if len(frames) > 1 else frames[0], format="extxyz")

if __name__ == "__main__":
    main()

