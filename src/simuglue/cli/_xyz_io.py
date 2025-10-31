#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
from ase import Atoms
from ase.io import read

def _iter_frames(xyz_path: Path, frames: str | int | None) -> Iterable[tuple[int, Atoms]]:
    """
    Yield (frame_index, Atoms) pairs.
    frames:
      - None  -> first frame only (index 0)
      - int   -> that exact frame index
      - "all" -> all frames
    """
    if frames is None:
        yield 0, read(str(xyz_path), format="extxyz", index=0)
        return

    if isinstance(frames, int):
        yield frames, read(str(xyz_path), format="extxyz", index=frames)
        return

    if isinstance(frames, str) and frames.lower() == "all":
        objs = read(str(xyz_path), format="extxyz", index=":")
        # ASE may return a single Atoms or a list; normalize to list
        if isinstance(objs, Atoms):
            yield 0, objs
        else:
            for i, at in enumerate(objs):
                yield i, at
        return

    raise ValueError("frames must be None, an integer, or the string 'all'.")
