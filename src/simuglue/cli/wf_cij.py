# src/simuglue/cli/wf_cij.py
from __future__ import annotations

import argparse
from typing import Sequence

from simuglue.workflow.cij import run_cij, post_cij


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=prog,
        description="Cij workflow: generate/runs cases and post-process elastic constants.",
    )
    sub = p.add_subparsers(dest="action", required=True)

    # sgl wf cij run -c cij.yaml
    prun = sub.add_parser("run", help="Generate and run deformation cases")
    prun.add_argument(
        "-c", "--config",
        default="cij.yaml",
        help="Path to Cij workflow YAML config (default: cij.yaml).",
    )

    # sgl wf cij post -c cij.yaml -o cij.json
    ppost = sub.add_parser("post", help="Post-process finished cases into Cij")
    ppost.add_argument(
        "-c", "--config",
        default="cij.yaml",
        help="Path to Cij workflow YAML config (default: cij.yaml).",
    )
    ppost.add_argument(
        "-o", "--output",
        default=None,
        help="Output JSON file name (overrides output.cij_json in config).",
    )

    return p


def main(
    argv: Sequence[str] | None = None,
    prog: str | None = None,
) -> int:
    parser = build_parser(prog=prog)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.action == "run":
        run_cij(args.config)
        return 0

    if args.action == "post":
        post_cij(args.config, outfile=args.output)
        return 0

    parser.error("No action given.")
    return 2

